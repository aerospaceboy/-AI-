from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from . import __version__
from .asr import resolve_device, transcribe
from .audio import AudioDecodeError, extract_audio
from .chatgpt_export import export_chatgpt_package
from .exporter import export_handoff, export_ocr, export_package_readme, export_srt, export_timeline, export_transcript
from .llm import NoLLMProvider, OpenAICompatibleProvider, provider_from_options
from .ocr import create_backend, run_ocr
from .report import export_report
from .scenes import extract_keyframes
from .summarizer import basic_summary, generate_summary
from .timeline import build_timeline
from .utils import read_json, utc_now, write_json, write_text
from .video import probe_video


LOG = logging.getLogger("video2ai")

VLM_INSTRUCTION = (
    "用一到两句中文客观描述这张视频截图的关键内容：出现的软件/界面名称、代码或文档要点、"
    "数据或图表结论。只描述画面上确实可见的内容，看不清或不确定的不要编造。"
)


def build_vlm_provider(options: Options) -> OpenAICompatibleProvider:
    base_url = (
        options.vlm_base_url or os.getenv("VIDEO2AI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = options.vlm_model or os.getenv("VIDEO2AI_VLM_MODEL") or "qwen3-vl-flash"
    return OpenAICompatibleProvider(base_url, model, options.api_key or os.getenv("OPENAI_API_KEY"))


@dataclass
class Options:
    video: Path
    output: Path
    language: str | None = None
    whisper_model: str = "small"
    hotwords: str | None = None
    initial_prompt_file: Path | None = None
    beam_size: int = 5
    batch_size: int = 0
    device: str = "auto"
    max_frame_gap: float = 30.0
    scene_sample_fps: float = 2.0
    timeline_chunk_seconds: float = 120.0
    no_ocr: bool = False
    no_llm: bool = False
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    llm_workers: int = 4
    no_vlm: bool = True
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_workers: int = 4
    overwrite: bool = False
    force_asr: bool = False
    force_frames: bool = False
    force_ocr: bool = False
    force_timeline: bool = False
    force_summary: bool = False


@dataclass(frozen=True)
class StageSignatures:
    """Cache fingerprints for each stage, plus the pre-sampling legacy variants
    used to migrate v0.1/v0.2 packages without forcing an expensive rebuild."""

    asr: str
    frames: str
    legacy_frames: str
    ocr: str
    legacy_ocr: str
    timeline: str
    legacy_timeline: str
    vlm: str | None = None


@dataclass
class RunContext:
    """Values shared by the pipeline stages; each stage records its outcome here."""

    options: Options
    video: Path
    output: Path
    state_path: Path
    state: dict[str, Any]
    source: dict[str, Any]
    source_signature: str
    info: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    asr_rebuilt: bool = False
    frames_rebuilt: bool = False
    frames_migrated: bool = False
    ocr_rebuilt: bool = False
    ocr_migrated: bool = False
    vlm_rebuilt: bool = False
    timeline_rebuilt: bool = False
    summary_rebuilt: bool = False
    summary_provider: str = "none"

    @property
    def raw(self) -> Path:
        return self.output / "raw"

    @property
    def frames_dir(self) -> Path:
        return self.output / "frames"

    @property
    def chunks(self) -> Path:
        return self.raw / "chunks"

    def save_state(self) -> None:
        write_json(self.state_path, self.state)


def _state(path: Path) -> dict[str, Any]:
    return read_json(path, {}) or {}


def _signature(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _cached(
    state: dict[str, Any], stage: str, signature: str, artifact: Path,
    allow_legacy: bool = False,
) -> bool:
    if not artifact.exists():
        return False
    entry = state.get(stage)
    if entry == "done" and allow_legacy:
        # Upgrade v0.1 state without forcing an expensive one-time rebuild.
        state[stage] = {"status": "done", "signature": signature, "migrated": True}
        return True
    return bool(isinstance(entry, dict) and entry.get("status") == "done" and entry.get("signature") == signature)


def _mark_done(state: dict[str, Any], stage: str, signature: str, **details: Any) -> None:
    state[stage] = {"status": "done", "signature": signature, **details}


def _frames_complete(output: Path, scenes: Any) -> bool:
    return bool(
        isinstance(scenes, list) and scenes
        and all(
            isinstance(item, dict) and isinstance(item.get("frame"), str)
            and Path(item["frame"]).name == item["frame"]
            and (output / "frames" / item["frame"]).is_file()
            and (output / "frames" / item["frame"]).stat().st_size > 0
            for item in scenes
        )
    )


def _package_complete(output: Path) -> bool:
    package = output / "chatgpt"
    manifest = read_json(package / "manifest.json", None)
    if not isinstance(manifest, dict):
        return False
    required = [
        output / "frames.zip", package / "upload_bundle.zip",
        package / "CHATGPT_HANDOFF.md", package / "VIDEO_EVIDENCE.md",
    ]
    for sheet in manifest.get("contact_sheets", []):
        if not isinstance(sheet, dict) or not isinstance(sheet.get("file"), str):
            return False
        required.append(package / sheet["file"])
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        with ZipFile(output / "frames.zip") as frame_archive:
            if not frame_archive.namelist():
                return False
        with ZipFile(package / "upload_bundle.zip") as upload_archive:
            names = set(upload_archive.namelist())
            expected = {"CHATGPT_HANDOFF.md", "VIDEO_EVIDENCE.md", "manifest.json"}
            expected.update(sheet["file"] for sheet in manifest.get("contact_sheets", []))
            return expected <= names
    except (BadZipFile, OSError):
        return False


def _load_initial_prompt(options: Options) -> str | None:
    if not options.initial_prompt_file:
        return None
    prompt_path = options.initial_prompt_file.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Initial prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip() or None


def _compute_signatures(
    options: Options, source_signature: str, initial_prompt: str | None,
    vlm_provider: OpenAICompatibleProvider | None = None,
) -> StageSignatures:
    asr = _signature({
        "version": 2, "source": source_signature, "model": options.whisper_model,
        "language": options.language, "hotwords": options.hotwords,
        "initial_prompt": initial_prompt, "beam_size": options.beam_size,
        "batch_size": options.batch_size,
    })
    legacy_frames = _signature({
        "version": 2, "source": source_signature, "max_frame_gap": float(options.max_frame_gap),
    })
    frames = _signature({
        "version": 3, "source": source_signature, "max_frame_gap": float(options.max_frame_gap),
        "scene_sample_fps": float(options.scene_sample_fps),
    })
    legacy_ocr = _signature({
        "version": 2, "frames": legacy_frames, "disabled": options.no_ocr,
        "language": options.language,
    })
    ocr = _signature({
        "version": 2, "frames": frames, "disabled": options.no_ocr,
        "language": options.language,
    })
    # When vision descriptions are disabled, keep the exact v2 timeline payload
    # so existing packages stay cached instead of rebuilding once.
    vlm = None
    timeline_payload: dict[str, Any] = {
        "version": 2, "asr": asr, "frames": frames,
        "ocr": ocr, "chunk_seconds": float(options.timeline_chunk_seconds),
    }
    if vlm_provider is not None:
        vlm = _signature({"version": 1, "frames": frames, "provider": vlm_provider.cache_key})
        timeline_payload = {
            "version": 3, "asr": asr, "frames": frames, "ocr": ocr,
            "chunk_seconds": float(options.timeline_chunk_seconds), "vlm": vlm,
        }
    legacy_timeline = _signature({
        "version": 2, "asr": asr, "frames": legacy_frames,
        "ocr": legacy_ocr, "chunk_seconds": float(options.timeline_chunk_seconds),
    })
    return StageSignatures(
        asr=asr, frames=frames, legacy_frames=legacy_frames, ocr=ocr,
        legacy_ocr=legacy_ocr, timeline=_signature(timeline_payload),
        legacy_timeline=legacy_timeline, vlm=vlm,
    )


def _stage_metadata(ctx: RunContext) -> dict[str, Any]:
    LOG.info("[1/8] Reading video metadata...")
    info = probe_video(ctx.video)
    _mark_done(ctx.state, "metadata", ctx.source_signature, source=ctx.source)
    ctx.save_state()
    return info


def _stage_transcript(ctx: RunContext, initial_prompt: str | None, signature: str) -> dict[str, Any]:
    options = ctx.options
    path = ctx.raw / "transcript.json"
    data = read_json(path, None)
    pending_error = isinstance(data, dict) and bool(data.get("asr_error"))
    if (
        data is not None and not pending_error
        and _cached(ctx.state, "asr", signature, path, allow_legacy=True)
        and not (options.overwrite or options.force_asr)
    ):
        LOG.info("[SKIP] Transcript already exists.")
    else:
        LOG.info("[2/8] Extracting / reading audio...")
        audio_path = ctx.raw / "audio.wav"
        try:
            has_audio = extract_audio(ctx.video, audio_path)
        except AudioDecodeError as exc:
            # A damaged audio track must not silently pass for "no audio";
            # the recorded error makes the next run retry this stage.
            LOG.warning("[WARN] Audio decoding failed; continuing without ASR. (%s)", exc)
            data = {
                "language": options.language, "model": options.whisper_model,
                "has_audio": True, "segments": [], "asr_error": str(exc),
            }
        else:
            if has_audio:
                LOG.info("[3/8] Running speech recognition...")
                data = transcribe(
                    audio_path, options.whisper_model, options.language, options.device,
                    hotwords=options.hotwords, initial_prompt=initial_prompt,
                    beam_size=options.beam_size, batch_size=options.batch_size,
                )
                data["has_audio"] = True
                try:
                    audio_path.unlink()
                except OSError:
                    pass
            else:
                LOG.warning("[WARN] No audio stream detected. Continuing without ASR.")
                data = {
                    "language": options.language, "model": options.whisper_model,
                    "has_audio": False, "segments": [],
                }
        write_json(path, data)
        details: dict[str, Any] = {"model": options.whisper_model}
        if isinstance(data, dict) and data.get("asr_error"):
            details["error"] = data["asr_error"]
        _mark_done(ctx.state, "asr", signature, **details)
        ctx.save_state()
        ctx.asr_rebuilt = True
    export_transcript(ctx.output / "transcript.md", data.get("segments", []))
    return data


def _stage_keyframes(ctx: RunContext, signatures: StageSignatures) -> list[dict[str, Any]]:
    options = ctx.options
    scenes_path = ctx.raw / "scenes.json"
    scenes = read_json(scenes_path, None)
    forced = options.overwrite or options.force_frames
    if (
        not forced and _frames_complete(ctx.output, scenes)
        and isinstance(ctx.state.get("frames"), dict)
        and ctx.state["frames"].get("signature") == signatures.legacy_frames
    ):
        # A full-frame v0.2 scan is richer than the new sampled scan, so it is safe to retain.
        _mark_done(
            ctx.state, "frames", signatures.frames, max_frame_gap=options.max_frame_gap,
            scene_sample_fps="legacy-full-scan", migrated=True,
        )
        ctx.save_state()
        ctx.frames_migrated = True
    if (
        not forced and _frames_complete(ctx.output, scenes)
        and _cached(ctx.state, "frames", signatures.frames, scenes_path)
    ):
        LOG.info("[SKIP] Scene detection already completed.")
        return scenes or []
    LOG.info("[4/8] Detecting keyframes...")
    new_scenes = extract_keyframes(
        ctx.video, ctx.frames_dir, ctx.info["duration"], options.max_frame_gap,
        options.scene_sample_fps, show_progress=True,
        filename_prefix=f"frame_{uuid.uuid4().hex[:8]}",
    )
    if not _frames_complete(ctx.output, new_scenes):
        raise RuntimeError("Keyframe extraction produced no complete visual evidence")
    write_json(scenes_path, new_scenes)
    _mark_done(
        ctx.state, "frames", signatures.frames, max_frame_gap=options.max_frame_gap,
        scene_sample_fps=options.scene_sample_fps,
    )
    ctx.save_state()
    ctx.frames_rebuilt = True
    return new_scenes


def _stage_ocr(
    ctx: RunContext, scenes: list[dict[str, Any]], signatures: StageSignatures,
) -> tuple[list[dict[str, Any]], str]:
    options = ctx.options
    ocr_path = ctx.raw / "ocr.json"
    records = read_json(ocr_path, None)
    engine = "none"
    forced = options.overwrite or options.force_ocr
    if (
        ctx.frames_migrated and records is not None
        and isinstance(ctx.state.get("ocr"), dict)
        and ctx.state["ocr"].get("signature") == signatures.legacy_ocr
        and not (forced or ctx.frames_rebuilt)
    ):
        engine = str(ctx.state["ocr"].get("engine", ctx.state.get("ocr_engine", "cached")))
        _mark_done(ctx.state, "ocr", signatures.ocr, engine=engine, migrated=True)
        ctx.save_state()
        ctx.ocr_migrated = True
    if options.no_ocr:
        LOG.info("[SKIP] OCR disabled.")
        records = []
        write_json(ocr_path, records)
        _mark_done(ctx.state, "ocr", signatures.ocr, engine="none")
        ctx.save_state()
    elif (
        records is not None and _cached(ctx.state, "ocr", signatures.ocr, ocr_path)
        and isinstance(ctx.state.get("ocr"), dict)
        and ctx.state["ocr"].get("engine") != "none"
        and not (forced or ctx.frames_rebuilt)
    ):
        LOG.info("[SKIP] OCR already completed.")
        engine = str(ctx.state.get("ocr_engine", "cached"))
    else:
        LOG.info("[5/8] Running OCR...")
        backend = create_backend(options.language, ctx.device == "cuda")
        engine = backend.name
        records = run_ocr(ctx.output, scenes, backend)
        write_json(ocr_path, records)
        _mark_done(ctx.state, "ocr", signatures.ocr, engine=engine)
        ctx.state["ocr_engine"] = engine
        ctx.save_state()
        ctx.ocr_rebuilt = True
    records = records or []
    export_ocr(ctx.output / "ocr.md", records)
    return records, engine


def _stage_vlm(
    ctx: RunContext, scenes: list[dict[str, Any]], signatures: StageSignatures,
    provider: OpenAICompatibleProvider | None,
) -> dict[str, str]:
    options = ctx.options
    if provider is None:
        LOG.info("[SKIP] Vision descriptions disabled.")
        return {}
    path = ctx.raw / "vlm.json"
    cached: dict[str, str] = {}
    data = read_json(path, None)
    if isinstance(data, dict):
        cached = {str(key): str(value) for key, value in data.items()}
    unique_frames = [scene["frame"] for scene in scenes if not scene.get("reused_frame")]
    if (
        all(frame in cached for frame in unique_frames)
        and _cached(ctx.state, "vlm", signatures.vlm, path)
        and not options.overwrite
    ):
        LOG.info("[SKIP] Vision descriptions already exist.")
        return cached
    LOG.info("[6/8] Running vision model on keyframes...")
    # Failed frames stay missing from vlm.json, so the next run retries only
    # those instead of paying for the whole set again.
    targets = unique_frames if options.overwrite else [
        frame for frame in unique_frames if frame not in cached
    ]
    total = len(unique_frames)
    described: dict[str, str] = {}
    workers = max(1, min(int(options.vlm_workers), len(targets) or 1))

    def describe(frame: str) -> tuple[str, str] | None:
        try:
            return frame, provider.describe_image(ctx.frames_dir / frame, VLM_INSTRUCTION)
        except Exception as exc:
            LOG.warning("[WARN] Vision description failed for %s (%s)", frame, exc)
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(describe, targets):
            if result is not None:
                described[result[0]] = result[1]
                LOG.info("Vision: %d/%d frames described.", len(described), len(targets))
    merged = {} if options.overwrite else dict(cached)
    merged.update(described)
    write_json(path, merged)
    _mark_done(ctx.state, "vlm", signatures.vlm, model=provider.model, described=len(merged))
    ctx.save_state()
    ctx.vlm_rebuilt = bool(described)
    return merged


def _stage_timeline(
    ctx: RunContext, segments: list[dict[str, Any]], ocr_records: list[dict[str, Any]],
    scenes: list[dict[str, Any]], signatures: StageSignatures,
    descriptions: dict[str, str],
) -> list[dict[str, Any]]:
    options = ctx.options
    path = ctx.output / "timeline.json"
    timeline = read_json(path, None)
    forced = options.overwrite or options.force_timeline
    upstream_rebuilt = ctx.asr_rebuilt or ctx.frames_rebuilt or ctx.ocr_rebuilt or ctx.vlm_rebuilt
    if (
        ctx.frames_migrated and ctx.ocr_migrated and timeline is not None
        and isinstance(ctx.state.get("timeline"), dict)
        and ctx.state["timeline"].get("signature") == signatures.legacy_timeline
        and not (forced or upstream_rebuilt)
    ):
        _mark_done(
            ctx.state, "timeline", signatures.timeline,
            chunk_seconds=options.timeline_chunk_seconds, migrated=True,
        )
        ctx.save_state()
    if (
        timeline is not None and _cached(ctx.state, "timeline", signatures.timeline, path)
        and not (forced or upstream_rebuilt)
    ):
        LOG.info("[SKIP] Timeline already exists.")
        return timeline or []
    LOG.info("[7/8] Building timeline...")
    timeline = build_timeline(
        segments, ocr_records, scenes, ctx.info["duration"], options.timeline_chunk_seconds,
        descriptions=descriptions,
    )
    write_json(path, timeline)
    _mark_done(ctx.state, "timeline", signatures.timeline, chunk_seconds=options.timeline_chunk_seconds)
    ctx.save_state()
    ctx.timeline_rebuilt = True
    export_timeline(ctx.output / "timeline.md", timeline)
    return timeline


def _stage_summary(ctx: RunContext, timeline: list[dict[str, Any]], timeline_signature: str) -> str:
    options = ctx.options
    provider = provider_from_options(options.no_llm, options.base_url, options.model, options.api_key)
    ctx.summary_provider = provider.cache_key
    signature = _signature({"version": 3, "timeline": timeline_signature, "provider": provider.cache_key})
    path = ctx.output / "summary.md"
    summary_state = ctx.state.get("summary")
    summary_failed = (
        not isinstance(provider, NoLLMProvider)
        and isinstance(summary_state, dict) and summary_state.get("llm") is False
    )
    if (
        _cached(ctx.state, "summary", signature, path)
        and not (options.overwrite or options.force_summary or ctx.timeline_rebuilt or summary_failed)
    ):
        LOG.info("[SKIP] Summary already exists.")
        return signature
    LOG.info("[8/8] Generating AI summary...")
    used_llm = not isinstance(provider, NoLLMProvider)
    llm_error: str | None = None
    try:
        summary = generate_summary(
            timeline, ctx.chunks, provider,
            options.overwrite or options.force_summary, max_workers=options.llm_workers,
        )
    except Exception as exc:
        LOG.warning("[WARN] LLM unavailable. Generating basic package without AI summary. (%s)", exc)
        summary = basic_summary(timeline)
        used_llm = False
        llm_error = str(exc)
    write_text(path, summary)
    details: dict[str, Any] = {"llm": used_llm, "provider": provider.cache_key}
    if llm_error:
        details["llm_error"] = llm_error
    _mark_done(ctx.state, "summary", signature, **details)
    ctx.save_state()
    ctx.summary_rebuilt = True
    return signature


def _stage_chatgpt_package(
    ctx: RunContext, scenes: list[dict[str, Any]], signatures: StageSignatures,
    summary_signature: str,
) -> None:
    signature = _signature({
        "version": 2, "frames": signatures.frames, "timeline": signatures.timeline,
        "summary": summary_signature,
    })
    manifest = ctx.output / "chatgpt" / "manifest.json"
    if (
        _cached(ctx.state, "chatgpt_package", signature, manifest)
        and _package_complete(ctx.output) and not (ctx.frames_rebuilt or ctx.summary_rebuilt)
    ):
        LOG.info("[SKIP] ChatGPT upload package already exists.")
        return
    export_chatgpt_package(ctx.output, scenes)
    _mark_done(ctx.state, "chatgpt_package", signature)
    active_frames = {scene["frame"] for scene in scenes}
    for stale in ctx.frames_dir.glob("frame_*.jpg"):
        if stale.name not in active_frames:
            try:
                stale.unlink()
            except OSError as exc:
                LOG.warning("[WARN] Could not remove stale generated frame %s: %s", stale, exc)


def run(options: Options) -> Path:
    video = options.video.expanduser().resolve()
    output = options.output.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    (output / "frames").mkdir(parents=True, exist_ok=True)
    (output / "raw" / "chunks").mkdir(parents=True, exist_ok=True)
    state_path = output / "raw" / "state.json"
    source = {
        "path": str(video), "size": video.stat().st_size,
        "mtime_ns": video.stat().st_mtime_ns,
    }
    ctx = RunContext(
        options=options, video=video, output=output, state_path=state_path,
        state=_state(state_path), source=source, source_signature=_signature(source),
    )

    ctx.info = _stage_metadata(ctx)
    ctx.device, _ = resolve_device(options.device)
    initial_prompt = _load_initial_prompt(options)
    vlm_provider = None if options.no_vlm else build_vlm_provider(options)
    signatures = _compute_signatures(options, ctx.source_signature, initial_prompt, vlm_provider)

    transcript = _stage_transcript(ctx, initial_prompt, signatures.asr)
    transcript = transcript or {"segments": [], "has_audio": False}
    segments = transcript.get("segments", [])
    if segments:
        export_srt(ctx.output / "subtitle.srt", segments)

    scenes = _stage_keyframes(ctx, signatures)
    ocr_records, ocr_engine = _stage_ocr(ctx, scenes, signatures)
    descriptions = _stage_vlm(ctx, scenes, signatures, vlm_provider)
    timeline = _stage_timeline(ctx, segments, ocr_records, scenes, signatures, descriptions)
    summary_signature = _stage_summary(ctx, timeline, signatures.timeline)

    metadata: dict[str, Any] = {
        "source_video": str(ctx.video), **ctx.info,
        "language": transcript.get("language") or options.language,
        "has_audio": bool(transcript.get("has_audio", True)),
        "asr_engine": "faster-whisper", "asr_model": options.whisper_model,
        "ocr_engine": ocr_engine, "device": transcript.get("device", ctx.device), "created_at": utc_now(),
        "frame_count": len({x["frame"] for x in scenes}),
        "visual_point_count": len(scenes), "python_version": platform.python_version(),
        "platform": platform.platform(), "video2ai_version": __version__,
        "summary_provider": ctx.summary_provider,
    }
    if vlm_provider is not None:
        metadata["vlm_model"] = vlm_provider.model
    if transcript.get("asr_error"):
        metadata["asr_error"] = transcript["asr_error"]
    write_json(output / "metadata.json", metadata)
    export_package_readme(output / "README.md", video, ctx.info["duration"])
    export_handoff(output / "AI_HANDOFF.md")
    export_report(output, metadata, timeline)
    _stage_chatgpt_package(ctx, scenes, signatures, summary_signature)
    ctx.save_state()
    return output
