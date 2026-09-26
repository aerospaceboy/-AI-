from __future__ import annotations

import logging
import platform
import sys
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from . import __version__
from .asr import resolve_device, transcribe
from .audio import extract_audio
from .chatgpt_export import export_chatgpt_package
from .exporter import export_handoff, export_ocr, export_package_readme, export_timeline, export_transcript
from .llm import NoLLMProvider, provider_from_options
from .ocr import create_backend, run_ocr
from .report import export_report
from .scenes import extract_keyframes
from .summarizer import basic_summary, generate_summary
from .timeline import build_timeline
from .utils import read_json, utc_now, write_json, write_text
from .video import probe_video


LOG = logging.getLogger("video2ai")


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
    overwrite: bool = False
    force_asr: bool = False
    force_frames: bool = False
    force_ocr: bool = False
    force_timeline: bool = False
    force_summary: bool = False


def _state(path: Path) -> dict[str, Any]:
    return read_json(path, {}) or {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, state)


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


def run(options: Options) -> Path:
    video = options.video.expanduser().resolve()
    output = options.output.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    frames = output / "frames"
    chunks = raw / "chunks"
    raw.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    chunks.mkdir(exist_ok=True)
    state_path = raw / "state.json"
    state = _state(state_path)
    force_all = options.overwrite
    source_identity = {
        "path": str(video), "size": video.stat().st_size,
        "mtime_ns": video.stat().st_mtime_ns,
    }
    source_signature = _signature(source_identity)

    LOG.info("[1/7] Reading video metadata...")
    info = probe_video(video)
    _mark_done(state, "metadata", source_signature, source=source_identity)
    _save_state(state_path, state)

    actual_device, _ = resolve_device(options.device)
    initial_prompt: str | None = None
    if options.initial_prompt_file:
        prompt_path = options.initial_prompt_file.expanduser().resolve()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Initial prompt file not found: {prompt_path}")
        initial_prompt = prompt_path.read_text(encoding="utf-8").strip() or None
    asr_signature = _signature({
        "version": 2, "source": source_signature, "model": options.whisper_model,
        "language": options.language, "hotwords": options.hotwords,
        "initial_prompt": initial_prompt, "beam_size": options.beam_size,
        "batch_size": options.batch_size,
    })
    transcript_path = raw / "transcript.json"
    transcript_data = read_json(transcript_path, None)
    has_audio = bool(transcript_data.get("has_audio", True)) if isinstance(transcript_data, dict) else True
    asr_rebuilt = False
    if transcript_data is not None and _cached(
        state, "asr", asr_signature, transcript_path, allow_legacy=True
    ) and not (force_all or options.force_asr):
        LOG.info("[SKIP] Transcript already exists.")
    else:
        LOG.info("[2/7] Extracting / reading audio...")
        audio_path = raw / "audio.wav"
        has_audio = extract_audio(video, audio_path)
        if has_audio:
            LOG.info("[3/7] Running speech recognition...")
            transcript_data = transcribe(
                audio_path, options.whisper_model, options.language, options.device,
                hotwords=options.hotwords, initial_prompt=initial_prompt,
                beam_size=options.beam_size, batch_size=options.batch_size,
            )
            transcript_data["has_audio"] = True
            try:
                audio_path.unlink()
            except OSError:
                pass
        else:
            LOG.warning("[WARN] No audio stream detected. Continuing without ASR.")
            transcript_data = {
                "language": options.language, "model": options.whisper_model,
                "has_audio": False, "segments": [],
            }
        write_json(transcript_path, transcript_data)
        _mark_done(state, "asr", asr_signature, model=options.whisper_model)
        _save_state(state_path, state)
        asr_rebuilt = True
    transcript_data = transcript_data or {"segments": [], "has_audio": False}
    segments = transcript_data.get("segments", [])
    export_transcript(output / "transcript.md", segments)

    scenes_path = raw / "scenes.json"
    legacy_frames_signature = _signature({
        "version": 2, "source": source_signature, "max_frame_gap": float(options.max_frame_gap),
    })
    frames_signature = _signature({
        "version": 3, "source": source_signature, "max_frame_gap": float(options.max_frame_gap),
        "scene_sample_fps": float(options.scene_sample_fps),
    })
    scenes = read_json(scenes_path, None)
    frame_state = state.get("frames")
    migrated_frames = False
    if (
        _frames_complete(output, scenes) and isinstance(frame_state, dict)
        and frame_state.get("signature") == legacy_frames_signature
        and not (force_all or options.force_frames)
    ):
        # A full-frame v0.2 scan is richer than the new sampled scan, so it is safe to retain.
        _mark_done(
            state, "frames", frames_signature, max_frame_gap=options.max_frame_gap,
            scene_sample_fps="legacy-full-scan", migrated=True,
        )
        _save_state(state_path, state)
        migrated_frames = True
    frames_rebuilt = False
    if _frames_complete(output, scenes) and _cached(state, "frames", frames_signature, scenes_path) and not (force_all or options.force_frames):
        LOG.info("[SKIP] Scene detection already completed.")
    else:
        LOG.info("[4/7] Detecting keyframes...")
        new_scenes = extract_keyframes(
            video, frames, info["duration"], options.max_frame_gap,
            options.scene_sample_fps, show_progress=True,
            filename_prefix=f"frame_{uuid.uuid4().hex[:8]}",
        )
        if not _frames_complete(output, new_scenes):
            raise RuntimeError("Keyframe extraction produced no complete visual evidence")
        scenes = new_scenes
        write_json(scenes_path, scenes)
        _mark_done(
            state, "frames", frames_signature, max_frame_gap=options.max_frame_gap,
            scene_sample_fps=options.scene_sample_fps,
        )
        _save_state(state_path, state)
        frames_rebuilt = True
    scenes = scenes or []

    ocr_path = raw / "ocr.json"
    legacy_ocr_signature = _signature({
        "version": 2, "frames": legacy_frames_signature, "disabled": options.no_ocr,
        "language": options.language,
    })
    ocr_signature = _signature({
        "version": 2, "frames": frames_signature, "disabled": options.no_ocr,
        "language": options.language,
    })
    ocr_records = read_json(ocr_path, None)
    ocr_engine = "none"
    ocr_state = state.get("ocr")
    migrated_ocr = False
    if (
        migrated_frames and ocr_records is not None and isinstance(ocr_state, dict)
        and ocr_state.get("signature") == legacy_ocr_signature
        and not (force_all or options.force_ocr or frames_rebuilt)
    ):
        ocr_engine = str(ocr_state.get("engine", state.get("ocr_engine", "cached")))
        _mark_done(state, "ocr", ocr_signature, engine=ocr_engine, migrated=True)
        _save_state(state_path, state)
        migrated_ocr = True
    ocr_rebuilt = False
    if options.no_ocr:
        LOG.info("[SKIP] OCR disabled.")
        ocr_records = []
        write_json(ocr_path, ocr_records)
        _mark_done(state, "ocr", ocr_signature, engine="none")
        _save_state(state_path, state)
    elif (
        ocr_records is not None and _cached(state, "ocr", ocr_signature, ocr_path)
        and isinstance(state.get("ocr"), dict)
        and state["ocr"].get("engine") != "none"
        and not (force_all or options.force_ocr or frames_rebuilt)
    ):
        LOG.info("[SKIP] OCR already completed.")
        ocr_engine = str(state.get("ocr_engine", "cached"))
    else:
        LOG.info("[5/7] Running OCR...")
        backend = create_backend(options.language, actual_device == "cuda")
        ocr_engine = backend.name
        ocr_records = run_ocr(output, scenes, backend)
        write_json(ocr_path, ocr_records)
        _mark_done(state, "ocr", ocr_signature, engine=ocr_engine)
        state["ocr_engine"] = ocr_engine
        _save_state(state_path, state)
        ocr_rebuilt = True
    ocr_records = ocr_records or []
    export_ocr(output / "ocr.md", ocr_records)

    timeline_path = output / "timeline.json"
    legacy_timeline_signature = _signature({
        "version": 2, "asr": asr_signature, "frames": legacy_frames_signature,
        "ocr": legacy_ocr_signature, "chunk_seconds": float(options.timeline_chunk_seconds),
    })
    timeline_signature = _signature({
        "version": 2, "asr": asr_signature, "frames": frames_signature,
        "ocr": ocr_signature, "chunk_seconds": float(options.timeline_chunk_seconds),
    })
    timeline = read_json(timeline_path, None)
    timeline_state = state.get("timeline")
    if (
        migrated_frames and migrated_ocr and timeline is not None and isinstance(timeline_state, dict)
        and timeline_state.get("signature") == legacy_timeline_signature
        and not (force_all or options.force_timeline or asr_rebuilt or frames_rebuilt or ocr_rebuilt)
    ):
        _mark_done(
            state, "timeline", timeline_signature,
            chunk_seconds=options.timeline_chunk_seconds, migrated=True,
        )
        _save_state(state_path, state)
    timeline_rebuilt = False
    if timeline is not None and _cached(state, "timeline", timeline_signature, timeline_path) and not (force_all or options.force_timeline or asr_rebuilt or frames_rebuilt or ocr_rebuilt):
        LOG.info("[SKIP] Timeline already exists.")
    else:
        LOG.info("[6/7] Building timeline...")
        timeline = build_timeline(
            segments, ocr_records, scenes, info["duration"], options.timeline_chunk_seconds
        )
        write_json(timeline_path, timeline)
        _mark_done(state, "timeline", timeline_signature, chunk_seconds=options.timeline_chunk_seconds)
        _save_state(state_path, state)
        timeline_rebuilt = True
    timeline = timeline or []
    export_timeline(output / "timeline.md", timeline)

    summary_path = output / "summary.md"
    provider = provider_from_options(options.no_llm, options.base_url, options.model, options.api_key)
    summary_signature = _signature({
        "version": 3, "timeline": timeline_signature, "provider": provider.cache_key,
    })
    summary_state = state.get("summary")
    summary_failed = (
        not isinstance(provider, NoLLMProvider)
        and isinstance(summary_state, dict) and summary_state.get("llm") is False
    )
    summary_rebuilt = False
    if _cached(state, "summary", summary_signature, summary_path) and not (force_all or options.force_summary or timeline_rebuilt or summary_failed):
        LOG.info("[SKIP] Summary already exists.")
    else:
        LOG.info("[7/7] Generating AI summary...")
        used_llm = not isinstance(provider, NoLLMProvider)
        llm_error: str | None = None
        try:
            summary = generate_summary(timeline, chunks, provider, force_all or options.force_summary)
        except Exception as exc:
            LOG.warning("[WARN] LLM unavailable. Generating basic package without AI summary. (%s)", exc)
            summary = basic_summary(timeline)
            used_llm = False
            llm_error = str(exc)
        write_text(summary_path, summary)
        details: dict[str, Any] = {"llm": used_llm, "provider": provider.cache_key}
        if llm_error:
            details["llm_error"] = llm_error
        _mark_done(state, "summary", summary_signature, **details)
        _save_state(state_path, state)
        summary_rebuilt = True

    metadata: dict[str, Any] = {
        "source_video": str(video), **info,
        "language": transcript_data.get("language") or options.language,
        "has_audio": bool(transcript_data.get("has_audio", has_audio)),
        "asr_engine": "faster-whisper", "asr_model": options.whisper_model,
        "ocr_engine": ocr_engine, "device": transcript_data.get("device", actual_device), "created_at": utc_now(),
        "frame_count": len({x["frame"] for x in scenes}),
        "visual_point_count": len(scenes), "python_version": platform.python_version(),
        "platform": platform.platform(), "video2ai_version": __version__,
    }
    write_json(output / "metadata.json", metadata)
    export_package_readme(output / "README.md", video, info["duration"])
    export_handoff(output / "AI_HANDOFF.md")
    export_report(output, metadata, timeline)
    chatgpt_signature = _signature({
        "version": 2, "frames": frames_signature, "timeline": timeline_signature,
        "summary": summary_signature,
    })
    chatgpt_manifest = output / "chatgpt" / "manifest.json"
    if (
        _cached(state, "chatgpt_package", chatgpt_signature, chatgpt_manifest)
        and _package_complete(output) and not (frames_rebuilt or summary_rebuilt)
    ):
        LOG.info("[SKIP] ChatGPT upload package already exists.")
    else:
        export_chatgpt_package(output, scenes)
        _mark_done(state, "chatgpt_package", chatgpt_signature)
        active_frames = {scene["frame"] for scene in scenes}
        for stale in frames.glob("frame_*.jpg"):
            if stale.name not in active_frames:
                try:
                    stale.unlink()
                except OSError as exc:
                    LOG.warning("[WARN] Could not remove stale generated frame %s: %s", stale, exc)
    _save_state(state_path, state)
    return output
