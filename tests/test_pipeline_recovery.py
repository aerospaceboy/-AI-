from pathlib import Path

import pytest

from src import pipeline
from src.llm import LLMProvider
from src.ocr import NoOCRBackend, OCRBackend


def _setup_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, list[int]]:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video")
    output = tmp_path / "video_ai"
    calls = [0]
    monkeypatch.setattr(pipeline, "probe_video", lambda _: {
        "duration": 1.0, "width": 64, "height": 48, "resolution": "64x48",
        "fps": 5.0, "video_frame_count": 5,
    })
    monkeypatch.setattr(pipeline, "resolve_device", lambda _: ("cpu", "int8"))
    monkeypatch.setattr(pipeline, "extract_audio", lambda *_: False)

    def extract(_, frames_dir, __, ___, ____, show_progress=False, filename_prefix="frame"):
        calls[0] += 1
        filename = f"{filename_prefix}_0001.jpg"
        (frames_dir / filename).write_bytes(b"frame data")
        return [{"frame": filename, "timestamp": 0.0, "reason": "start"}]

    monkeypatch.setattr(pipeline, "extract_keyframes", extract)
    return video, output, calls


def test_failed_frame_rebuild_keeps_previous_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)
    options = pipeline.Options(video=video, output=output, no_ocr=True, no_llm=True)
    pipeline.run(options)
    previous = pipeline.read_json(output / "raw" / "scenes.json")[0]["frame"]
    assert (output / "frames" / previous).is_file()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline, "extract_keyframes", fail)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run(pipeline.Options(
            video=video, output=output, no_ocr=True, no_llm=True, force_frames=True,
        ))
    assert (output / "frames" / previous).is_file()
    assert pipeline.read_json(output / "raw" / "scenes.json")[0]["frame"] == previous


def test_forced_asr_refreshes_downstream_even_with_same_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)
    calls = [0]
    original = pipeline.build_timeline

    def count_timeline(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_timeline", count_timeline)
    pipeline.run(pipeline.Options(video=video, output=output, no_ocr=True, no_llm=True))
    assert calls[0] == 1
    pipeline.run(pipeline.Options(
        video=video, output=output, no_ocr=True, no_llm=True, force_asr=True,
    ))
    assert calls[0] == 2
    assert (output / "chatgpt" / "upload_bundle.zip").is_file()


def test_missing_bundle_invalidates_package_cache(tmp_path: Path) -> None:
    package = tmp_path / "chatgpt"
    package.mkdir()
    pipeline.write_json(package / "manifest.json", {"contact_sheets": []})
    assert not pipeline._package_complete(tmp_path)


def test_ocr_retries_after_backend_becomes_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)

    class WorkingOCR(OCRBackend):
        name = "test-ocr"

        def recognize(self, image_path: Path) -> list[str]:
            return ["recovered text"]

    available = [False]
    monkeypatch.setattr(
        pipeline, "create_backend",
        lambda *_: WorkingOCR() if available[0] else NoOCRBackend(),
    )
    options = pipeline.Options(video=video, output=output, no_llm=True)
    pipeline.run(options)
    assert pipeline.read_json(output / "raw" / "ocr.json")[0]["text"] == []

    available[0] = True
    pipeline.run(options)
    assert pipeline.read_json(output / "raw" / "ocr.json")[0]["text"] == ["recovered text"]
    assert pipeline.read_json(output / "raw" / "state.json")["ocr"]["engine"] == "test-ocr"


def test_llm_retries_after_request_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)

    class RecoveringLLM(LLMProvider):
        calls = 0
        available = False

        @property
        def cache_key(self) -> str:
            return "test-model"

        def summarize(self, content: str, instruction: str) -> str:
            self.calls += 1
            if not self.available:
                raise RuntimeError("service unavailable")
            return "generated summary"

    provider = RecoveringLLM()
    monkeypatch.setattr(pipeline, "provider_from_options", lambda *_: provider)
    options = pipeline.Options(video=video, output=output, no_ocr=True)
    pipeline.run(options)
    assert pipeline.read_json(output / "raw" / "state.json")["summary"]["llm"] is False

    provider.available = True
    pipeline.run(options)
    assert (output / "summary.md").read_text(encoding="utf-8").strip() == "generated summary"
    assert pipeline.read_json(output / "raw" / "state.json")["summary"]["llm"] is True


def test_audio_decode_failure_retries_on_next_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)

    def broken_extract(*_args, **_kwargs):
        raise pipeline.AudioDecodeError("corrupt audio track")

    monkeypatch.setattr(pipeline, "extract_audio", broken_extract)
    options = pipeline.Options(video=video, output=output, no_ocr=True, no_llm=True)
    pipeline.run(options)
    transcript = pipeline.read_json(output / "raw" / "transcript.json")
    assert transcript["asr_error"] == "corrupt audio track"
    assert pipeline.read_json(output / "raw" / "state.json")["asr"]["error"] == "corrupt audio track"
    assert pipeline.read_json(output / "metadata.json")["asr_error"] == "corrupt audio track"

    monkeypatch.setattr(pipeline, "extract_audio", lambda *_: False)
    pipeline.run(options)
    transcript = pipeline.read_json(output / "raw" / "transcript.json")
    assert "asr_error" not in transcript
    assert "asr_error" not in pipeline.read_json(output / "metadata.json")


def test_vlm_stage_retries_only_missing_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    video, output, _ = _setup_video(monkeypatch, tmp_path)

    class FakeVLM:
        model = "fake-vl"
        calls = 0

        @property
        def cache_key(self) -> str:
            return "fake-vl"

        def describe_image(self, image_path: Path, instruction: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")
            return "IDE 编辑界面截图"

    provider = FakeVLM()
    monkeypatch.setattr(pipeline, "build_vlm_provider", lambda options: provider)
    options = pipeline.Options(video=video, output=output, no_ocr=True, no_llm=True, no_vlm=False)
    pipeline.run(options)
    assert provider.calls == 1  # the only frame failed once
    frame = pipeline.read_json(output / "raw" / "scenes.json")[0]["frame"]
    assert pipeline.read_json(output / "raw" / "vlm.json") == {}

    pipeline.run(options)  # retry: only the missing frame is described
    assert provider.calls == 2
    assert pipeline.read_json(output / "raw" / "vlm.json") == {frame: "IDE 编辑界面截图"}
    timeline = pipeline.read_json(output / "timeline.json")
    assert timeline[0]["descriptions"][0]["text"] == "IDE 编辑界面截图"

    pipeline.run(options)  # complete now: the provider is not called again
    assert provider.calls == 2
