from pathlib import Path
import os
import sys

import src.desktop as desktop
from src.desktop import build_command, load_api_key_from_project


def test_desktop_command_preserves_unicode_paths_and_flags() -> None:
    video = Path(r"D:\视频\演示.mp4")
    output = Path(r"D:\资料包\演示")
    command = build_command(video, output, "large-v3", "cuda", True, False)
    assert str(video) in command
    assert str(output) in command
    assert command[-4:] == ["--device", "cuda", "--ocr", "--no-llm"]
    assert "--whisper-model" in command
    assert "--config" in command


def test_portable_command_uses_its_own_executable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Video2AI\Video2AI.exe")
    monkeypatch.setattr(desktop, "CONFIG_PATH", Path(r"C:\Video2AI\video2ai.yaml"))
    video = Path(r"D:\视频\演示.mp4")
    command = build_command(video, None, "base", "cpu", False, False)
    assert command[:3] == [r"C:\Video2AI\Video2AI.exe", "--worker", str(video)]
    assert command[-4:] == ["--device", "cpu", "--no-ocr", "--no-llm"]


def test_key_file_detection_includes_notepad_bom(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "OPENAI_API_KEY.txt").write_text("sk-test-key\n", encoding="utf-8-sig")
    assert desktop.load_api_key_from_project(tmp_path) is True
    assert os.environ["OPENAI_API_KEY"] == "sk-test-key"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert desktop.load_api_key_from_project(tmp_path / "missing") is False
    assert load_api_key_from_project(tmp_path / "missing") is False


def test_progress_patterns_match_worker_log_lines() -> None:
    summary = desktop.SUMMARY_PROGRESS_PATTERN.match("Summary: chunk 03/15 generated.")
    assert summary.groups() == ("03", "15")
    ocr = desktop.OCR_PROGRESS_PATTERN.match("OCR progress: 25 unique frames processed")
    assert ocr.group(1) == "25"
    vision = desktop.VISION_PROGRESS_PATTERN.match("Vision: 4/12 frames described.")
    assert vision.groups() == ("4", "12")
    batch = desktop.BATCH_PATTERN.match("=== [2/5] lecture.mp4 ===")
    assert batch.groups() == ("2", "5", "lecture.mp4")
    # Scene-scan lines must not be mistaken for the other progress kinds.
    assert not desktop.SUMMARY_PROGRESS_PATTERN.match("  Detected: 0 | Progress: 100%|")
    assert not desktop.OCR_PROGRESS_PATTERN.match("Summary: chunk 01/01 generated.")


def test_build_command_vlm_flag(tmp_path: Path) -> None:
    with_flag = build_command(tmp_path / "v.mp4", None, "base", "cpu", False, False, vlm=True)
    assert with_flag[-1] == "--vlm"
    without = build_command(tmp_path / "v.mp4", None, "base", "cpu", False, False, vlm=False)
    assert without[-1] == "--no-vlm"
    default = build_command(tmp_path / "v.mp4", None, "base", "cpu", False, False)
    assert "--vlm" not in default and "--no-vlm" not in default


def test_write_env_key_preserves_other_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text('VIDEO2AI_MODEL=qwen3.8-flash\nOPENAI_API_KEY=old\n', encoding="utf-8")
    desktop.write_env_key(env_path, "sk-new")
    content = env_path.read_text(encoding="utf-8")
    assert "VIDEO2AI_MODEL=qwen3.8-flash" in content
    assert "OPENAI_API_KEY=sk-new" in content
    assert "old" not in content
    desktop.write_env_key(env_path, "")
    assert "OPENAI_API_KEY" not in env_path.read_text(encoding="utf-8")
