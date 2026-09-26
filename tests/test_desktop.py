from pathlib import Path
import sys

import src.desktop as desktop
from src.desktop import build_command


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
