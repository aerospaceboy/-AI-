from pathlib import Path

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
