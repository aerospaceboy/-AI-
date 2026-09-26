from pathlib import Path
import sys

from src.utils import require_ffmpeg


def test_frozen_app_uses_bundled_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    bundled = tmp_path / "ffmpeg-win-x86_64.exe"
    bundled.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert require_ffmpeg() == str(bundled)
