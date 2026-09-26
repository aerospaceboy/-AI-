from pathlib import Path
import sys

from src.asr import model_location
from src.utils import require_ffmpeg


def test_frozen_app_uses_bundled_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    bundled = tmp_path / "ffmpeg-win-x86_64.exe"
    bundled.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert require_ffmpeg() == str(bundled)


def test_frozen_app_uses_bundled_base_model(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "models" / "base"
    model_dir.mkdir(parents=True)
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (model_dir / name).touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Video2AI.exe"))
    assert model_location("base") == str(model_dir)
    assert model_location("small") == "small"
    (model_dir / "model.bin").unlink()
    assert model_location("base") == "base"
