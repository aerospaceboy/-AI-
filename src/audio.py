from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import require_ffmpeg


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    ffmpeg = require_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        # A video without an audio stream is valid input.
        if audio_path.exists():
            audio_path.unlink()
        return False
    return audio_path.exists() and audio_path.stat().st_size > 44
