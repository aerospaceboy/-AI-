from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import require_ffmpeg


class AudioDecodeError(RuntimeError):
    """The video has an audio stream, but FFmpeg could not decode it."""


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    ffmpeg = require_ffmpeg()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode == 0:
        return audio_path.exists() and audio_path.stat().st_size > 44
    if audio_path.exists():
        audio_path.unlink()
    if not _has_audio_stream(video_path):
        # A video without an audio stream is valid input.
        return False
    detail = (completed.stderr or "").strip().splitlines()
    raise AudioDecodeError(
        "FFmpeg could not decode the audio track"
        + (f": {detail[-1]}" if detail else f" (exit code {completed.returncode})")
    )


def _has_audio_stream(video_path: Path) -> bool:
    """Best-effort probe via PyAV, which ships with faster-whisper.
    Returns True when the answer cannot be determined, so real failures surface."""
    try:
        import av
    except ImportError:
        return True
    try:
        with av.open(str(video_path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return True
