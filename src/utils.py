from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger("video2ai")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        force=True,
    )
    # Keep model download libraries from flooding the user-facing progress log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def format_timestamp(seconds: float, include_ms: bool = False) -> str:
    seconds = max(0.0, float(seconds or 0))
    whole = int(seconds)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if include_ms:
        millis = int(round((seconds - whole) * 1000))
        if millis == 1000:
            whole += 1
            hours, rem = divmod(whole, 3600)
            minutes, secs = divmod(rem, 60)
            millis = 0
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    temp.replace(path)


def require_ffmpeg() -> str:
    executable = None
    if getattr(sys, "frozen", False):
        bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        bundled = next(bundle_dir.glob("ffmpeg*.exe"), None)
        if bundled is not None:
            executable = str(bundled)
    if not executable:
        executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    return executable


def default_output_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}_ai")


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def normalized_lines(lines: Iterable[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        clean = re.sub(r"\s+", " ", str(line)).strip()
        if clean:
            result.append(clean)
    return result


def env_value(name: str, fallback: str | None = None) -> str | None:
    return os.environ.get(name) or fallback
