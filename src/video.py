from __future__ import annotations

from pathlib import Path
from typing import Any


def probe_video(video_path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Run: pip install -r requirements.txt") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = count / fps if fps > 0 else 0.0
        return {
            "duration": round(duration, 3),
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "video_frame_count": count,
        }
    finally:
        capture.release()
