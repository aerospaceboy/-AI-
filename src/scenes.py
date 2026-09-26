from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .utils import format_timestamp


LOG = logging.getLogger("video2ai")


def _frame_skip_for_fps(video_fps: float, sample_fps: float) -> int:
    if video_fps <= 0 or sample_fps <= 0 or sample_fps >= video_fps:
        return 0
    return max(0, int(round(video_fps / sample_fps)) - 1)


def _scene_candidates(
    video_path: Path,
    threshold: float = 27.0,
    sample_fps: float = 2.0,
    show_progress: bool = False,
) -> list[float]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        logging.getLogger("pyscenedetect").setLevel(logging.WARNING)
        video = open_video(str(video_path))
        frame_skip = _frame_skip_for_fps(float(video.frame_rate), sample_fps)
        effective_fps = float(video.frame_rate) / (frame_skip + 1)
        LOG.info(
            "Scene scan: %.2f FPS sample (source %.2f FPS, skip %d).",
            effective_fps, float(video.frame_rate), frame_skip,
        )
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video=video, frame_skip=frame_skip, show_progress=show_progress)
        scenes = manager.get_scene_list(start_in_scene=True)
        return [scene[0].get_seconds() for scene in scenes[1:]]
    except Exception as exc:
        LOG.warning("[WARN] PySceneDetect unavailable/failed; using max-gap frames only: %s", exc)
        return []


def _is_duplicate(previous: Any, current: Any) -> bool:
    """Only drop virtually identical images; small code changes should survive."""
    import cv2
    import numpy as np

    if previous is None:
        return False
    a = cv2.resize(previous, (320, 180), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, (320, 180), interpolation=cv2.INTER_AREA)
    difference = float(np.mean(cv2.absdiff(a, b)))
    return difference < 0.65


def extract_keyframes(
    video_path: Path,
    frames_dir: Path,
    duration: float,
    max_gap: float = 30.0,
    scene_sample_fps: float = 2.0,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for keyframe extraction.") from exc

    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[float, str] = {0.0: "start"}
    for timestamp in _scene_candidates(video_path, sample_fps=scene_sample_fps, show_progress=show_progress):
        candidates[round(timestamp, 3)] = "scene_change"
    point = max_gap
    while point < duration:
        rounded = round(point, 3)
        if rounded not in candidates:
            candidates[rounded] = "max_gap"
        point += max_gap

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    records: list[dict[str, Any]] = []
    saved_count = 0
    previous = None
    previous_filename: str | None = None
    previous_timestamp: float | None = None
    try:
        for timestamp, reason in sorted(candidates.items()):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                LOG.warning("[WARN] Could not read frame at %s; skipped.", format_timestamp(timestamp))
                continue
            if _is_duplicate(previous, frame):
                # Keep the visual evidence point without storing the same JPEG again.
                if previous_filename is not None:
                    records.append({
                        "frame": previous_filename,
                        "timestamp": timestamp,
                        "timestamp_text": format_timestamp(timestamp),
                        "reason": reason,
                        "reused_frame": True,
                        "duplicate_of_timestamp": previous_timestamp,
                    })
                continue
            saved_count += 1
            filename = f"frame_{saved_count:04d}.jpg"
            target = frames_dir / filename
            # cv2.imwrite is not Unicode-safe on some Windows builds.
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            try:
                if encoded_ok:
                    encoded.tofile(str(target))
            except OSError:
                encoded_ok = False
            if not encoded_ok or not target.exists():
                LOG.warning("[WARN] Could not save %s; skipped.", target)
                saved_count -= 1
                continue
            records.append({
                "frame": filename,
                "timestamp": timestamp,
                "timestamp_text": format_timestamp(timestamp),
                "reason": reason,
                "reused_frame": False,
            })
            previous = frame
            previous_filename = filename
            previous_timestamp = timestamp
    finally:
        capture.release()
    return records
