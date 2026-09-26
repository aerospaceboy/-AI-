from pathlib import Path

import cv2
import numpy as np

from src.scenes import _frame_skip_for_fps, extract_keyframes


def test_scene_sampling_frame_skip() -> None:
    assert _frame_skip_for_fps(30.0, 2.0) == 14
    assert _frame_skip_for_fps(5.0, 2.0) == 1
    assert _frame_skip_for_fps(2.0, 2.0) == 0
    assert _frame_skip_for_fps(30.0, 0.0) == 0


def test_keyframe_writes_to_unicode_path(tmp_path: Path) -> None:
    video = tmp_path / "测试 视频.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        # The installed OpenCV build may not provide an encoder; integration smoke covers FFmpeg.
        return
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
    writer.release()
    output = tmp_path / "中文帧"
    records = extract_keyframes(video, output, duration=1, max_gap=30)
    assert records
    assert (output / records[0]["frame"]).is_file()


def test_static_video_reuses_file_but_keeps_visual_points(tmp_path: Path) -> None:
    video = tmp_path / "static.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        return
    for _ in range(15):
        writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
    writer.release()
    output = tmp_path / "frames"
    records = extract_keyframes(video, output, duration=3, max_gap=1)
    assert [x["timestamp"] for x in records] == [0.0, 1, 2]
    assert len({x["frame"] for x in records}) == 1
    assert records[1]["reused_frame"] is True
