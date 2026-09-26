from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from src.chatgpt_export import export_chatgpt_package


def _write_unicode_jpg(path: Path) -> None:
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(path))


def test_chatgpt_package_contains_combined_evidence_and_sheet(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    _write_unicode_jpg(frames / "frame_0001.jpg")
    (tmp_path / "summary.md").write_text("# 视频总结\n\n测试", encoding="utf-8")
    (tmp_path / "timeline.md").write_text("# 视频时间轴", encoding="utf-8")
    (tmp_path / "transcript.md").write_text("# 视频完整字幕", encoding="utf-8")
    export_chatgpt_package(tmp_path, [{
        "frame": "frame_0001.jpg", "timestamp": 1.5, "reason": "start",
    }])
    assert (tmp_path / "chatgpt" / "contact_sheet_001.jpg").is_file()
    evidence = (tmp_path / "chatgpt" / "VIDEO_EVIDENCE.md").read_text(encoding="utf-8")
    assert "测试" in evidence
    assert (tmp_path / "chatgpt" / "manifest.json").is_file()
    assert (tmp_path / "frames.zip").is_file()
    bundle = tmp_path / "chatgpt" / "upload_bundle.zip"
    assert bundle.is_file()
    with ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "CHATGPT_HANDOFF.md", "VIDEO_EVIDENCE.md", "manifest.json",
            "contact_sheet_001.jpg",
        }
