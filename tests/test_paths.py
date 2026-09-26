from pathlib import Path

from src.utils import default_output_path, read_json, write_json


def test_windows_and_chinese_path_name() -> None:
    value = default_output_path(Path("D:/视频/测试 视频.mp4"))
    assert value.name == "测试 视频_ai"


def test_json_unicode_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "中文.json"
    write_json(path, {"文字": "参数"})
    assert read_json(path) == {"文字": "参数"}
