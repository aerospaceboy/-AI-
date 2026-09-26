from src.timeline import build_timeline


def test_timeline_fuses_sources_and_classifies() -> None:
    transcript = [{"start": 1, "end": 2, "text": "定义 FIFO 参数"}]
    ocr = [{"timestamp": 2, "frame": "frame_0001.jpg", "text": [
        "parameter DATA_WIDTH = 8;", "vlog fifo.v", "Error: compile failed"
    ]}]
    scenes = [{"timestamp": 2, "frame": "frame_0001.jpg"}]
    result = build_timeline(transcript, ocr, scenes, 10)
    assert result[0]["speech"] == ["定义 FIFO 参数"]
    assert "parameter DATA_WIDTH = 8;" in result[0]["code"]
    assert "vlog fifo.v" in result[0]["commands"]
    assert "Error: compile failed" in result[0]["errors"]


def test_empty_sources() -> None:
    result = build_timeline([], [], [], 10)
    assert len(result) == 1
    assert result[0]["speech"] == []


def test_zero_duration_and_empty_sources() -> None:
    assert build_timeline([], [], [], 0) == []
