from pathlib import Path

from src.exporter import export_srt


def test_export_srt_format_and_numbering(tmp_path: Path) -> None:
    segments = [
        {"start": 0.0, "end": 2.5, "text": "定义 FIFO 参数"},
        {"start": 3.0, "end": 3.0, "text": "第二句"},  # zero duration gets bumped
        {"start": 5.0, "end": 6.0, "text": "   "},  # empty text is skipped
        {"start": 6.0, "end": 8.0, "text": "最后一句"},
    ]
    destination = tmp_path / "subtitle.srt"
    export_srt(destination, segments)
    text = destination.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\n定义 FIFO 参数" in text
    assert "00:00:03,000 --> 00:00:03,500" in text
    cues = text.strip().split("\n\n")
    assert [cue.splitlines()[0] for cue in cues] == ["1", "2", "3"]
    assert cues[-1].splitlines()[-1] == "最后一句"


def test_export_srt_empty_transcript(tmp_path: Path) -> None:
    destination = tmp_path / "subtitle.srt"
    export_srt(destination, [])
    assert "无可用字幕" in destination.read_text(encoding="utf-8")
