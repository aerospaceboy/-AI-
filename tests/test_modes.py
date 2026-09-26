from pathlib import Path

from src.exporter import export_ocr, export_transcript
from src.llm import NoLLMProvider
from src.summarizer import generate_summary


def test_no_audio_and_no_ocr_exports(tmp_path: Path) -> None:
    export_transcript(tmp_path / "transcript.md", [])
    export_ocr(tmp_path / "ocr.md", [])
    assert "未检测到音频" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "未启用 OCR" in (tmp_path / "ocr.md").read_text(encoding="utf-8")


def test_no_llm_summary(tmp_path: Path) -> None:
    summary = generate_summary([], tmp_path / "chunks", NoLLMProvider())
    assert summary.startswith("# 视频总结")
    assert "[未确认]" in summary
