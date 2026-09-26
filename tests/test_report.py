from pathlib import Path

from src.report import export_report, render_report


def test_report_escapes_untrusted_video_text_and_only_links_package_frames() -> None:
    html = render_report(
        {"duration": 12, "frame_count": 1},
        [{
            "start": 0, "end": 12, "title": '<img src=x onerror="alert(1)">',
            "speech": ["</script><script>alert(2)</script>"],
            "ocr": ["WIDTH < 8 && FLAG=1"],
            "evidence": [
                {"timestamp": 0, "frame": "frames/frame_0001.jpg", "reason": "start"},
                {"timestamp": 1, "frame": "../outside.jpg", "reason": "invalid"},
            ],
        }],
        "# Summary <b>untrusted</b>",
    )
    assert '<img src=x onerror="alert(1)">' not in html
    assert "</script><script>alert(2)</script>" not in html
    assert "&lt;/script&gt;&lt;script&gt;alert(2)&lt;/script&gt;" in html
    assert "WIDTH &lt; 8 &amp;&amp; FLAG=1" in html
    assert 'src="frames/frame_0001.jpg"' in html
    assert "../outside.jpg" not in html
    assert 'id="search"' in html


def test_report_exports_single_offline_file(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("已完成字幕与 OCR", encoding="utf-8")
    destination = export_report(tmp_path, {"duration": 2, "frame_count": 0}, [])
    text = destination.read_text(encoding="utf-8")
    assert destination == tmp_path / "report.html"
    assert "已完成字幕与 OCR" in text
    assert "没有可用的时间轴内容" in text
    assert "https://" not in text
