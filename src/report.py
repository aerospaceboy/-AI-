"""Self-contained offline HTML report for a generated evidence package."""

from __future__ import annotations

from html import escape
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from .utils import format_timestamp, write_text


STYLE = """
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#172333;background:#f4f7fa}
*{box-sizing:border-box}body{margin:0}.layout{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:100vh}
aside{position:sticky;top:0;height:100vh;overflow:auto;background:#10243a;color:#e9f1f8;padding:25px 18px}
.brand{font-size:19px;font-weight:750;letter-spacing:.02em;margin:0 0 7px}.aside-note{color:#aabbd0;font-size:13px;line-height:1.5}
.nav-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#91a8c1;margin:28px 0 10px}
.nav-link{display:block;color:#e9f1f8;text-decoration:none;padding:9px 10px;border-radius:9px;font-size:13px;line-height:1.35}
.nav-link:hover,.nav-link:focus{background:#28445f}.nav-time{display:block;color:#9cc8e3;font-variant-numeric:tabular-nums;font-size:12px}
main{max-width:1120px;width:100%;padding:34px 38px 70px;margin:auto}.eyebrow{color:#286b8d;font-weight:700;font-size:12px;letter-spacing:.1em;text-transform:uppercase}
h1{font-size:clamp(28px,4vw,42px);line-height:1.16;margin:8px 0 10px}h2{font-size:20px;margin:0}h3{font-size:15px;margin:0 0 12px}
.lede{color:#5d6b7b;line-height:1.6;margin:0 0 24px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 23px}
.metric,.panel,.chapter{background:#fff;border:1px solid #dfe7ef;border-radius:14px;box-shadow:0 3px 14px #18314e09}
.metric{padding:15px}.metric strong{display:block;font-size:22px;font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:#627384}
.toolbar{display:flex;align-items:center;gap:12px;margin:0 0 20px}.search{width:100%;padding:13px 15px;font:inherit;border:1px solid #b8c8d7;border-radius:10px;background:#fff}
.search:focus{outline:2px solid #70b4d9;outline-offset:2px}.count{white-space:nowrap;color:#5c7183;font-size:13px}
.panel{padding:20px;margin:0 0 18px}.summary{font-family:inherit;font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;margin:14px 0 0;color:#34495b}
.chapter{margin:0 0 18px;padding:23px;scroll-margin-top:18px}.chapter-head{display:flex;gap:16px;align-items:flex-start;margin-bottom:18px}
.time{flex:0 0 142px;color:#217092;font-size:13px;font-weight:750;font-variant-numeric:tabular-nums;padding-top:3px}
.chapter-title{line-height:1.4;overflow-wrap:anywhere}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.content-block{background:#f6f9fc;border:1px solid #e3ebf2;border-radius:10px;padding:16px;min-width:0}
.line{margin:0 0 10px;line-height:1.65;font-size:14px;overflow-wrap:anywhere}.line:last-child{margin-bottom:0}
.empty{color:#8493a3;font-size:13px}.frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:16px}
figure{margin:0;border:1px solid #e0e8f0;border-radius:10px;overflow:hidden;background:#f6f9fc}figure a{display:block}figure img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#091827}
figcaption{padding:8px 10px;font-size:12px;color:#526477;font-variant-numeric:tabular-nums}footer{color:#748595;font-size:12px;line-height:1.6;margin-top:24px}
[hidden]{display:none!important}@media(max-width:900px){.layout{display:block}aside{position:static;height:auto;max-height:210px}.metrics{grid-template-columns:repeat(2,1fr)}main{padding:24px 18px 54px}}
@media(max-width:620px){.columns{grid-template-columns:1fr}.chapter-head{display:block}.time{display:block;margin-bottom:8px}.toolbar{display:block}.count{display:block;margin-top:8px}}
"""

SCRIPT = """
const search = document.getElementById('search');
const chapters = [...document.querySelectorAll('.chapter')];
const count = document.getElementById('result-count');
search.addEventListener('input', () => {
  const query = search.value.trim().toLocaleLowerCase();
  let shown = 0;
  for (const chapter of chapters) {
    const match = !query || chapter.textContent.toLocaleLowerCase().includes(query);
    chapter.hidden = !match;
    if (match) shown += 1;
  }
  count.textContent = `${shown} / ${chapters.length} 章节`;
});
"""


def _safe_frame_href(value: Any) -> str | None:
    """Only link to files directly inside this package's frames directory."""
    name = str(value).replace("\\", "/")
    path = PurePosixPath(name)
    if len(path.parts) != 2 or path.parts[0] != "frames":
        return None
    if path.parts[1] in {"", ".", ".."} or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return None
    return quote(name, safe="/")


def _render_lines(lines: Any) -> str:
    if not isinstance(lines, list) or not lines:
        return '<p class="empty">此章节无记录</p>'
    return "".join(f'<p class="line">{escape(str(line))}</p>' for line in lines)


def _render_frames(item: dict[str, Any]) -> str:
    figures: list[str] = []
    seen: set[str] = set()
    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    for point in evidence:
        if not isinstance(point, dict):
            continue
        href = _safe_frame_href(point.get("frame", ""))
        if href is None or href in seen:
            continue
        seen.add(href)
        time = escape(format_timestamp(point.get("timestamp", 0)))
        reason = escape(str(point.get("reason") or "画面证据"))
        figures.append(
            f'<figure><a href="{href}" target="_blank" rel="noopener">'
            f'<img src="{href}" loading="lazy" alt="{time} 的视频画面"></a>'
            f'<figcaption>{time} · {reason}</figcaption></figure>'
        )
    return f'<div class="frames">{"".join(figures)}</div>' if figures else ""


def _render_descriptions(item: dict[str, Any]) -> str:
    descriptions = item.get("descriptions")
    if not isinstance(descriptions, list) or not descriptions:
        return ""
    rows: list[str] = []
    for entry in descriptions:
        if not isinstance(entry, dict):
            continue
        frame_label = escape(str(entry.get("frame", "")))
        text = escape(str(entry.get("text", "")))
        rows.append(f'<p class="line"><strong>{frame_label}</strong> {text}</p>')
    if not rows:
        return ""
    return f'<div class="content-block" style="margin-top:12px"><h3>AI 画面理解（模型生成，请以原图核实）</h3>{"".join(rows)}</div>'


def render_report(metadata: dict[str, Any], timeline: list[dict[str, Any]], summary: str) -> str:
    duration = format_timestamp(metadata.get("duration", 0))
    frame_count = int(metadata.get("frame_count", 0) or 0)
    speech_count = sum(len(item.get("speech", [])) for item in timeline)
    ocr_count = sum(len(item.get("ocr", [])) for item in timeline)
    navigation: list[str] = []
    chapters: list[str] = []
    for index, item in enumerate(timeline, 1):
        start = format_timestamp(item.get("start", 0))
        end = format_timestamp(item.get("end", 0))
        title = escape(str(item.get("title", "未命名章节")))
        chapter_id = f"chapter-{index}"
        navigation.append(
            f'<a class="nav-link" href="#{chapter_id}"><span class="nav-time">{start} – {end}</span>{title}</a>'
        )
        chapters.append(
            f'<section class="chapter" id="{chapter_id}">'
            f'<header class="chapter-head"><span class="time">{start} – {end}</span>'
            f'<h2 class="chapter-title">{title}</h2></header>'
            f'<div class="columns"><div class="content-block"><h3>语音字幕</h3>{_render_lines(item.get("speech"))}</div>'
            f'<div class="content-block"><h3>画面文字 OCR</h3>{_render_lines(item.get("ocr"))}</div></div>'
            f'{_render_descriptions(item)}{_render_frames(item)}</section>'
        )
    content = "".join(chapters) or '<div class="panel empty">没有可用的时间轴内容。</div>'
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Video → AI 资料包报告</title><style>' + STYLE + '</style></head><body>'
        '<div class="layout"><aside><p class="brand">Video → AI</p>'
        '<p class="aside-note">离线证据报告<br>点击章节跳转；右侧可搜索字幕与 OCR。</p>'
        '<p class="nav-title">章节导航</p>' + "".join(navigation) + '</aside><main>'
        '<div class="eyebrow">本地视频资料包</div><h1>视频证据报告</h1>'
        '<p class="lede">按时间查看讲解、屏幕文字与关键画面。文字识别可能存在误差，请结合原始画面核实。</p>'
        '<div class="metrics">'
        f'<div class="metric"><strong>{duration}</strong><span>视频时长</span></div>'
        f'<div class="metric"><strong>{len(timeline)}</strong><span>时间章节</span></div>'
        f'<div class="metric"><strong>{speech_count}</strong><span>语音片段</span></div>'
        f'<div class="metric"><strong>{frame_count}</strong><span>唯一关键帧 · OCR {ocr_count} 行</span></div>'
        '</div><details class="panel"><summary>查看完整总结</summary>'
        f'<pre class="summary">{escape(summary)}</pre></details>'
        '<div class="toolbar"><input class="search" id="search" type="search" '
        'placeholder="搜索字幕、OCR 或章节标题…" aria-label="搜索报告">'
        f'<span class="count" id="result-count">{len(timeline)} / {len(timeline)} 章节</span></div>'
        + content +
        '<footer>本报告在本机离线打开。技术结论应回到字幕、OCR 和关键画面核实。</footer>'
        '</main></div><script>' + SCRIPT + '</script></body></html>'
    )


def export_report(output: Path, metadata: dict[str, Any], timeline: list[dict[str, Any]]) -> Path:
    summary_path = output / "summary.md"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "暂无总结。"
    destination = output / "report.html"
    write_text(destination, render_report(metadata, timeline, summary))
    return destination
