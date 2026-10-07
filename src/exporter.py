from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import format_timestamp, write_text


def export_transcript(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = ["# 视频完整字幕", ""]
    if not segments:
        lines += ["未检测到音频或没有可用字幕。", ""]
    for item in segments:
        lines += [
            f"## {format_timestamp(item['start'])} - {format_timestamp(item['end'])}", "",
            item.get("text", ""), "",
        ]
    write_text(path, "\n".join(lines))


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis >= 1000:
        whole += 1
        millis = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    """Player-ready SRT subtitles built from the transcript segments."""
    blocks: list[str] = []
    for item in segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", 0) or 0)
        end = float(item.get("end", 0) or 0)
        if end <= start:
            end = start + 0.5
        blocks.append(
            f"{len(blocks) + 1}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}"
        )
    write_text(path, "\n\n".join(blocks) + ("\n" if blocks else "无可用字幕。\n"))


def export_ocr(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# 屏幕文字 OCR", ""]
    visible = [x for x in records if x.get("text")]
    if not visible:
        lines += ["未启用 OCR、OCR 不可用，或未识别到文字。", ""]
    for item in visible:
        lines += [
            f"## {format_timestamp(item['timestamp'])}", "", "来源：", "",
            f"`frames/{item['frame']}`", "", "识别结果（保留 OCR 原文）：", "",
            "```text", *item["text"], "```", "",
        ]
    write_text(path, "\n".join(lines))


def export_timeline(path: Path, timeline: list[dict[str, Any]]) -> None:
    lines = ["# 视频时间轴", ""]
    if not timeline:
        lines += ["没有可用时间轴内容。", ""]
    for item in timeline:
        lines += [
            f"## {format_timestamp(item['start'])} - {format_timestamp(item['end'])}", "",
            f"### {item['title']}", "",
        ]
        if item.get("speech"):
            lines += ["讲解内容：", "", *[f"- {x}" for x in item["speech"]], ""]
        if item.get("code"):
            lines += ["关键代码（OCR 原文）：", "", "```text", *item["code"], "```", ""]
        if item.get("commands"):
            lines += ["执行命令：", "", "```bash", *item["commands"], "```", ""]
        if item.get("errors"):
            lines += ["报错/异常线索：", "", "```text", *item["errors"], "```", ""]
        remaining = [x for x in item.get("ocr", []) if x not in item.get("code", []) + item.get("commands", []) + item.get("errors", [])]
        if remaining:
            lines += ["其他屏幕文字：", "", "```text", *remaining, "```", ""]
        if item.get("descriptions"):
            lines += ["AI 画面理解（模型生成，请以原图核实）：", "", *[
                f"- `{d['frame']}`：{d['text']}" for d in item["descriptions"]
            ], ""]
        if item.get("frames"):
            lines += ["相关画面：", "", *[f"`{x}`" for x in item["frames"]], ""]
        lines += ["---", ""]
    write_text(path, "\n".join(lines))


def export_package_readme(path: Path, source: Path, duration: float) -> None:
    text = f"""# Video AI Package

原始视频：`{source}`

视频时长：{format_timestamp(duration)}

## 给 AI Agent 的读取建议

交给 ChatGPT 时，优先上传 `chatgpt/upload_bundle.zip`。如果不支持读取 ZIP，则上传 `chatgpt/` 内的两个 Markdown 文件及所有联系表图片。

首先读取：

1. `summary.md`
2. `timeline.md`

需要核实时再读取 `transcript.md`、`ocr.md`、`timeline.json` 和 `frames/`。

## 文件说明

- `summary.md`：快速了解内容和当前状态。
- `report.html`：在浏览器中离线浏览和搜索证据。
- `timeline.md`：按时间记录完整过程。
- `transcript.md`：原始语音转写。
- `subtitle.srt`：可直接导入播放器或剪辑软件的字幕文件。
- `ocr.md`：屏幕 OCR 原文。
- `timeline.json`：机器可解析时间轴。
- `frames/`：关键视频画面。
- `frames.zip`：全部唯一关键帧的同步压缩包。
- `chatgpt/upload_bundle.zip`：可直接上传的精简资料包。

## 使用原则

技术事实应回到时间轴、OCR 或关键帧核实。无法确认的信息标记为 `[未确认]`。
"""
    write_text(path, text)


def export_handoff(path: Path) -> None:
    write_text(path, """# AI Agent 接手提示

你现在需要接手一个通过视频记录的项目、实验或者教程。

请先完整读取：

1. `README.md`
2. `summary.md`
3. `timeline.md`

如有必要，再读取 `transcript.md`、`ocr.md`、`timeline.json` 和 `frames/`。

阅读完成后，请总结视频目标、当前进度、已完成和未完成步骤、关键文件、代码、命令、参数、问题及下一步。

任何无法由资料包确认的信息不要猜测，标记为 `[未确认]`。继续修改项目时必须优先基于资料包中的事实。
""")
