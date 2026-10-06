from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .llm import LLMProvider, NoLLMProvider
from .utils import format_timestamp, write_json, write_text


LOG = logging.getLogger("video2ai")
SUMMARY_PROMPT_VERSION = 2
DEFAULT_SUMMARY_WORKERS = 4
SUMMARY_SECTIONS = [
    "视频主要内容", "当前完成了什么", "核心知识点", "操作步骤", "使用的软件和工具",
    "涉及的文件", "关键代码", "关键命令", "关键参数", "出现的问题", "报错信息",
    "解决方法", "当前结果", "当前进度", "未完成内容", "后续建议",
]


def _chunk_source(item: dict[str, Any]) -> str:
    lines = [f"时间: {format_timestamp(item['start'])} - {format_timestamp(item['end'])}"]
    for speech in item.get("speech", []):
        lines.append(f"[语音] {speech}")
    for text in item.get("ocr", []):
        lines.append(f"[OCR] {text}")
    for description in item.get("descriptions", []):
        lines.append(f"[画面描述 {description.get('frame', '')}] {description.get('text', '')}")
    for frame in item.get("frames", []):
        lines.append(f"[画面] {frame}")
    return "\n".join(lines)


def basic_summary(timeline: list[dict[str, Any]]) -> str:
    speech_count = sum(len(item.get("speech", [])) for item in timeline)
    ocr_count = sum(len(item.get("ocr", [])) for item in timeline)
    frames = sum(len(item.get("frames", [])) for item in timeline)
    commands = [x for item in timeline for x in item.get("commands", [])]
    errors = [x for item in timeline for x in item.get("errors", [])]
    code = [x for item in timeline for x in item.get("code", [])]
    all_text = "\n".join(
        [x for item in timeline for x in item.get("speech", []) + item.get("ocr", [])]
    )
    known_tools = [
        "VSCode", "Visual Studio Code", "Vivado", "ModelSim", "Quartus", "MATLAB",
        "Python", "Verilog", "FPGA", "ROS2", "Gazebo", "RViz", "Jupyter",
        "PyCharm", "Keil", "STM32CubeMX", "PowerShell", "Anaconda", "Linux",
    ]
    tools = [tool for tool in known_tools if re.search(re.escape(tool), all_text, re.I)]
    files = sorted(set(re.findall(
        r"(?<!\w)(?:[\w.-]+\.(?:py|c|cpp|h|hpp|v|sv|vhd|xdc|tcl|txt|yaml|yml|json|md))(?!\w)",
        all_text, re.I,
    )))
    parameters = list(dict.fromkeys(
        line for line in code if "=" in line or re.search(r"\bparameter\b", line, re.I)
    ))
    lines = [
        "# 视频总结", "", "## 视频主要内容", "",
        "本文件为无 LLM 模式生成的事实索引，不推断视频中未明确说明的信息。", "",
        *[
            f"- `{format_timestamp(item['start'])} - {format_timestamp(item['end'])}`：{item['title']}"
            for item in timeline
        ], "",
        "## 当前结果", "", f"- 语音片段：{speech_count}", f"- OCR 文本行：{ocr_count}",
        f"- 关键画面：{frames}", "",
    ]
    if tools:
        lines += ["## 使用的软件和工具", "", *[f"- {x}" for x in tools], ""]
    if files:
        lines += ["## 涉及的文件", "", *[f"- `{x}`" for x in files[:100]], ""]
    if code:
        lines += ["## 关键代码", "", "```text", *code[:100], "```", ""]
    if commands:
        lines += ["## 关键命令", "", "```text", *commands[:100], "```", ""]
    if parameters:
        lines += ["## 关键参数", "", "```text", *parameters[:100], "```", ""]
    if errors:
        lines += ["## 报错信息", "", "```text", *errors[:100], "```", ""]
    lines += ["## 当前进度", "", "[未确认]（无 LLM 模式不会推断项目进度。）", ""]
    return "\n".join(lines)


def _summarize_chunk(
    index: int,
    total: int,
    item: dict[str, Any],
    chunks_dir: Path,
    provider: LLMProvider,
    instruction: str,
    force: bool,
) -> str:
    data_path = chunks_dir / f"chunk_{index:03d}.json"
    summary_path = chunks_dir / f"chunk_{index:03d}_summary.md"
    cache_path = chunks_dir / f"chunk_{index:03d}_summary.json"
    cache_source = json.dumps({
        "version": SUMMARY_PROMPT_VERSION,
        "provider": provider.cache_key,
        "instruction": instruction,
        "item": item,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_signature = hashlib.sha256(cache_source.encode("utf-8")).hexdigest()
    write_json(data_path, item)
    cache_data: dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_data = {}
    if summary_path.exists() and cache_data.get("signature") == cache_signature and not force:
        LOG.info("Summary: chunk %02d/%02d cached.", index, total)
        return summary_path.read_text(encoding="utf-8")
    if not item.get("speech") and not item.get("ocr"):
        # A text-only LLM cannot extract anything from frame paths alone; even a
        # forced rebuild has no evidence to re-summarize, so skip the API call.
        summary = "此时间段没有语音讲解，也没有可识别的屏幕文字。"
        write_text(summary_path, summary)
        write_json(cache_path, {"signature": cache_signature, "provider": provider.cache_key})
        LOG.info("Summary: chunk %02d/%02d skipped (no speech or OCR text).", index, total)
        return summary
    summary = provider.summarize(_chunk_source(item)[:60000], instruction)
    write_text(summary_path, summary)
    write_json(cache_path, {"signature": cache_signature, "provider": provider.cache_key})
    LOG.info("Summary: chunk %02d/%02d generated.", index, total)
    return summary


def generate_summary(
    timeline: list[dict[str, Any]],
    chunks_dir: Path,
    provider: LLMProvider,
    force: bool = False,
    max_workers: int = DEFAULT_SUMMARY_WORKERS,
) -> str:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(provider, NoLLMProvider):
        return basic_summary(timeline)

    instruction = (
        "将这一时间段整理成简洁、可追溯的 Markdown。区分操作、代码、命令、参数、报错、解决办法和结果；"
        "技术事实注明时间或画面来源。不要添加来源中没有的内容。"
    )
    total = len(timeline)
    # Chunks are independent, so their requests run concurrently; the cached
    # per-chunk files keep retrying failed runs cheap. pool.map preserves
    # source order, which the final synthesis depends on.
    workers = max(1, min(int(max_workers), total or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        summaries = list(pool.map(
            lambda pair: _summarize_chunk(
                pair[0], total, pair[1], chunks_dir, provider, instruction, force
            ),
            enumerate(timeline, 1),
        ))

    requested = "\n".join(f"## {section}" for section in SUMMARY_SECTIONS)
    final_instruction = (
        "根据分段摘要生成全视频总结。只保留有证据的章节，没有内容的章节省略。"
        "关键代码、命令、参数、报错和结论尽量保留时间证据。必须使用以下 Markdown 标题体系：\n"
        f"# 视频总结\n{requested}"
    )
    return provider.summarize("\n\n---\n\n".join(summaries)[:100000], final_instruction)
