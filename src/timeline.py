from __future__ import annotations

import re
from typing import Any


COMMAND_RE = re.compile(
    r"^(?:\$|>|PS\s|python\s|pip\s|conda\s|git\s|cmake\s|make\s|ros2\s|colcon\s|"
    r"vlog\s|vsim\s|run\s+-|gcc\s|g\+\+\s|ffmpeg\s|docker\s)", re.I
)
CODE_RE = re.compile(
    r"(?:\b(?:def|class|module|always|assign|parameter|include|import|return|void|int|float)\b|"
    r"[{};]|=>|==|@\(posedge|#include)", re.I
)
ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|fatal|报错|错误|失败)\b", re.I)


def classify_lines(lines: list[str]) -> dict[str, list[str]]:
    result = {"code": [], "commands": [], "errors": [], "notes": []}
    for line in lines:
        if ERROR_RE.search(line):
            result["errors"].append(line)
        elif COMMAND_RE.search(line):
            result["commands"].append(line)
        elif CODE_RE.search(line):
            result["code"].append(line)
    return result


def build_timeline(
    transcript: list[dict[str, Any]],
    ocr: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    duration: float,
    chunk_seconds: float = 120.0,
    descriptions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    descriptions = descriptions or {}
    if duration <= 0:
        ends = [float(x.get("end", x.get("timestamp", 0))) for x in transcript + ocr + scenes]
        duration = max(ends, default=0.0)
    if duration <= 0:
        return []

    timeline: list[dict[str, Any]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        speech = [x.get("text", "") for x in transcript if start <= float(x.get("start", 0)) < end and x.get("text")]
        relevant_ocr = [x for x in ocr if start <= float(x.get("timestamp", 0)) < end]
        ocr_lines = [line for item in relevant_ocr for line in item.get("text", [])]
        relevant_scenes = [x for x in scenes if start <= float(x.get("timestamp", 0)) < end]
        classified = classify_lines(ocr_lines)
        title_source = next((text for text in speech if text.strip()), "")
        title = (title_source[:48] + ("…" if len(title_source) > 48 else "")) or "画面记录"
        frame_paths = list(dict.fromkeys(f"frames/{x['frame']}" for x in relevant_scenes))
        frame_descriptions: list[dict[str, str]] = []
        seen_frames: set[str] = set()
        for x in relevant_scenes:
            frame = x["frame"]
            if frame in descriptions and frame not in seen_frames:
                seen_frames.add(frame)
                frame_descriptions.append({"frame": f"frames/{frame}", "text": descriptions[frame]})
        timeline.append({
            "start": round(start, 3), "end": round(end, 3), "title": title,
            "speech": speech, "ocr": ocr_lines,
            "frames": frame_paths,
            "descriptions": frame_descriptions,
            "code": classified["code"], "commands": classified["commands"],
            "errors": classified["errors"], "notes": classified["notes"],
            "evidence": [
                {
                    "timestamp": x["timestamp"], "frame": f"frames/{x['frame']}",
                    "reason": x.get("reason"), "reused_frame": bool(x.get("reused_frame")),
                }
                for x in relevant_scenes
            ],
        })
        start = end
    return timeline
