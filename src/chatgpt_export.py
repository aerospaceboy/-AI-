from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .utils import format_timestamp, write_json, write_text


def _read_image(path: Path) -> Any:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _write_image(path: Path, image: Any) -> None:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"Could not encode contact sheet: {path}")
    encoded.tofile(str(path))


def create_contact_sheets(
    output_dir: Path,
    scenes: list[dict[str, Any]],
    package_dir: Path,
    per_sheet: int = 9,
) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    package_dir.mkdir(parents=True, exist_ok=True)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scene in scenes:
        if scene["frame"] not in seen:
            seen.add(scene["frame"])
            unique.append(scene)

    cell_width, image_height, label_height = 480, 270, 30
    columns, rows = 3, 3
    manifests: list[dict[str, Any]] = []
    for sheet_index in range(0, len(unique), per_sheet):
        group = unique[sheet_index:sheet_index + per_sheet]
        canvas = np.zeros((rows * (image_height + label_height), columns * cell_width, 3), dtype=np.uint8)
        included: list[dict[str, Any]] = []
        for offset, scene in enumerate(group):
            source = output_dir / "frames" / scene["frame"]
            image = _read_image(source)
            if image is None:
                continue
            height, width = image.shape[:2]
            scale = min(cell_width / width, image_height / height)
            resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
            row, column = divmod(offset, columns)
            x = column * cell_width + (cell_width - resized.shape[1]) // 2
            y = row * (image_height + label_height) + (image_height - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            label_y = row * (image_height + label_height) + image_height + 21
            label = f"{scene['frame']}  {format_timestamp(scene['timestamp'])}  {scene.get('reason', '')}"
            cv2.putText(canvas, label, (column * cell_width + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            included.append({
                "frame": scene["frame"], "timestamp": scene["timestamp"],
                "timestamp_text": format_timestamp(scene["timestamp"]),
            })
        filename = f"contact_sheet_{len(manifests) + 1:03d}.jpg"
        _write_image(package_dir / filename, canvas)
        manifests.append({"file": filename, "frames": included})
    return manifests


def export_chatgpt_package(output_dir: Path, scenes: list[dict[str, Any]]) -> None:
    package_dir = output_dir / "chatgpt"
    package_dir.mkdir(parents=True, exist_ok=True)
    for old in package_dir.glob("contact_sheet_*.jpg"):
        old.unlink()
    sheets = create_contact_sheets(output_dir, scenes, package_dir)

    parts = [
        "# Video Evidence Package", "",
        "本文件合并了视频总结、时间轴、OCR 和字幕。技术事实应通过时间戳与联系表核实。", "",
    ]
    for name in ("summary.md", "timeline.md", "ocr.md", "transcript.md"):
        path = output_dir / name
        if path.exists():
            parts += ["---", "", f"<!-- SOURCE: {name} -->", "", path.read_text(encoding="utf-8"), ""]
    write_text(package_dir / "VIDEO_EVIDENCE.md", "\n".join(parts))

    sheet_lines: list[str] = []
    for sheet in sheets:
        sheet_lines += [f"- `{sheet['file']}`", *[
            f"  - {item['timestamp_text']} → `{item['frame']}`" for item in sheet["frames"]
        ]]
    write_text(package_dir / "CHATGPT_HANDOFF.md", """# ChatGPT 接手提示

请完整阅读 `VIDEO_EVIDENCE.md`，并结合上传的 `contact_sheet_*.jpg` 核实画面。

请输出：视频目标、章节结构、已完成事项、关键工具/文件/命令/参数、问题与解决方法、当前结果及下一步。

不得补充证据中不存在的信息；无法确认时标记 `[未确认]`。

## 联系表索引

""" + "\n".join(sheet_lines))
    write_json(package_dir / "manifest.json", {
        "upload_first": ["upload_bundle.zip"],
        "upload_separately": ["CHATGPT_HANDOFF.md", "VIDEO_EVIDENCE.md"],
        "contact_sheets": sheets,
        "note": "Upload upload_bundle.zip, or upload the two Markdown files and contact sheets separately.",
    })

    # Keep both archives synchronized with the evidence that generated them.
    # frames.zip is retained for compatibility with packages created by v0.1.
    unique_frames = list(dict.fromkeys(scene["frame"] for scene in scenes))
    with ZipFile(output_dir / "frames.zip", "w", ZIP_DEFLATED) as archive:
        for filename in unique_frames:
            source = output_dir / "frames" / filename
            if source.is_file():
                archive.write(source, f"frames/{filename}")

    with ZipFile(package_dir / "upload_bundle.zip", "w", ZIP_DEFLATED) as archive:
        for filename in ("CHATGPT_HANDOFF.md", "VIDEO_EVIDENCE.md", "manifest.json"):
            archive.write(package_dir / filename, filename)
        for sheet in sheets:
            filename = sheet["file"]
            archive.write(package_dir / filename, filename)
