from __future__ import annotations

import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .utils import normalized_lines


LOG = logging.getLogger("video2ai")


class OCRBackend(ABC):
    name = "abstract"

    @abstractmethod
    def recognize(self, image_path: Path) -> list[str]:
        raise NotImplementedError


class NoOCRBackend(OCRBackend):
    name = "none"

    def recognize(self, image_path: Path) -> list[str]:
        return []


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"

    def __init__(self, language: str | None = None, use_gpu: bool = False) -> None:
        # Keep downloaded OCR models inside the isolated Conda environment.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(sys.prefix) / "paddlex_cache"))
        import paddle
        import paddleocr
        from paddleocr import PaddleOCR

        lang = "ch" if not language or language.startswith("zh") else "en"
        major = int(str(getattr(paddleocr, "__version__", "2")).split(".")[0])
        self.v3 = major >= 3
        if self.v3:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            kwargs: dict[str, Any] = {
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                # Paddle 3.3 + OCRv6 can hit an unsupported oneDNN PIR attribute on Windows.
                "enable_mkldnn": False,
            }
            if use_gpu and paddle.is_compiled_with_cuda():
                kwargs["device"] = "gpu:0"
            self.engine = PaddleOCR(**kwargs)
        else:
            self.engine = PaddleOCR(
                use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False
            )

    def recognize(self, image_path: Path) -> list[str]:
        if self.v3:
            result = self.engine.predict(str(image_path))
            lines = [
                str(text)
                for page in result or []
                for text in (page.get("rec_texts", []) if hasattr(page, "get") else [])
            ]
            return normalized_lines(lines)

        result = self.engine.ocr(str(image_path), cls=True)
        lines: list[str] = []
        for page in result or []:
            for item in page or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    detail = item[1]
                    if isinstance(detail, (list, tuple)) and detail:
                        lines.append(str(detail[0]))
        return normalized_lines(lines)


def create_backend(language: str | None, use_gpu: bool) -> OCRBackend:
    try:
        return PaddleOCRBackend(language, use_gpu)
    except Exception as exc:
        LOG.warning("[WARN] OCR initialization failed. Continue without OCR. (%s)", exc)
        return NoOCRBackend()


def _signature(lines: list[str]) -> set[str]:
    text = "\n".join(lines).lower()
    # Single digits are significant in code/configuration (e.g. WIDTH=8 -> 9).
    return set(re.findall(r"(?:[\w./\\:#@+\-]{2,}|\b\d\b)", text))


def is_duplicate_ocr(previous: list[str], current: list[str], threshold: float = 0.97) -> bool:
    if not previous or not current:
        return False
    if previous == current:
        return True
    # Be deliberately conservative when code-like punctuation or numbers changed.
    previous_text = "\n".join(previous)
    current_text = "\n".join(current)
    if re.search(r"[=;{}()\[\]]|\d", previous_text + current_text):
        return False
    a, b = _signature(previous), _signature(current)
    if not a or not b:
        return previous == current
    similarity = len(a & b) / len(a | b)
    return similarity >= threshold


def run_ocr(
    output_dir: Path,
    scenes: list[dict[str, Any]],
    backend: OCRBackend,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous: list[str] = []
    recognized_frames: dict[str, list[str]] = {}
    unique_processed = 0
    for scene in scenes:
        path = output_dir / "frames" / scene["frame"]
        if scene["frame"] in recognized_frames:
            lines = recognized_frames[scene["frame"]]
        else:
            try:
                lines = backend.recognize(path)
                recognized_frames[scene["frame"]] = lines
                unique_processed += 1
                if unique_processed == 1 or unique_processed % 5 == 0:
                    LOG.info("OCR progress: %s unique frames processed", unique_processed)
            except Exception as exc:
                LOG.warning("[WARN] OCR failed for %s; skipped. (%s)", path.name, exc)
                continue
        duplicate = is_duplicate_ocr(previous, lines)
        records.append({
            "timestamp": scene["timestamp"],
            "frame": scene["frame"],
            "text": [] if duplicate else lines,
            "duplicate_of_previous": duplicate,
        })
        if lines and not duplicate:
            previous = lines
    return records
