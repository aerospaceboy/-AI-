from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    """Small .env reader; existing environment variables always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_yaml_options(path: Path) -> dict[str, Any]:
    """Load supported CLI defaults from a YAML configuration file."""
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a YAML mapping")

    supported = {
        "output", "language", "whisper_model", "hotwords", "initial_prompt_file",
        "beam_size", "batch_size", "device", "max_frame_gap",
        "scene_sample_fps", "timeline_chunk_seconds", "ocr", "llm",
    }
    unknown = sorted(set(data) - supported)
    if unknown:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")

    options = {key: value for key, value in data.items() if key not in {"ocr", "llm"}}
    for integer_key in ("beam_size", "batch_size"):
        if integer_key in options:
            options[integer_key] = int(options[integer_key])
    for float_key in ("max_frame_gap", "scene_sample_fps", "timeline_chunk_seconds"):
        if float_key in options:
            options[float_key] = float(options[float_key])
    if options.get("device") not in {None, "auto", "cuda", "cpu"}:
        raise ValueError("Config option 'device' must be auto, cuda, or cpu")
    if options.get("whisper_model") not in {None, "tiny", "base", "small", "medium", "large-v3"}:
        raise ValueError("Unsupported whisper_model in config")
    for path_key in ("output", "initial_prompt_file"):
        if options.get(path_key):
            configured_path = Path(str(options[path_key])).expanduser()
            options[path_key] = configured_path if configured_path.is_absolute() else path.parent / configured_path

    ocr = data.get("ocr", {})
    if not isinstance(ocr, dict):
        raise ValueError("Config option 'ocr' must be a mapping")
    unknown_ocr = sorted(set(ocr) - {"enabled", "backend"})
    if unknown_ocr:
        raise ValueError(f"Unknown OCR config option(s): {', '.join(unknown_ocr)}")
    if "backend" in ocr and ocr["backend"] != "paddleocr":
        raise ValueError("Only the 'paddleocr' OCR backend is currently supported")
    if "enabled" in ocr:
        if not isinstance(ocr["enabled"], bool):
            raise ValueError("OCR 'enabled' must be true or false")
        options["no_ocr"] = not bool(ocr["enabled"])

    llm = data.get("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("Config option 'llm' must be a mapping")
    unknown_llm = sorted(set(llm) - {"enabled", "base_url", "model", "api_key"})
    if unknown_llm:
        raise ValueError(f"Unknown LLM config option(s): {', '.join(unknown_llm)}")
    if "enabled" in llm:
        if not isinstance(llm["enabled"], bool):
            raise ValueError("LLM 'enabled' must be true or false")
        options["no_llm"] = not bool(llm["enabled"])
    for key in ("base_url", "model", "api_key"):
        if key in llm:
            options[key] = llm[key]
    return options
