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
        "scene_sample_fps", "timeline_chunk_seconds", "ocr", "llm", "vlm",
    }
    unknown = sorted(set(data) - supported)
    if unknown:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")

    options = {key: value for key, value in data.items() if key not in {"ocr", "llm", "vlm"}}
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
    if "api_key" in llm:
        raise ValueError(
            "LLM 'api_key' must not be stored in the YAML config: the file is tracked by Git. "
            "Put the key in .env (OPENAI_API_KEY) or pass --api-key on the command line."
        )
    unknown_llm = sorted(set(llm) - {"enabled", "base_url", "model", "workers"})
    if unknown_llm:
        raise ValueError(f"Unknown LLM config option(s): {', '.join(unknown_llm)}")
    if "enabled" in llm:
        if not isinstance(llm["enabled"], bool):
            raise ValueError("LLM 'enabled' must be true or false")
        options["no_llm"] = not bool(llm["enabled"])
    if "workers" in llm:
        try:
            workers = int(llm["workers"])
        except (TypeError, ValueError):
            raise ValueError("LLM 'workers' must be an integer") from None
        if not 1 <= workers <= 16:
            raise ValueError("LLM 'workers' must be between 1 and 16")
        options["llm_workers"] = workers
    for key in ("base_url", "model"):
        if key in llm:
            options[key] = llm[key]

    vlm = data.get("vlm", {})
    if not isinstance(vlm, dict):
        raise ValueError("Config option 'vlm' must be a mapping")
    if "api_key" in vlm:
        raise ValueError(
            "VLM 'api_key' must not be stored in the YAML config: the file is tracked by Git. "
            "Put the key in .env (OPENAI_API_KEY) or pass --api-key on the command line."
        )
    unknown_vlm = sorted(set(vlm) - {"enabled", "base_url", "model", "workers"})
    if unknown_vlm:
        raise ValueError(f"Unknown VLM config option(s): {', '.join(unknown_vlm)}")
    if "enabled" in vlm:
        if not isinstance(vlm["enabled"], bool):
            raise ValueError("VLM 'enabled' must be true or false")
        options["no_vlm"] = not bool(vlm["enabled"])
    if "workers" in vlm:
        try:
            vlm_workers = int(vlm["workers"])
        except (TypeError, ValueError):
            raise ValueError("VLM 'workers' must be an integer") from None
        if not 1 <= vlm_workers <= 16:
            raise ValueError("VLM 'workers' must be between 1 and 16")
        options["vlm_workers"] = vlm_workers
    for key in ("base_url", "model"):
        if key in vlm:
            options[f"vlm_{key}"] = vlm[key]
    return options
