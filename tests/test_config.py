from pathlib import Path

import pytest

from src.config import load_yaml_options


def test_yaml_config_flattens_sections_and_resolves_paths(tmp_path: Path) -> None:
    config = tmp_path / "video2ai.yaml"
    config.write_text("""
whisper_model: large-v3
device: cuda
initial_prompt_file: prompts/terms.txt
ocr:
  enabled: false
  backend: paddleocr
llm:
  enabled: true
  base_url: http://127.0.0.1:11434/v1
  model: qwen2.5
""", encoding="utf-8")

    options = load_yaml_options(config)
    assert options["whisper_model"] == "large-v3"
    assert options["device"] == "cuda"
    assert options["initial_prompt_file"] == tmp_path / "prompts" / "terms.txt"
    assert options["no_ocr"] is True
    assert options["no_llm"] is False
    assert options["model"] == "qwen2.5"


def test_yaml_config_rejects_unknown_options(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("mystery: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config"):
        load_yaml_options(config)


def test_yaml_config_rejects_api_key(tmp_path: Path) -> None:
    config = tmp_path / "key.yaml"
    config.write_text("llm:\n  enabled: true\n  api_key: sk-secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_yaml_options(config)


def test_yaml_config_llm_workers(tmp_path: Path) -> None:
    config = tmp_path / "workers.yaml"
    config.write_text("llm:\n  enabled: true\n  workers: 2\n", encoding="utf-8")
    options = load_yaml_options(config)
    assert options["llm_workers"] == 2

    config.write_text("llm:\n  workers: 99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        load_yaml_options(config)

    config.write_text("llm:\n  workers: fast\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        load_yaml_options(config)


def test_yaml_config_vlm_section(tmp_path: Path) -> None:
    config = tmp_path / "vlm.yaml"
    config.write_text(
        "vlm:\n  enabled: true\n  model: qwen3-vl-flash\n  workers: 2\n"
        "  base_url: https://example.com/v1\n",
        encoding="utf-8",
    )
    options = load_yaml_options(config)
    assert options["no_vlm"] is False
    assert options["vlm_model"] == "qwen3-vl-flash"
    assert options["vlm_workers"] == 2
    assert options["vlm_base_url"] == "https://example.com/v1"

    config.write_text("vlm:\n  workers: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        load_yaml_options(config)

    config.write_text("vlm:\n  mystery: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown VLM"):
        load_yaml_options(config)


def test_yaml_config_normalizes_numeric_fingerprints(tmp_path: Path) -> None:
    config = tmp_path / "numeric.yaml"
    config.write_text("max_frame_gap: 30\ntimeline_chunk_seconds: 120\n", encoding="utf-8")
    options = load_yaml_options(config)
    assert options["max_frame_gap"] == 30.0
    assert isinstance(options["max_frame_gap"], float)
    assert options["timeline_chunk_seconds"] == 120.0
