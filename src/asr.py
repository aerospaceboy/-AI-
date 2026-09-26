from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


LOG = logging.getLogger("video2ai")


def model_location(model_name: str) -> str:
    """Use the bundled default model in the Windows starter build."""
    if getattr(sys, "frozen", False) and model_name == "base":
        bundled = Path(sys.executable).resolve().parent / "models" / "base"
        if all((bundled / name).is_file() for name in ("config.json", "model.bin", "tokenizer.json")):
            return str(bundled)
    return model_name


def resolve_device(requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "int8"
    cuda_available = False
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import ctranslate2
        cuda_available = cuda_available or ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    if requested == "cuda" and not cuda_available:
        LOG.warning("CUDA unavailable, fallback to CPU.")
    return ("cuda", "float16") if cuda_available and requested != "cpu" else ("cpu", "int8")


def transcribe(
    audio_path: Path,
    model_name: str = "small",
    language: str | None = None,
    device: str = "auto",
    hotwords: str | None = None,
    initial_prompt: str | None = None,
    beam_size: int = 5,
    batch_size: int = 0,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc

    actual_device, compute_type = resolve_device(device)

    def run_once(selected_device: str, selected_compute_type: str) -> tuple[list[dict[str, Any]], Any]:
        LOG.info("ASR device: %s (%s)", selected_device, selected_compute_type)
        model = WhisperModel(model_location(model_name), device=selected_device, compute_type=selected_compute_type)
        kwargs = {
            "language": language,
            "vad_filter": True,
            "word_timestamps": True,
            "beam_size": beam_size,
            "hotwords": hotwords,
            "initial_prompt": initial_prompt,
        }
        if batch_size > 0:
            from faster_whisper import BatchedInferencePipeline

            LOG.info("ASR batched inference: batch_size=%s", batch_size)
            segments, info = BatchedInferencePipeline(model=model).transcribe(
                str(audio_path), batch_size=batch_size, **kwargs
            )
        else:
            segments, info = model.transcribe(str(audio_path), **kwargs)
        output: list[dict[str, Any]] = []
        for segment in segments:
            words = [
                {"start": word.start, "end": word.end, "word": word.word}
                for word in (segment.words or [])
            ]
            output.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": words,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
                "compression_ratio": getattr(segment, "compression_ratio", None),
            })
        return output, info

    try:
        output, info = run_once(actual_device, compute_type)
    except Exception as exc:
        if actual_device != "cuda":
            raise
        LOG.warning("[WARN] CUDA ASR failed; fallback to CPU. (%s)", exc)
        actual_device, compute_type = "cpu", "int8"
        output, info = run_once(actual_device, compute_type)
    return {
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "model": model_name,
        "device": actual_device,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "hotwords": hotwords,
        "segments": output,
    }
