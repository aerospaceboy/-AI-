from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .config import load_dotenv, load_yaml_options
from .pipeline import Options, run
from .utils import default_output_path, setup_logging


LOG = logging.getLogger("video2ai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a video into an AI-readable evidence package.")
    parser.add_argument("--version", action="version", version=f"video2ai {__version__}")
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, help="YAML defaults file (auto-loads ./video2ai.yaml when present).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--whisper-model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--hotwords", help="Comma-separated domain terms to bias ASR.")
    parser.add_argument("--initial-prompt-file", type=Path, help="UTF-8 text file used as the Whisper initial prompt.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0, help="Enable batched ASR when greater than 0 (e.g. 8).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-frame-gap", type=float, default=30.0)
    parser.add_argument("--scene-sample-fps", type=float, default=2.0, help="FPS used for scene-change analysis; 0 scans every frame.")
    parser.add_argument("--timeline-chunk-seconds", type=float, default=120.0)
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--ocr", dest="no_ocr", action="store_false", help="Enable OCR, overriding config.")
    ocr_group.add_argument("--no-ocr", dest="no_ocr", action="store_true")
    parser.set_defaults(no_ocr=False)
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", dest="no_llm", action="store_false", help="Enable configured LLM.")
    llm_group.add_argument("--no-llm", dest="no_llm", action="store_true")
    parser.set_defaults(no_llm=False)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-asr", action="store_true")
    parser.add_argument("--force-frames", action="store_true")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--force-timeline", action="store_true")
    parser.add_argument("--force-summary", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = argv if argv is not None else sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(effective_argv)
    config_path = pre_args.config
    if config_path is None:
        automatic = Path.cwd() / "video2ai.yaml"
        config_path = automatic if automatic.is_file() else None
    parser = build_parser()
    if config_path is not None:
        try:
            parser.set_defaults(**load_yaml_options(config_path.expanduser().resolve()))
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(effective_argv)
    if args.beam_size < 1:
        raise SystemExit("--beam-size must be at least 1")
    if args.batch_size < 0:
        raise SystemExit("--batch-size cannot be negative")
    if args.max_frame_gap <= 0 or args.timeline_chunk_seconds <= 0:
        raise SystemExit("Frame gap and timeline chunk size must be positive")
    if args.scene_sample_fps < 0:
        raise SystemExit("--scene-sample-fps cannot be negative")
    setup_logging(args.verbose)
    load_dotenv(Path.cwd() / ".env")
    video = args.video.expanduser()
    output = args.output or default_output_path(video)
    conda_env = os.getenv("CONDA_DEFAULT_ENV", "(not active)")
    LOG.info("=" * 50)
    LOG.info("            Video → AI Package")
    LOG.info("=" * 50)
    if getattr(sys, "frozen", False):
        LOG.info("\nApplication:\n%s", sys.executable)
    else:
        LOG.info("\nConda Environment:\n%s", conda_env)
        LOG.info("\nPython:\n%s", sys.executable)
    LOG.info("\nVideo:\n%s\n", video.resolve())
    if config_path is not None:
        LOG.info("Config: %s", config_path.expanduser().resolve())
    if not getattr(sys, "frozen", False) and conda_env != "video2ai":
        LOG.warning("[WARN] Expected Conda environment 'video2ai'. No packages will be installed, but runtime dependencies may be missing.")
    try:
        result = run(Options(
            video=video, output=output, language=args.language, whisper_model=args.whisper_model,
            hotwords=args.hotwords, initial_prompt_file=args.initial_prompt_file,
            beam_size=args.beam_size, batch_size=args.batch_size,
            device=args.device, max_frame_gap=args.max_frame_gap, no_ocr=args.no_ocr,
            scene_sample_fps=args.scene_sample_fps,
            timeline_chunk_seconds=args.timeline_chunk_seconds,
            no_llm=args.no_llm, base_url=args.base_url, api_key=args.api_key, model=args.model,
            overwrite=args.overwrite, force_asr=args.force_asr, force_frames=args.force_frames,
            force_ocr=args.force_ocr, force_timeline=args.force_timeline, force_summary=args.force_summary,
        ))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        LOG.error("[ERROR] %s", exc)
        return 1
    LOG.info("\nDone.\n\nOutput:\n%s", result)
    return 0
