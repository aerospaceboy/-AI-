"""Build the Windows CPU starter bundle inside the existing video2ai environment."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
BASE_MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


def main() -> int:
    if os.name != "nt":
        raise SystemExit("The beginner bundle must be built on Windows.")
    if os.environ.get("CONDA_DEFAULT_ENV") != "video2ai":
        raise SystemExit("Activate the dedicated video2ai Conda environment before building.")

    import imageio_ffmpeg

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"Static FFmpeg binary not found: {ffmpeg}")

    dist_dir = ROOT / "dist"
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--windowed", "--disable-windowed-traceback", "--noupx", "--name", "Video2AI",
        "--distpath", str(dist_dir), "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--add-binary", f"{ffmpeg};.",
    ]
    for package in ("ctranslate2", "av", "onnxruntime"):
        command += ["--collect-binaries", package]
    command += ["--collect-data", "faster_whisper"]
    for package in ("paddle", "paddleocr", "torch", "transformers", "librosa", "pandas", "pytest", "imageio_ffmpeg"):
        command += ["--exclude-module", package]
    command.append(str(ROOT / "video2ai_app.py"))
    subprocess.run(command, check=True, cwd=ROOT)

    app_dir = dist_dir / "Video2AI"
    shutil.copyfile(ROOT / "config.beginner.yaml", app_dir / "video2ai.yaml")
    shutil.copyfile(ROOT / "给新手的使用说明.txt", app_dir / "先看我.txt")
    from huggingface_hub import snapshot_download

    model_snapshot = Path(snapshot_download(
        repo_id="Systran/faster-whisper-base",
        revision=BASE_MODEL_REVISION,
        allow_patterns=list(BASE_MODEL_FILES),
    ))
    model_dir = app_dir / "models" / "base"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in BASE_MODEL_FILES:
        source = model_snapshot / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Bundled model file missing: {source}")
        shutil.copyfile(source, model_dir / name)
    shutil.copyfile(ROOT / "MODEL_NOTICE.txt", app_dir / "模型来源与许可.txt")
    archive_path = dist_dir / "Video2AI-Windows-CPU.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for source in app_dir.rglob("*"):
            if source.is_file():
                archive.write(source, source.relative_to(dist_dir).as_posix())
    sha256 = hashlib.sha256()
    with archive_path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    (dist_dir / "Video2AI-Windows-CPU.zip.sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    print(f"Portable bundle: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / 1024**2:.1f} MiB")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
