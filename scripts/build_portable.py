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
    for package in ("paddle", "paddleocr", "torch", "transformers", "librosa", "pandas", "pytest"):
        command += ["--exclude-module", package]
    command.append(str(ROOT / "video2ai_app.py"))
    subprocess.run(command, check=True, cwd=ROOT)

    app_dir = dist_dir / "Video2AI"
    shutil.copyfile(ROOT / "config.beginner.yaml", app_dir / "video2ai.yaml")
    shutil.copyfile(ROOT / "给新手的使用说明.txt", app_dir / "先看我.txt")
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
