from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import ctypes


def command_version(command: str, argument: str = "--version") -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, argument], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        output = (result.stdout or result.stderr).splitlines()
        return output[0] if output else executable
    except (OSError, subprocess.SubprocessError):
        return None


def cuda_status() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"PyTorch: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                ctypes.WinDLL("cublas64_12.dll")
                ctypes.WinDLL("cudnn64_9.dll")
            except OSError as exc:
                return False, f"GPU found but CUDA runtime DLL is unavailable: {exc}"
            return True, f"CTranslate2 devices: {ctranslate2.get_cuda_device_count()}; cuBLAS/cuDNN: OK"
    except Exception:
        pass
    smi = command_version("nvidia-smi")
    return False, "nvidia-smi found, but runtime CUDA unavailable" if smi else "not detected"


def main() -> int:
    env = os.getenv("CONDA_DEFAULT_ENV", "(not active)")
    cuda, cuda_detail = cuda_status()
    ffmpeg = command_version("ffmpeg", "-version")
    try:
        import paddleocr  # noqa: F401
        ocr = "YES (PaddleOCR)"
    except Exception:
        ocr = "NO (optional)"
    print(f"Conda env: {env}")
    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"FFmpeg: {'OK - ' + ffmpeg if ffmpeg else 'MISSING'}")
    print(f"CUDA: {'YES' if cuda else 'NO'} - {cuda_detail}")
    print(f"OCR: {ocr}")
    if env != "video2ai":
        print("\nERROR: Activate the dedicated environment first: conda activate video2ai")
        return 2
    if not ffmpeg:
        print("\nERROR: Install FFmpeg in this environment: conda install -c conda-forge ffmpeg")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
