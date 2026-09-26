@echo off
setlocal
cd /d "%~dp0"
set "VIDEO2AI_CONDA=D:\anaconda\anaconda3\Scripts\conda.exe"
if not exist "%VIDEO2AI_CONDA%" (
    where conda >nul 2>nul
    if errorlevel 1 (
        echo Could not find Conda. Open Anaconda Prompt and run: conda activate video2ai
        pause
        exit /b 1
    )
    set "VIDEO2AI_CONDA=conda"
)
"%VIDEO2AI_CONDA%" run --no-capture-output -n video2ai python video2ai_app.py
if errorlevel 1 pause
endlocal
