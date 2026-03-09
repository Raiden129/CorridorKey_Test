@echo off
TITLE BiRefNet Setup Wizard
cd /d "%~dp0"
echo ===================================================
echo   BiRefNet (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

:: Check that uv sync has been run (the .venv directory should exist)
if not exist ".venv" (
    echo [ERROR] Project environment not found.
    echo Please run Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

:: Download the BiRefNet model snapshot used by the CLI backend
echo [1/1] Downloading BiRefNet Model Files...
if not exist "BiRefNet\checkpoints" mkdir "BiRefNet\checkpoints"

echo Downloading BiRefNet_HR-matting from HuggingFace...
.\.venv\Scripts\python.exe -m huggingface_hub.cli.hf download zhengpeng7/BiRefNet_HR-matting --local-dir BiRefNet\checkpoints\BiRefNet_HR-matting
if %errorlevel% neq 0 (
    echo [ERROR] BiRefNet download failed. Please check the output above for details.
    pause
    exit /b
)

echo.
echo ===================================================
echo   BiRefNet Setup Complete!
echo ===================================================
pause
