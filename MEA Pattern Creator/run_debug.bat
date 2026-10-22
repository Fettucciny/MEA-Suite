@echo off
title MEA Pattern Creator (debug)
rem === Runs the GUI with a visible console so errors are readable. ===
set "HERE=%~dp0"
set "PYC=%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\python.exe"
if not exist "%PYC%" set "PYC=python"
echo Using Python: %PYC%
"%PYC%" -c "import sys;print('version',sys.version)"
echo.
echo Launching MEA Pattern Creator... (leave this window open while it runs)
echo.
"%PYC%" "%HERE%app.py"
echo.
echo App exited with code %errorlevel%.
if exist "%HERE%error.log" (
    echo ---- error.log ----
    type "%HERE%error.log"
)
echo.
pause
