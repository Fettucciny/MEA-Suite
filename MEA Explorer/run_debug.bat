@echo off
rem ============================================================
rem  MEA Explorer - DEBUG launcher.
rem  Runs with python.exe so a console stays open and any error
rem  or traceback is visible. Use this if the app does nothing
rem  when launched normally.
rem ============================================================
setlocal
cd /d "%~dp0"

set "PYC=%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\python.exe"

if not exist "%PYC%" (
    echo MEA environment not found at "%PYC%".
    echo Run "setup.bat" in the MEA Analyzer folder once to build it.
    pause
    exit /b 1
)

"%PYC%" "%~dp0app.py" %*
echo.
echo (viewer closed)
pause
endlocal
