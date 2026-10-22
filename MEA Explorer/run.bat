@echo off
rem ============================================================
rem  MEA Explorer launcher (no console window).
rem  Launches with pythonw.exe from the shared environment under
rem  Documents so no terminal stays open. For a completely
rem  flash-free launch, double-click "MEA Explorer.vbs" in the
rem  parent folder. To see errors in a console, use run_debug.bat.
rem
rem  Usage:  run.bat                (pick a spike-data folder)
rem          run.bat 400um_2_D4     (open a specific stem)
rem          run.bat 400um_2_D4 --no-qc
rem ============================================================
cd /d "%~dp0"

set "PYW=%USERPROFILE%\Documents\MEA Analyzer\.venv\Scripts\pythonw.exe"

if not exist "%PYW%" (
    echo.
    echo   MEA environment not found at:
    echo     "%PYW%"
    echo   Run "setup.bat" in the MEA Analyzer folder once to build it.
    echo.
    pause
    exit /b 1
)

rem Loaders folder (mea_nap_io.py etc.). Override here if the repo moves.
rem set "MEANAP_DIR=C:\Users\Yikan\Nextcloud2\MEA\MEA Analyzer\Scripts\MEA-NAP"

start "" /b "%PYW%" "%~dp0app.py" %*
