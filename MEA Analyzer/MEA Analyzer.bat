@echo off
setlocal EnableExtensions
title MEA Analyzer

REM ==================================================================
REM   MEA Analyzer  --  double-click launcher for the MEA-NAP GUI
REM ------------------------------------------------------------------
REM   This copy lives INSIDE the "MEA Analyzer\MEA Analyzer" code
REM   folder, next to the "MEA-NAP" folder.
REM
REM   Uses the Python environment built by setup.bat at:
REM       %USERPROFILE%\Documents\MEA Analyzer\.venv
REM   This launcher NEVER installs anything. If the environment does
REM   not exist yet, run "setup.bat" (in the parent folder) once by
REM   hand, then start this.
REM
REM   To force a specific interpreter, put its full path in
REM   MEANAP_PYTHON below.
REM ==================================================================
set "MEANAP_PYTHON="

set "HERE=%~dp0"
set "GUI=%HERE%MEA-NAP\mea_nap_gui.py"
set "VENV=%USERPROFILE%\Documents\MEA Analyzer\.venv"

if not exist "%GUI%" (
    echo Could not find the GUI at:
    echo     "%GUI%"
    echo Make sure this .bat sits in the "MEA Analyzer\MEA Analyzer" folder,
    echo next to the "MEA-NAP" folder.
    echo.
    pause
    exit /b 1
)

REM ---- choose the interpreter ----
if defined MEANAP_PYTHON (
    set "PYC=%MEANAP_PYTHON%"
) else (
    set "PYC=%VENV%\Scripts\python.exe"
)

REM ---- make sure the environment exists and has the dependencies ----
set "ENVOK=1"
if not exist "%PYC%" set "ENVOK="
if defined ENVOK "%PYC%" -c "import h5py, scipy, matplotlib" 1>nul 2>nul
if defined ENVOK if errorlevel 1 set "ENVOK="

if not defined ENVOK (
    echo.
    echo The Python environment is not ready at:
    echo     "%VENV%"
    echo Please run "setup.bat" in the parent folder once to build it,
    echo then start this program again.
    echo.
    pause
    exit /b 1
)

REM ---- launch windowed (pythonw = no console window) ----
set "PYW=%PYC:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PYC%"
start "" "%PYW%" "%GUI%"
exit /b 0
