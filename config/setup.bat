@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MEA Analyzer  --  environment setup

REM ==================================================================
REM   MEA Analyzer  --  one-click environment installer
REM ------------------------------------------------------------------
REM   The Python environment is built under your Documents folder, NOT
REM   inside this (synced) program folder, so each PC keeps its own
REM   private copy and nothing is synced between machines:
REM
REM       %USERPROFILE%\Documents\MEA Analyzer\.venv
REM
REM   You normally do NOT need to run this by hand: the launchers call
REM   it automatically the first time they cannot find a working
REM   environment. You can still double-click it to (re)build. It will:
REM       1. Find or install Python 3.14 (64-bit)
REM       2. Create the .venv under Documents
REM       3. Install PyTorch (CUDA 12.8 build) + all requirements
REM       4. Verify the key packages import
REM       5. Apply the MEA Suite folder icon
REM
REM   ---- CPU-only or different CUDA? -------------------------------
REM   If the target PC has no NVIDIA GPU, or a different CUDA version,
REM   edit the two TORCH_ lines just below:
REM       CPU only :  set "TORCH_SPEC=torch"
REM                   set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
REM       CUDA 12.1:  set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
REM ==================================================================

set "PYVER=3.14"
set "TORCH_SPEC=torch==2.11.0+cu128"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_FALLBACK=torch==2.11.0"

REM  When called by a launcher we skip the final "press any key" pause
REM  so the program can start straight away.
set "FROMLAUNCHER="
if /i "%~1"=="/fromlauncher" set "FROMLAUNCHER=1"

REM  HERE      = this program folder (holds requirements.txt, launchers)
REM  VENVROOT  = per-machine home for the environment, under Documents
set "HERE=%~dp0"
set "VENVROOT=%USERPROFILE%\Documents\MEA Analyzer"
set "VENV=%VENVROOT%\.venv"
set "REQ=%HERE%requirements.txt"

echo.
echo ==================================================================
echo   MEA Analyzer setup
echo   Program folder : "%HERE%"
echo   Environment    : "%VENV%"
echo ==================================================================
echo.

if not exist "%REQ%" (
    echo [ERROR] requirements.txt was not found next to this file:
    echo     "%REQ%"
    echo Keep setup.bat and requirements.txt together in the same folder.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------------
REM  Step 1 : locate a Python 3.14 interpreter
REM ------------------------------------------------------------------
echo [1/5] Looking for Python %PYVER% ...
set "PYEXE="

REM  1a) the py launcher (installed with python.org builds)
py -%PYVER% -c "import sys" 1>nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%p in ('py -%PYVER% -c "import sys;print(sys.executable)"') do set "PYEXE=%%p"
)

REM  1b) python-build-standalone / pymanager location
if not defined PYEXE if exist "%LocalAppData%\Python\pythoncore-%PYVER%-64\python.exe" set "PYEXE=%LocalAppData%\Python\pythoncore-%PYVER%-64\python.exe"

REM  1c) standard python.org per-user install
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python314\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python314\python.exe"

REM  1d) standard python.org all-users install
if not defined PYEXE if exist "%ProgramFiles%\Python314\python.exe" set "PYEXE=%ProgramFiles%\Python314\python.exe"

if defined PYEXE goto have_python

REM ------------------------------------------------------------------
REM  Python 3.14 not found -- try to install it
REM ------------------------------------------------------------------
echo     Python %PYVER% was not found. Trying to install it with winget ...
where winget 1>nul 2>nul
if errorlevel 1 goto no_winget

winget install --id Python.Python.3.14 -e --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo [ERROR] winget could not install Python %PYVER% automatically.
    goto manual_python
)

REM  re-detect after install
py -%PYVER% -c "import sys" 1>nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%p in ('py -%PYVER% -c "import sys;print(sys.executable)"') do set "PYEXE=%%p"
)
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python314\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python314\python.exe" set "PYEXE=%ProgramFiles%\Python314\python.exe"

if defined PYEXE goto have_python

echo.
echo [ERROR] Python was installed but still could not be located.
echo Please close this window, open a NEW one, and run setup.bat again.
echo.
pause
exit /b 1

:no_winget
echo.
echo [ERROR] winget is not available on this machine.
goto manual_python

:manual_python
echo.
echo   Please install Python %PYVER% (64-bit) manually from:
echo       https://www.python.org/downloads/
echo   During install, tick "Add python.exe to PATH", then try again.
echo.
pause
exit /b 1

:have_python
echo     Using Python at: "%PYEXE%"
"%PYEXE%" --version
echo.

REM ------------------------------------------------------------------
REM  Step 2 : create the virtual environment under Documents
REM ------------------------------------------------------------------
echo [2/5] Creating virtual environment at "%VENV%" ...
if not exist "%VENVROOT%" mkdir "%VENVROOT%"
set "VPY=%VENV%\Scripts\python.exe"

REM  Test that any existing venv's python actually RUNS on this PC; if it
REM  does not (corrupted, or a leftover from another machine), rebuild it.
set "VENV_OK="
if exist "%VPY%" (
    "%VPY%" -c "import sys" 1>nul 2>nul
    if not errorlevel 1 set "VENV_OK=1"
)

if defined VENV_OK (
    echo     A working .venv already exists here -- reusing it.
) else (
    if exist "%VENV%" (
        echo     Existing .venv is unusable -- rebuilding it.
        rmdir /s /q "%VENV%"
    )
    "%PYEXE%" -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create the virtual environment.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%VPY%" (
    echo.
    echo [ERROR] Expected python.exe inside the venv but it is missing:
    echo     "%VPY%"
    echo.
    pause
    exit /b 1
)

REM  final sanity check that the venv python is usable on this machine
"%VPY%" -c "import sys" 1>nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] The virtual environment's Python does not run on this machine.
    echo Delete the folder below by hand and run setup.bat again:
    echo     "%VENV%"
    echo.
    pause
    exit /b 1
)

echo     Upgrading pip ...
"%VPY%" -m pip install --upgrade pip 1>nul
echo.

REM ------------------------------------------------------------------
REM  Step 3 : install PyTorch, then the rest of the requirements
REM ------------------------------------------------------------------
echo [3/5] Installing PyTorch  (%TORCH_SPEC%) ...
echo     This is a large download and may take several minutes.
"%VPY%" -m pip install "%TORCH_SPEC%" --index-url "%TORCH_INDEX%"
if errorlevel 1 (
    echo.
    echo     GPU build failed -- falling back to the CPU-only PyTorch build.
    "%VPY%" -m pip install "%TORCH_FALLBACK%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not install PyTorch. Check your internet connection.
        echo.
        pause
        exit /b 1
    )
)
echo.

echo     Installing the remaining requirements ...
"%VPY%" -m pip install -r "%REQ%"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install one or more packages from requirements.txt.
    echo Scroll up to see which package failed.
    echo.
    pause
    exit /b 1
)
echo.

REM ------------------------------------------------------------------
REM  Step 4 : verify the important packages import
REM ------------------------------------------------------------------
echo [4/5] Verifying the environment ...
"%VPY%" -c "import numpy, scipy, pandas, h5py, matplotlib, sklearn, numba, spkit; print('   core scientific stack OK')"
if errorlevel 1 (
    echo.
    echo [WARNING] Some core packages did not import cleanly. See the error above.
    echo.
    pause
    exit /b 1
)

"%VPY%" -c "import torch; print('   torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

REM ------------------------------------------------------------------
REM  Step 5 : apply the MEA Suite folder icon to the PARENT folder
REM ------------------------------------------------------------------
echo [5/5] Applying the MEA Suite folder icon ...
for %%I in ("%HERE%..") do set "ROOT=%%~fI"
if not exist "%HERE%mea_suite.ico" (
    echo     Skipped - mea_suite.ico not found in the config folder.
) else (
    if not exist "%ROOT%\desktop.ini" (
        >"%ROOT%\desktop.ini" echo [.ShellClassInfo]
        >>"%ROOT%\desktop.ini" echo IconResource=config\mea_suite.ico,0
        >>"%ROOT%\desktop.ini" echo ConfirmFileOp=0
    )
    attrib +h +s "%ROOT%\desktop.ini" 1>nul 2>nul
    attrib +r "%ROOT%" 1>nul 2>nul
    echo     Folder icon set - press F5 in Explorer if it does not show yet.
)
echo.


echo.
if defined FROMLAUNCHER (
    echo ==================================================================
    echo   Environment ready. Starting the program now ...
    echo ==================================================================
) else (
    echo ==================================================================
    echo   Setup complete.  Environment installed at:
    echo     "%VENV%"
    echo   You can now start any of the MEA programs, e.g. by
    echo   double-clicking "MEA Suite.exe" in the parent folder.
    echo ==================================================================
    echo.
    pause
)
exit /b 0
