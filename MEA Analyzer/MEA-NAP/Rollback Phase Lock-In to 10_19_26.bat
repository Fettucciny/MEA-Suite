@echo off
setlocal EnableExtensions
title Roll back Phase Lock-In
REM ==================================================================
REM   Swaps the phase lock-in module between the FAST build and the
REM   LEGACY build saved on 2026-10-19 (before the vectorised search
REM   and parallel sampling were installed).
REM
REM   Run it once  -> legacy module is active, fast one kept as
REM                   mea_nap_phase_lockin_fast.py
REM   Run it again -> fast module is restored.
REM
REM   The two builds give the same numbers to floating-point rounding
REM   (verified on the self-test); this exists so you can prove that
REM   on your own data whenever you like.
REM ==================================================================
set "HERE=%~dp0"
set "LIVE=%HERE%mea_nap_phase_lockin.py"
set "LEGACY=%HERE%mea_nap_phase_lockin_legacy_10_19_26.py"
set "FAST=%HERE%mea_nap_phase_lockin_fast.py"

if not exist "%LEGACY%" (
    echo Legacy copy not found:
    echo     "%LEGACY%"
    pause & exit /b 1
)

findstr /C:"ProcessPoolExecutor" "%LIVE%" >nul 2>&1
if errorlevel 1 goto :restore_fast

REM ---- live is the FAST build -> switch to legacy ----
copy /y "%LIVE%" "%FAST%" >nul
copy /y "%LEGACY%" "%LIVE%" >nul
echo.
echo   Phase Lock-In is now the LEGACY (10_19_26) build.
echo   The fast build is kept as mea_nap_phase_lockin_fast.py
echo   Run this file again to switch back.
echo.
goto :done

:restore_fast
if not exist "%FAST%" (
    echo Live module is already the legacy build and no fast copy was found.
    pause & exit /b 1
)
copy /y "%FAST%" "%LIVE%" >nul
echo.
echo   Phase Lock-In is now the FAST build again.
echo.

:done
if exist "%HERE%__pycache__\mea_nap_phase_lockin*.pyc" del /q "%HERE%__pycache__\mea_nap_phase_lockin*.pyc"
echo   Restart MEA Analyzer for the change to take effect.
echo.
pause
exit /b 0
