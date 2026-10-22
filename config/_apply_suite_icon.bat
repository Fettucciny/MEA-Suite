@echo off
setlocal EnableExtensions
set "HERE=%~dp0"
for %%I in ("%HERE%..") do set "ROOT=%%~fI"
if not exist "%HERE%mea_suite.ico" ( echo mea_suite.ico missing in config folder. & pause & exit /b 1 )
if not exist "%ROOT%\desktop.ini" (
    >"%ROOT%\desktop.ini" echo [.ShellClassInfo]
    >>"%ROOT%\desktop.ini" echo IconResource=config\mea_suite.ico,0
    >>"%ROOT%\desktop.ini" echo ConfirmFileOp=0
)
attrib +h +s "%ROOT%\desktop.ini" 1>nul 2>nul
attrib +r "%ROOT%" 1>nul 2>nul
echo.
echo Folder icon applied to:
echo   "%ROOT%"
echo Press F5 in Explorer if it does not show yet. Run once per computer.
echo.
pause
