@echo off
REM Launch RxClarify. Double-click this file, or run it from any terminal.
REM
REM Batch files are not subject to PowerShell's execution policy, so this works
REM even when .venv\Scripts\Activate.ps1 is blocked.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   No virtualenv found at .venv
    echo.
    echo   Create it first:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py

REM Keep the window open if something failed, so the message is readable
REM when this was launched by double-clicking.
if errorlevel 1 pause
