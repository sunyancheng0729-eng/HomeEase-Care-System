@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. See README.md for installation.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py %*
pause
