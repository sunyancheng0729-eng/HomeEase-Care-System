@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" run.py --stop
