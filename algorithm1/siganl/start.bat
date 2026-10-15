@echo off
cd /d "%~dp0"
set PY=E:\anaconda3\envs\animedev\python.exe
if not exist "%PY%" set PY=python
echo ================================================
echo  Fall Detection System  (acceleration)
echo  env: %PY%
echo ================================================
"%PY%" run.py --host 127.0.0.1 --port 8000
pause
