@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Algorithm 3 demo
set "PYTHONPATH=%~dp0src"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
set "PYTHONWARNINGS=ignore::FutureWarning"

rem 1. the private environment of a previous first run
if exist ".venv\ready.txt" goto run_venv

rem 2. find Python 3.12 (a path in the PYTHON312 variable wins)
set "BASE="
if defined PYTHON312 set BASE="%PYTHON312%"
if not defined BASE (
  where py >nul 2>nul && (py -3.12 -c "import sys" >nul 2>nul && set "BASE=py -3.12")
)
if not defined BASE (
  where python >nul 2>nul && (python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>nul && set "BASE=python")
)
if not defined BASE (
  where python3 >nul 2>nul && (python3 -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>nul && set "BASE=python3")
)
if not defined BASE (
  echo Python 3.12 was not found.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and double-click start.bat again.
  echo If you have Python 3.12 somewhere else ^(for example a conda environment^), set PYTHON312 to its python.exe first:
  echo   set PYTHON312=C:\path\to\python.exe
  echo   start.bat
  pause
  exit /b 1
)

rem 3. an environment that already has every package: use it as it is
if not exist ".venv" (
  %BASE% -c "import faster_whisper, onnxruntime, sklearn, scipy, joblib, soundfile, fastapi, uvicorn, multipart, pydantic, yaml" >nul 2>nul
  if not errorlevel 1 (
    echo Using the Python packages that are already installed.
    %BASE% -m algorithm3
    goto end
  )
)

rem 4. first run: a private environment inside this folder
echo First run: installing the packages into .venv ^(internet is needed once, a few minutes^)...
if not exist ".venv\Scripts\python.exe" (
  %BASE% -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip --no-warn-script-location
".venv\Scripts\python.exe" -m pip install -r requirements.txt --no-warn-script-location
if errorlevel 1 goto fail
echo ok> ".venv\ready.txt"

:run_venv
".venv\Scripts\python.exe" -m algorithm3
goto end

:fail
echo.
echo The installation failed, see the messages above.
echo If pip is slow, run this once and then double-click start.bat again:
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt --no-warn-script-location -i https://pypi.tuna.tsinghua.edu.cn/simple
pause
exit /b 1

:end
endlocal
