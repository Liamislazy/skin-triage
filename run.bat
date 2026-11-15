@echo off
cd /d "%~dp0"
title SkinTriage

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a virtual environment and installing packages.
  echo This downloads about 300 MB and takes a few minutes...
  python -m venv .venv
  if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 ( echo Package install failed - see the messages above. & rmdir /s /q .venv & pause & exit /b 1 )
)

echo Starting SkinTriage... your browser will open at http://127.0.0.1:7860
echo Close this window to stop the app.
".venv\Scripts\python.exe" app.py --checkpoint model\best.pt --inbrowser
pause
