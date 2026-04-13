@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo Python is not installed or not in PATH.
  pause
  exit /b 1
)

echo Starting local server on http://localhost:8000
python -m http.server 8000
