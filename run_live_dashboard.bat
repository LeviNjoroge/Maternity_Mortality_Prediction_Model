@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo Python is not installed or not in PATH.
  pause
  exit /b 1
)

start "MMR Backend API" cmd /k "cd /d \"%~dp0\" && python notebook_backend_service.py --host 127.0.0.1 --port 5000"
start "MMR Dashboard Server" cmd /k "cd /d \"%~dp0\" && python -m http.server 8000"

echo.
echo Live stack started.
echo Backend : http://127.0.0.1:5000/api/health
echo Frontend: http://127.0.0.1:8000/analytics_dashboard.html
echo.
pause
