@echo off
cd /d "%~dp0"

python notebook_backend_service.py --host 127.0.0.1 --port 5000
