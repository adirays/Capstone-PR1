@echo off
echo ===================================================
echo   Starting Manufacturing AI Dashboard Locally
echo ===================================================

echo [1/3] Starting Backend API (in new window)...
start "Manufacturing Backend" cmd /k "uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Waiting for Backend to Initialize (5 seconds)...
timeout /t 5 >nul

echo [3/3] Starting Frontend Dashboard...
streamlit run frontend/app.py

echo.
echo ===================================================
echo   App should be running!
echo   Backend: http://localhost:8000
echo   Frontend: http://localhost:8501
echo ===================================================
