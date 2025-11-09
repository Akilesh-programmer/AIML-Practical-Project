@echo off
echo 🚀 Starting Universal Multi-Metal Alloy Optimizer
echo ============================================================
echo 📁 Working Directory: %CD%
echo 🌐 Server URL: http://localhost:8001
echo 📡 API Endpoint: POST /optimize
echo 🛑 Press Ctrl+C to stop
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found in PATH
    echo Please make sure Python is installed and added to PATH
    pause
    exit /b 1
)

REM Check if quick_alloy_api.py exists
if not exist "quick_alloy_api.py" (
    echo ❌ ERROR: quick_alloy_api.py not found
    echo Make sure you're running this from the sample directory
    pause
    exit /b 1
)

REM Start the server
echo 🔄 Starting API server...
python quick_alloy_api.py

echo.
echo 🛑 Server stopped
pause