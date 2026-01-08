@echo off
title Ollama Chat Manager - All-in-One Installer
echo ========================================================
echo Setting up the Ollama Chat Environment...
echo ========================================================

:: 1. Check/Install Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python is missing. Installing Python 3.11...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    
    :: FORCE REFRESH PATH (The Trick)
    echo [INFO] Refreshing system paths...
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%i"
)

:: 2. Check/Install Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama is missing. Installing...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    
    :: FORCE REFRESH PATH AGAIN
    echo [INFO] Refreshing system paths...
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%i"
)

:: 3. Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: 4. Activate & Install Dependencies
call venv\Scripts\activate
if exist "requirements.txt" (
    echo [INFO] Installing libraries from requirements.txt...
    pip install -r requirements.txt --upgrade --quiet
) else (
    echo [WARNING] requirements.txt not found. Skipping pip install.
)

:: 5. Pull Model
echo [INFO] Ensuring model 'neural-chat:7b' is available...
ollama list | findstr "neural-chat:7b" >nul
if %errorlevel% neq 0 (
    echo [INFO] Downloading model (this may take a few minutes)...
    ollama pull neural-chat:7b
)

:: 6. Launch App
echo ========================================================
echo [SUCCESS] Everything is ready! Launching Chat App...
echo ========================================================
if exist "chat_app.py" (
    python chat_app.py
) else (
    echo [ERROR] chat_app.py not found in this folder!
)

:: Keep window open if app crashes
pause
