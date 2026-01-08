@echo off
setlocal enabledelayedexpansion
title Ollama Chat Manager
echo ========================================================
echo Initializing Environment...
echo ========================================================

:: 1. Check/Install Python & Ollama
python --version >nul 2>&1 || (
    echo [INFO] Installing Python...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
)
ollama --version >nul 2>&1 || (
    echo [INFO] Installing Ollama...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
)

:: Refresh Path so new installs are recognized
for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")

:: 2. START & VERIFY OLLAMA SERVER [Fixes the crash]
echo [INFO] Ensuring Ollama Server is running...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if !errorlevel! neq 0 (
    echo [INFO] Starting background engine...
    start /B ollama serve >nul 2>&1
)

:: Wait for the API to respond before trying to list models
:wait_server
curl -s http://localhost:11434 >nul
if !errorlevel! neq 0 (
    echo [INFO] Waiting for engine to wake up...
    timeout /t 2 >nul
    goto wait_server
)

:: 3. Setup Virtual Env & Dependencies
if not exist "venv" (python -m venv venv)
call venv\Scripts\activate
echo [INFO] Syncing libraries (tiktoken, etc)...
pip install tiktoken ollama --upgrade --quiet
if exist "requirements.txt" pip install -r requirements.txt --upgrade --quiet

:: 4. SAFE MODEL CHECK
echo [INFO] Checking for model: neural-chat:7b...
ollama list > models.tmp 2>&1
findstr "neural-chat:7b" models.tmp >nul
if !errorlevel! neq 0 (
    echo [INFO] Model not found. Pulling 'neural-chat:7b'...
    ollama pull neural-chat:7b
)
del models.tmp

:: 5. Launch App
echo [SUCCESS] Launching Chat App...
python chat_app.py

:: Keep window open if app crashes
pause



