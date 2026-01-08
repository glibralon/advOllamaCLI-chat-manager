@echo off
setlocal enabledelayedexpansion
title Ollama Chat Manager - Final Fix
echo ========================================================
echo Initializing Environment...
echo ========================================================

:: 1. Check Python & Ollama
python --version >nul 2>&1 || (echo [ERROR] Install Python! && pause && exit)
ollama --version >nul 2>&1 || (echo [ERROR] Install Ollama! && pause && exit)

:: 2. START THE OLLAMA SERVER (The Engine)
echo [INFO] Ensuring Ollama Server is running...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if !errorlevel! neq 0 (
    echo [INFO] Starting background server...
    start /B ollama serve >nul 2>&1
)

:: 3. WAIT FOR SERVER TO RESPOND
echo [INFO] Waiting for server to wake up...
:wait_server
curl -s http://localhost:11434 >nul
if !errorlevel! neq 0 (
    timeout /t 2 >nul
    goto wait_server
)
echo [SUCCESS] Server is responding.

:: 4. THE MODEL PULL
echo [INFO] Checking for model: neural-chat:7b...
ollama list > models_check.tmp 2>&1
findstr "neural-chat:7b" models_check.tmp >nul
if !errorlevel! neq 0 (
    echo [WARNING] Model NOT found. Starting download (4GB)...
    echo This will take time. DO NOT CLOSE THIS WINDOW.
    ollama pull neural-chat:7b
) else (
    echo [SUCCESS] Model found on system.
)
del models_check.tmp

:: 5. SHOW MODELS
echo.
echo Your current Ollama models:
ollama list
echo.

:: 6. Setup VENV and Launch
if not exist "venv" (python -m venv venv)
call venv\Scripts\activate
pip install tiktoken ollama --upgrade --quiet
if exist "requirements.txt" pip install -r requirements.txt --upgrade --quiet

echo [INFO] Launching chat_app.py...
python chat_app.py

:: Keep window open if app crashes
pause
