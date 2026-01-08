@echo off
title Ollama Chat Manager
echo ========================================================
echo Setting up Environment (Headless Mode)...
echo ========================================================

:: 1. Check/Install Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing Python...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 2. Check/Install Ollama & Block Window
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing Ollama...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    timeout /t 5 >nul
    taskkill /f /im "Ollama.exe" >nul 2>&1
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 3. Setup Virtual Env
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: 4. Activate & Install Dependencies
call venv\Scripts\activate

:: NEW: Ensure tiktoken is in requirements.txt
findstr /i "tiktoken" requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Adding tiktoken to requirements.txt...
    echo tiktoken >> requirements.txt
)

if exist "requirements.txt" (
    echo [INFO] Installing libraries...
    pip install -r requirements.txt --upgrade --quiet
)

:: 5. Pull AI Model
echo [INFO] Ensuring 'neural-chat:7b' is available...
ollama list | findstr "neural-chat:7b" >nul
if %errorlevel% neq 0 (
    ollama pull neural-chat:7b
)

:: 6. Launch Your App
echo [SUCCESS] Launching AdvOllama CLI application...
python chat_app.py

:: Keep window open if app crashes
pause

