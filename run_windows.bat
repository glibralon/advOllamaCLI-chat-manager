@echo off
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

:: Refresh Path (so we can use 'python' and 'ollama' immediately)
for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")

:: 2. ENSURE OLLAMA SERVER IS RUNNING [Fixes "Missing Models" in App]
echo [INFO] Starting Ollama Background Server...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if %errorlevel% neq 0 (
    start /B ollama serve >nul 2>&1
)

:: Wait for server to respond before continuing
:wait_ollama
curl -s http://localhost:11434 >nul
if %errorlevel% neq 0 (
    timeout /t 1 >nul
    goto wait_ollama
)

:: 3. VIRTUAL ENVIRONMENT & PACKAGES [Fixes "Module Not Found" errors]
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

echo [INFO] Activating environment and checking libraries...
call venv\Scripts\activate
:: Ensure tiktoken and other needs are there
pip install tiktoken ollama --upgrade --quiet
if exist "requirements.txt" pip install -r requirements.txt --upgrade --quiet

:: 4. ENSURE MODEL IS DOWNLOADED
echo [INFO] Verifying model 'neural-chat:7b'...
ollama list | findstr "neural-chat:7b" >nul
if %errorlevel% neq 0 (
    echo [INFO] Downloading model (this may take a few minutes)...
    ollama pull neural-chat:7b
)

:: 5. LAUNCH APP
echo ========================================================
echo [SUCCESS] Launching Chat App...
echo ========================================================
:: We are already inside the 'venv' here, so python will find your packages
python chat_app.py

:: Keep window open if app crashes
pause
