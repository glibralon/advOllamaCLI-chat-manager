@echo off
setlocal enabledelayedexpansion
title Ollama Chat Manager (Stable Version)
echo ========================================================
echo Initializing System...
echo ========================================================

:: 1. Check/Install Python
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Installing Python 3.11...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 2. Check/Install Ollama
where ollama >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Installing Ollama...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    timeout /t 5 >nul
    taskkill /f /im "Ollama.exe" >nul 2>&1
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 3. Start Ollama Server
echo [INFO] Starting Ollama Server...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if !errorlevel! neq 0 (
    start /B ollama serve >nul 2>&1
)

:: Wait for server API to respond
:wait_server
curl -s http://localhost:11434 >nul
if !errorlevel! neq 0 (
    echo [INFO] Waiting for server...
    timeout /t 2 >nul
    goto wait_server
)

:: 4. Virtual Environment & Dependencies
if not exist "venv" (
    python -m venv venv || (echo [ERROR] Failed to create venv. && pause && exit)
)
call venv\Scripts\activate
echo [INFO] Syncing libraries...
:: Fix for tiktoken and ollama python library
pip install tiktoken ollama --upgrade --quiet
if exist "requirements.txt" pip install -r requirements.txt --upgrade --quiet

:: 5. CRASH-PROOF MODEL CHECK
echo [INFO] Checking for model: neural-chat:7b...
:: Instead of a pipe, we save the list to a temporary file to prevent crashes
ollama list > models.tmp 2>&1
findstr "neural-chat:7b" models.tmp >nul
if !errorlevel! neq 0 (
    echo [INFO] Model not found. Pulling 'neural-chat:7b' (approx 4GB)...
    echo This may take several minutes depending on your internet.
    ollama pull neural-chat:7b
) else (
    echo [INFO] Model is already installed.
)
del models.tmp

:: 6. Launch App [cite: 8]
echo ========================================================
echo [SUCCESS] Launching Chat Application...
echo ========================================================
if exist "chat_app.py" (
    python chat_app.py
) else (
    echo [ERROR] chat_app.py missing. Please put this .bat file in your app folder.
    dir /b
)

:: Keep window open if app crashes
pause


