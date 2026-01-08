@echo off
title Ollama Chat Manager (Debug Mode)
echo ========================================================
echo Starting System Diagnostics...
echo ========================================================

:: 1. Check/Install Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [DEBUG] Python missing. Installing...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 2. Check/Install Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [DEBUG] Ollama missing. Installing...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    timeout /t 5 >nul
    taskkill /f /im "Ollama.exe" >nul 2>&1
    for /f "tokens=* usebackq" %%p in (`powershell -Command "& {[System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')}"`) do (set "PATH=%%p")
)

:: 3. Force Start Ollama Server (Important for listing models)
echo [DEBUG] Checking Ollama Server...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if %errorlevel% neq 0 (
    start /B ollama serve >nul 2>&1
)

:: Wait for server to be responsive
:wait_server
curl -s http://localhost:11434 >nul
if %errorlevel% neq 0 (
    echo [DEBUG] Waiting for Ollama API...
    timeout /t 2 >nul
    goto wait_server
)

:: 4. Virtual Environment & Dependencies
if not exist "venv" (
    echo [DEBUG] Creating VENV...
    python -m venv venv || (echo [ERROR] VENV creation failed! && pause && exit)
)

call venv\Scripts\activate
echo [DEBUG] Installing requirements...
:: Force tiktoken install here since it was missing before
pip install tiktoken --quiet
if exist "requirements.txt" (
    pip install -r requirements.txt --upgrade --quiet || (echo [ERROR] Pip install failed! && pause && exit)
)

:: 5. Model Pull (This is where scripts often "hang")
echo [DEBUG] Checking Model...
ollama list | findstr "neural-chat:7b" >nul
if %errorlevel% neq 0 (
    echo [DEBUG] Pulling model (Please wait, this looks like it's doing nothing but it is)...
    ollama pull neural-chat:7b || (echo [ERROR] Model pull failed! && pause && exit)
)

:: 6. Launching App
echo ========================================================
echo [DEBUG] Reached the end! Launching chat_app.py...
echo ========================================================
if exist "chat_app.py" (
    python chat_app.py
) else (
    echo [ERROR] chat_app.py NOT FOUND in this folder!
    dir /b
)

:: Keep window open if app crashes
pause

