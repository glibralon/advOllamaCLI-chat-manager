@echo off
title Ollama Chat Manager Launcher
echo ========================================================
echo Setting up the Ollama Chat Environment...
echo ========================================================

:: 1. Check/Install Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python is not installed. Attempting to install Python 3.11 via winget...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [ERROR] Automatic Python installation failed. Please install it manually from python.org. [cite: 1]
        pause
        exit
    )
    echo [SUCCESS] Python installed. Please RESTART this script to refresh system paths.
    pause
    exit
)

:: 2. Check/Install Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama is not installed. Attempting to install via winget...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo [ERROR] Automatic Ollama installation failed. Please download it from https://ollama.com/ 
        pause
        exit
    )
    echo [SUCCESS] Ollama installed. Please RESTART this script to refresh system paths.
    pause
    exit
)

:: 3. Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment... 
    python -m venv venv 
)

:: 4. Activate & Install Dependencies
call venv\Scripts\activate 
if exist "requirements.txt" (
    echo Checking for updates and installing libraries... 
    pip install -r requirements.txt --upgrade --quiet 
) else (
    echo [WARNING] requirements.txt not found. Skipping library installation.
)

:: 5. Auto-Pull Default Model (neural-chat:7b)
echo Checking for default AI model (neural-chat:7b)... [cite: 3]
ollama list | findstr "neural-chat:7b" >nul [cite: 3]
if %errorlevel% neq 0 (
    echo Model 'neural-chat:7b' not found. Downloading now... (This may take a while) [cite: 3]
    ollama pull neural-chat:7b [cite: 3]
) else (
    echo Model 'neural-chat:7b' is ready. [cite: 3]
)

:: 6. Run the App
echo.
echo Launching Chat App... 
echo.
if exist "chat_app.py" (
    python chat_app.py 
) else (
    echo [ERROR] chat_app.py not found in this directory.
)

:: Keep window open if app crashes
pause
