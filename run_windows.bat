@echo off
title Ollama Chat Manager Launcher
echo ========================================================
echo Setting up the Ollama Chat Environment...
echo ========================================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed! Please install Python 3.10+ from python.org.
    pause
    exit
)

:: 2. Check Ollama (Crucial: App won't work without it)
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not installed or not in your PATH.
    echo Please download and install it from https://ollama.com/
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
echo Checking for updates and installing libraries...
pip install -r requirements.txt --upgrade --quiet

:: 5. Auto-Pull Default Model (neural-chat:7b)
:: You can change 'neural-chat:7b' below to whatever you set in config.json
echo Checking for default AI model (neural-chat:7b)...
ollama list | findstr "neural-chat:7b" >nul
if %errorlevel% neq 0 (
    echo Model 'neural-chat:7b' not found. Downloading now... (This may take a while)
    ollama pull neural-chat:7b
) else (
    echo Model 'neural-chat:7b' is ready.
)

:: 6. Run the App
echo.
echo Launching Chat App...
echo.
python chat_app.py

:: Keep window open if app crashes
pause