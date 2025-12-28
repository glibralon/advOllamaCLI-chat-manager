#!/bin/bash
echo "Setting up Ollama Chat Environment..."

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 could not be found. Please install it."
    exit
fi

# 2. Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] Ollama is not installed. Please install it from https://ollama.com/"
    exit
fi

# 3. Create Virtual Env
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Activate & Install
source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt -q

# 5. Auto-Pull Model
# Change 'neural-chat:7b' here if your config.json uses a different default
if ! ollama list | grep -q "neural-chat:7b"; then
    echo "Model 'neural-chat:7b' not found. Downloading now... (This may take a while)"
    ollama pull neural-chat:7b
else
    echo "Model 'neural-chat:7b' is ready."
fi

# 6. Run App
echo "Launching App..."
python3 chat_app.py