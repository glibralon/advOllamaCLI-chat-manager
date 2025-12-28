# advOllamaCLI-chat-manager
A powerful CLI chat manager for Ollama with export features.

# Ollama Chat Manager & Session Tool

A robust, local AI chat interface for Ollama. This tool allows you to chat with different AI models, manage long conversations (with memory), and export your chats to professional formats like PDF, Word, and Excel.

## Key Features
* **Context Memory:** Automatically summarizes long conversations so the AI doesn't "forget."
* **File Ingestion:** Type `/ingest "my_document.pdf"` to let the AI read your files.
* **Professional Exports:** Save any response or full chat to `.pdf`, `.docx`, `.xlsx`, or `.md`.
* **Session Management:** Save, load, and search through past conversations.

## Prerequisites
Before running this tool, you must have two things installed:

1.  **Python (3.10 or newer):** [Download from python.org](https://www.python.org/downloads/)
2.  **Ollama:** [Download from ollama.com](https://ollama.com/)
    * *Note:* Ensure Ollama is running in the background (you should see the little llama icon in your taskbar).

## Installation & Setup

### 1. Download
Click the green **Code** button above and select **Download ZIP**. Extract the folder to your Desktop or Documents.

### 2. Run the Auto-Installer
We have included a "double-click" script that sets up everything for you (libraries, virtual environment, and model).

**For Windows Users:**
1.  Open the extracted folder.
2.  Double-click **`run_windows.bat`**.
3.  **Wait:** The first time you run this, it may take **5-10 minutes** to download the default AI model (`llama3`). Please be patient; subsequent runs will be instant.

**For Mac/Linux Users:**
1.  Open your Terminal.
2.  Navigate to the folder: `cd path/to/folder`
3.  Run the script: `sh run_mac_linux.sh`

## How to Use
Once the chat window opens, you can type normally. Here are the special commands:

* **/model [name]**: Switch AI models (e.g., `/model mistral`).
* **/ingest [filepath]**: Read a local file (PDF, DOCX, XLSX, TXT) into the chat.
* **/export --save [filename]**: Save the *next* AI response to a file.
    * *Example:* `/export --save meeting_notes.pdf`
* **/export --save [filename] --all**: Save the *entire* conversation history.
* **/context [number]**: Change how many past messages the AI remembers.
* **/bye**: Exit the application.

## Troubleshooting
* **PDF Export looks wrong?** Ensure `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` are in the same folder as the script. (They are included in this repository).
* **"Ollama not found"?** Make sure you installed Ollama and it is running in your system tray.
