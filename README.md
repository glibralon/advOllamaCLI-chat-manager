# Ollama Chat Manager & Session Tool (AdvOllamaCLI)

A robust, command-line interface (CLI) for interacting with local Ollama AI models. This tool goes beyond basic chat by offering persistent session management, file ingestion (Context RAG), professional export options, and detailed performance metrics.

## Key Features

* **Session Management:**
    * **Save & Resume:** Automatically saves conversation history to JSON. Resume any chat exactly where you left off.
    * **Search:** specific keywords or topics across all your past chat history.
    * **Organize:** Rename topics dynamically and delete old sessions.
* **Advanced Context:**
    * **File Ingestion:** "Read" documents into the chat. Supports **.pdf, .docx, .xlsx (Excel), .md, .txt, .py, .json, and .csv**.
    * **Long-Term Memory:** Automatically summarizes conversations when they exceed the context window so the AI doesn't "forget" earlier details.
    * **Context Control:** Manually adjust the context window size or trigger summaries.
* **Professional Export:**
    * Save any AI response or the entire chat history to **.docx (Word), .pdf, .xlsx (Excel), .md (Markdown), or .txt**.
* **Performance Metrics:**
    * Tracks **Time to First Token (TTFT)** (responsiveness) and **Total Response Time** for every message.
* **Model Management:**
    * Switch models mid-chat, list available local models, and pull new models directly from the menu.

## Prerequisites

1.  **Python (3.10+):** [Download Here](https://www.python.org/downloads/)
2.  **Ollama:** The engine required to run the AI models. [Download Here](https://ollama.com/)
    * *Note:* Ensure Ollama is running in the background (you should see the llama icon in your taskbar).

## 🛠️ Installation & Setup

### Option A: The Easy Way (Windows)
1.  **Download:** Click the green **Code** button > **Download ZIP** and extract it.
2.  **Run:** Double-click the **`run_windows.bat`** file.
    * *First Run:* This script will automatically create a virtual environment, install all dependencies, and pull the default model (`llama3`). This may take 5-10 minutes. Subsequent runs will be instant.

### Option B: Mac / Linux
1.  Open Terminal in the downloaded folder.
2.  Run the installer script:
    ```bash
    sh run_mac_linux.sh
    ```

### Option C: Manual Installation (Advanced)
If you prefer to set it up yourself:
1.  Create a virtual environment: `python -m venv venv`
2.  Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies include: `ollama`, `tiktoken`, `pandas`, `fpdf`, `python-docx`, `pypdf2`, `openpyxl`, `halo`, `tabulate`)*

## Configuration (config.json) 

The app looks for a `config.json` file to set default behaviors. You can modify this file to change settings:

```json
{
    "CONTEXT_WINDOW_SIZE": 10,           // How many past messages the AI "sees"
    "DEFAULT_SYSTEM_PROMPT": "You are...", // The default personality of the AI
    "DEFAULT_MODEL": "llama3",           // The model used if none is selected
    "CHATS_DIRECTORY": "chats",          // Folder where history is saved
    "OLLAMA_HOST": "http://localhost:11434" // URL of the Ollama server
}
```

**### How to use**  
**Main Menu**  
When you launch the app, you will see options to:  
**Start a New Chat:** Create a new session (you will be asked for a Topic name).  
**Load Existing Chat:** Choose from a list of previous conversations.  
**Delete Session:** Permanently remove old history files.  
**Search Sessions:** Find specific text across all your saved chats.  
**Manage Models:** List, delete, or pull (install) new Ollama models.  
  
**In-Chat Commands**  
While chatting, you can use these special commands:  
**EOF** - Type EOF on a new line and press Enter to send a multi-line message (great for pasting code).  
**/ingest "file.pdf"** - Reads a file and adds it to the context. Supports PDF, DOCX, Excel, Code, etc.  
**/export --save "file.docx"** - Saves the next AI response to a file.  
**/export --save "file.md" --all** - Saves the entire chat history to a file.  
**/model [name]** - Switches the active model immediately (e.g., /model mistral).  
**/topic [name]** - Renames the current session topic (and the save file).  
**/sys_prompt [text]** - Changes the System Prompt (personality) for the current session.  
**/context [number]** - Changes the memory window size (e.g., /context 20).  
**/summarize_context** - Forces a manual summary of the conversation history.  
**/bye** - Exits the session and saves your history.  
  
**Command Line Arguments**  
You can also run the script with arguments to skip the menu:  
**--ingest "file.txt":** Start a new chat immediately with this file ingested.  
**--ollama-host "url":** Connect to a remote Ollama server (e.g., http://192.168.1.50:11434).  
  
**⚠️ Troubleshooting**  
**PDF Export Issues:** If PDF export fails or looks standard, ensure DejaVuSans.ttf and DejaVuSans-Bold.ttf are present in the application folder.  
**"Ollama not found":** Ensure the Ollama application is running.  
**Excel Import:** Requires openpyxl (installed automatically by the setup script).
