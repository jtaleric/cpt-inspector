# CPT Inspector

A simple chatbot web application built with FastAPI and Ollama, featuring MCP (Model Context Protocol) server integration using the official MCP Python SDK.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the Application**:
   Edit `config.json` to set your preferences:
   ```json
   {
     "ollama_url": "http://localhost:11434",
     "default_model": "llama2",
     "mcp_servers": [
       {
         "name": "My MCP Server",
         "url": "http://localhost:3000/mcp",
         "api_key": "your-api-key",
         "enabled": true
       }
     ]
   }
   ```

3. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

4. **Run the Application**:
   ```bash
   python main.py
   ```

5. **Open your browser** and navigate to `http://localhost:8000`

## Configuration

The application uses a `config.json` file for configuration:

### Ollama Settings
- `ollama_url`: URL of your Ollama server (default: `http://localhost:11434`)
- `default_model`: The Ollama model to use (default: `llama2`)

### MCP Servers
- `mcp_servers`: Array of MCP server configurations
  - `name`: Display name for the server
  - `url`: Server URL (should be the MCP endpoint)
  - `api_key`: Optional API key for authentication
  - `enabled`: Whether the server is active
