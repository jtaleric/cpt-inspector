"""
CPT Inspector - A chatbot application with Ollama and MCP integration.
"""

import json
import logging
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.llm_client import LLMClientFactory
from src.mcp_client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cpt-inspector")
logger.setLevel(logging.DEBUG)

# Load configuration
def load_config():
    """Load configuration from config.json file."""
    config_path = "config.json"
    if not os.path.exists(config_path):
        logger.warning("Config file not found, creating default config")
        default_config = {
            "ollama": {
                "url": "http://localhost:11434",
                "model": "llama3.2"
            },
            "mcp_servers": [
                {
                    "name": "default",
                    "url": "http://localhost:3000",
                    "enabled": True
                }
            ]
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        return default_config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logger.info("Configuration loaded successfully")
        logger.info("Ollama URL: %s", config_data.get('ollama', {}).get('url'))
        logger.info("Ollama Model: %s", config_data.get('ollama', {}).get('model'))
        logger.info("MCP Servers: %s", config_data.get('mcp_servers', []))
        return config_data
    except json.JSONDecodeError as e:
        logger.error("Error loading config: %s", e)
        return {}

config = load_config()

# Initialize FastAPI app
app = FastAPI(title="CPT Inspector", version="1.0.0")
templates = Jinja2Templates(directory="templates")

# Initialize LLM client
llm_factory = LLMClientFactory()
llm_client = llm_factory.create_client("ollama", config.get("ollama", {}))

# Initialize MCP servers
mcp_servers = {}
for mcp_server in config.get("mcp_servers", []):
    logger.info("MCP server: %s", mcp_server)
    if not mcp_server.get("enabled", True):
        continue
    try:
        mcp_client = MCPClient(mcp_server.get("url"))
        server_name = mcp_server.get("name", "unknown")
        mcp_servers[server_name] = mcp_client
        logger.info(
            "MCP server '%s' initialized at %s", server_name, mcp_server.get("url")
        )
    except Exception as e:
        logger.error(
            "Failed to initialize MCP server '%s': %s", mcp_server.get("name", "unknown"), e
        )

# Chat sessions storage
chat_sessions = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model": config.get("ollama", {}).get("model", "Unknown"),
            "mcp_servers": list(mcp_servers.keys()),
        },
    )

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Handle chat requests."""
    try:
        form_data = await request.form()
        message = form_data.get("prompt", "")
        session_id = form_data.get("session_id")

        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Add user message to session
        chat_sessions[session_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info("Processing chat request for session %s", session_id)
        logger.info("User message: %s", message)

        # Get response from LLM
        try:
            response = await llm_client.chat(chat_sessions[session_id], mcp_servers)
            logger.info("LLM response: %s", response)
        except Exception as e:
            logger.error("LLM error: %s", e)
            response = f"Error: {str(e)}"

        # Add assistant response to session
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })

        return {
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("Chat endpoint error: %s", e)
        return {"error": str(e)}

@app.get("/mcp/servers")
async def list_mcp_servers():
    """List available MCP servers."""
    servers = []
    for name, client in mcp_servers.items():
        try:
            status = "connected"
            tools = await client.list_tools()
            servers.append({
                "name": name,
                "url": client.url,
                "status": status,
                "tools_count": len(tools),
            })
        except Exception as e:
            logger.error("Error checking MCP server '%s': %s", name, e)
            servers.append({
                "name": name,
                "url": client.url,
                "status": "error",
                "tools_count": 0,
            })
    return {"servers": servers}

@app.get("/mcp/servers/{server_name}/tools")
async def list_mcp_tools(server_name: str):
    """List tools available from a specific MCP server."""
    if server_name not in mcp_servers:
        return {"error": f"Server '{server_name}' not found"}

    try:
        tools = await mcp_servers[server_name].list_tools()
        return {"tools": tools}
    except Exception as e:
        logger.error(
            "Error listing tools for server '%s': %s", server_name, e
        )
        return {"error": str(e)}

@app.post("/mcp/servers/{server_name}/tools/{tool_name}")
async def call_mcp_tool(server_name: str, tool_name: str, request: Request):
    """Call a specific tool on an MCP server."""
    if server_name not in mcp_servers:
        return {"error": f"Server '{server_name}' not found"}

    try:
        data = await request.json()
        arguments = data.get("arguments", {})
        result = await mcp_servers[server_name].call_tool(tool_name, arguments)
        return {"result": result}
    except Exception as e:
        logger.error(
            "Error calling tool '%s' on server '%s': %s", tool_name, server_name, e
        )
        return {"error": str(e)}

@app.get("/mcp/servers/{server_name}/resources")
async def list_mcp_resources(server_name: str):
    """List resources available from a specific MCP server."""
    if server_name not in mcp_servers:
        return {"error": f"Server '{server_name}' not found"}

    try:
        resources = await mcp_servers[server_name].list_resources()
        return {"resources": resources}
    except Exception as e:
        logger.error(
            "Error listing resources for server '%s': %s", server_name, e
        )
        return {"error": str(e)}

@app.get("/mcp/servers/{server_name}/resources/{resource_name}")
async def get_mcp_resource(server_name: str, resource_name: str):
    """Get a specific resource from an MCP server."""
    if server_name not in mcp_servers:
        return {"error": f"Server '{server_name}' not found"}

    try:
        resource = await mcp_servers[server_name].get_resource(resource_name)
        return {"resource": resource}
    except Exception as e:
        logger.error(
            "Error getting resource '%s' from server '%s': %s", resource_name, server_name, e
        )
        return {"error": str(e)}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get chat session history."""
    if session_id not in chat_sessions:
        return {"error": "Session not found"}
    return {"session": chat_sessions[session_id]}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "Session deleted"}
    return {"error": "Session not found"}

@app.get("/api/config")
async def get_config():
    """Get current application configuration."""
    return {
        "ollama": {
            "url": config.get("ollama", {}).get("url", "http://localhost:11434"),
            "model": config.get("ollama", {}).get("model", "llama3.2")
        },
        "mcp_servers": [
            {
                "name": server.get("name", "unknown"),
                "url": server.get("url", ""),
                "enabled": server.get("enabled", True)
            }
            for server in config.get("mcp_servers", [])
        ]
    }

@app.get("/api/mcp/tools")
async def get_all_mcp_tools():
    """Get all tools from all MCP servers."""
    all_tools = {}
    for server_name, client in mcp_servers.items():
        try:
            tools = await client.list_tools()
            all_tools[server_name] = tools.tools
        except Exception as e:
            logger.error("Error getting tools from server '%s': %s", server_name, e)
            all_tools[server_name] = []
    return {"tools": all_tools}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 