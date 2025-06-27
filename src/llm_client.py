"""
LLM Client module for CPT Inspector.

Provides client implementations for various LLM providers including Ollama.
"""

import logging
import pprint
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ollama import AsyncClient

logger = logging.getLogger("cpt-inspector.llm")


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], mcp_servers: Optional[Dict[str, Any]] = None) -> str:
        """Send a chat message and return the response."""
        pass


class OllamaClient(LLMClient):
    """Ollama LLM client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama client with configuration."""
        self.url = config.get("url", "http://localhost:11434")
        self.model = config.get("model", "llama3.2")
        self.client = AsyncClient(host=self.url)
        # Convert MCP servers list to dictionary for compatibility
        mcp_servers_list = config.get("mcp_servers", [])
        self.mcp_servers = {}
        for server in mcp_servers_list:
            if server.get("enabled", True):
                self.mcp_servers[server.get("name", "unknown")] = server

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP servers."""
        available_tools = []
        for mcp_name, mcp_client in self.mcp_servers.items():
            try:
                server_tools = await mcp_client.list_tools()
                available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                } for tool in server_tools.tools]
                logger.info("Found %d tools from server %s: %s", len(available_tools), mcp_name, available_tools)
            except Exception as e:
                logger.error("Error listing tools from server %s: %s", mcp_name, e)
        return available_tools

    async def chat(self, messages: List[Dict[str, str]], mcp_servers: Optional[Dict[str, Any]] = None) -> str:
        """Send a chat message and return the response."""
        self.mcp_servers = mcp_servers or {}
        tools = await self.list_tools()
        content = ""
        while True:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
            )
            content = response['message']['content']
            logger.info("OllamaClient.chat: response: %s", response)
            tool_calls = response['message'].get('tool_calls', [])
            if not tool_calls:
                logger.info("No tool calls found in response")
                break
            for tool_call in tool_calls:
                logger.info("OllamaClient detected tool call: %s", tool_call)
                # Extract the Function object from the ToolCall
                if hasattr(tool_call, 'function'):
                    function_obj = tool_call.function
                    logger.info("Extracted function: %s", function_obj)
                    tool_result = await self._call_mcp_tool(function_obj, mcp_servers)
                    logger.info("MCP tool result: %s", tool_result)
                    messages.append({"role": "tool", "content": str(tool_result)})
                else:
                    logger.warning("ToolCall does not have function attribute: %s", tool_call)
            continue
        return content

    async def _call_mcp_tool(self, tool_call: dict, mcp_servers: Dict[str, Any]) -> Any:
        """Call MCP tool based on parsed tool call."""
        logger.info("OllamaClient._call_mcp_tool: tool_call: %s", tool_call)
        pprint.pprint(tool_call)
        # Handle Function objects from Ollama client
        if hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
            tool_name = tool_call.name
            args = tool_call.arguments
            logger.info("Function object - name: %s, arguments: %s", tool_name, args)
        else:
            # Fallback to dictionary format
            tool_name = tool_call.get('tool') or tool_call.get('name')
            args = tool_call.get('args', {}) or tool_call.get('arguments', {})
            logger.info("Dictionary format - tool_name: %s, args: %s", tool_name, args)
        logger.info("Final extracted tool_name: %s, args: %s", tool_name, args)
        # For now, just use the first enabled MCP server
        for server_name, server in mcp_servers.items():
            if getattr(server, 'enabled', True):
                try:
                    logger.info("Calling MCP tool '%s' with args: %s", tool_name, args)
                    return await server.call_tool(tool_name, args)
                except Exception as e:
                    logger.error("Error calling MCP tool '%s': %s", tool_name, e)
                    return {"error": str(e)}
        return {"error": f"No enabled MCP server found for tool '{tool_name}'"}


class LLMClientFactory:
    """Factory for creating LLM clients."""
    @staticmethod
    def create_client(client_type: str, config: Dict[str, Any]) -> LLMClient:
        """Create an LLM client of the specified type."""
        if client_type == "ollama":
            return OllamaClient(config)
        else:
            raise ValueError(f"Unsupported LLM client type: {client_type}") 