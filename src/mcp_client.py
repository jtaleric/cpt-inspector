"""
MCP Client module for CPT Inspector.

Provides client implementation for Model Context Protocol servers.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger("cpt-inspector.mcp")


class MCPClient:
    """MCP client for communicating with MCP servers."""

    def __init__(self, url: str):
        """Initialize MCP client with server URL."""
        self.url = url
        self.session: Optional[ClientSession] = None
        self._client_context = None

    async def _get_session(self) -> ClientSession:
        """Get or create MCP client session."""
        if self.session is None:
            try:
                logger.info("Creating streamable HTTP client for URL: %s", self.url)
                
                # Test basic connectivity first
                import httpx
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(self.url, timeout=5.0)
                        logger.info("HTTP connectivity test successful: %s", response.status_code)
                except Exception as http_error:
                    logger.warning("HTTP connectivity test failed: %s", http_error)
                
                # Create streamable HTTP client connection
                self._client_context = streamablehttp_client(self.url)
                logger.debug("Client context created, entering context")
                read_stream, write_stream, _ = await self._client_context.__aenter__()
                logger.debug("Got read/write streams from client context")
                
                # Create client session
                logger.debug("Creating ClientSession with streams")
                self.session = ClientSession(read_stream, write_stream)
                logger.debug("Entering session context")
                await self.session.__aenter__()
                
                # Initialize the session
                logger.debug("Initializing session")
                await self.session.initialize()
                logger.info("MCP session established with %s", self.url)
            except Exception as e:
                logger.error("Failed to establish MCP session with %s: %s", self.url, e)
                logger.error("Exception type: %s", type(e).__name__)
                import traceback
                logger.error("Traceback: %s", traceback.format_exc())
                # Clean up on error
                if self._client_context:
                    try:
                        await self._client_context.__aexit__(None, None, None)
                    except:
                        pass
                    self._client_context = None
                raise
        return self.session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        try:
            logger.debug("Attempting to list tools from MCP server: %s", self.url)
            session = await self._get_session()
            logger.debug("Session established, calling list_tools")
            tools_response = await session.list_tools()
            logger.info("Retrieved %d tools from MCP server", len(tools_response.tools))
            return tools_response
        except Exception as e:
            logger.error("Error listing tools from MCP server %s: %s", self.url, e)
            logger.error("Exception type: %s", type(e).__name__)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server."""
        try:
            session = await self._get_session()
            result = await session.call_tool(tool_name, arguments)
            logger.debug("Tool call result: %s", result)
            return result
        except Exception as e:
            logger.error("Error calling tool '%s': %s", tool_name, e)
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from the MCP server."""
        try:
            session = await self._get_session()
            resources_response = await session.list_resources()
            logger.debug("Retrieved %d resources from MCP server", len(resources_response.resources))
            return [resource.model_dump() for resource in resources_response.resources]
        except Exception as e:
            logger.error("Error listing resources from MCP server: %s", e)
            return []

    async def get_resource(self, resource_name: str) -> Any:
        """Get a specific resource from the MCP server."""
        try:
            session = await self._get_session()
            resource = await session.read_resource(resource_name)
            logger.debug("Resource retrieved: %s", resource)
            return resource
        except Exception as e:
            logger.error("Error getting resource '%s': %s", resource_name, e)
            raise

    async def close(self):
        """Close the MCP client session."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                self.session = None
                logger.info("MCP session closed")
            except Exception as e:
                logger.error("Error closing MCP session: %s", e)
        
        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
                self._client_context = None
                logger.info("MCP client context closed")
            except Exception as e:
                logger.error("Error closing MCP client context: %s", e)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 
