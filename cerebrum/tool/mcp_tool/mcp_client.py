from functools import wraps
from mcp import ClientSession, StdioServerParameters, Tool
from typing import List, Dict, Callable, Awaitable, Optional, Any
from mcp.client.stdio import stdio_client
from datetime import datetime
from .type import BaseMCPClient
from .utils import CONSOLE
import asyncio
from contextlib import AsyncExitStack


class MCPClient(BaseMCPClient):
    """
    A client class for interacting with the MCP (Model Control Protocol) server.
    This class manages the connection and communication with the SQLite database through MCP.
    """

    @classmethod
    def from_smithery(cls, pkg_name: str, suffix_args: List[str] = []):
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@smithery/cli@latest", "run", pkg_name, *suffix_args],
        )
        # CONSOLE.log(f"Use MCP: {pkg_name} from smithery")
        return cls(pkg_name, server_params)

    @classmethod
    def from_npx(
        cls, pkg_name: str, prefix_args: List[str] = [], suffix_args: List[str] = []
    ):
        server_params = StdioServerParameters(
            command="npx",
            args=[*prefix_args, pkg_name, *suffix_args],
            # **extra_args
        )
        print(f"Use MCP: {pkg_name} from npx")
        return cls(pkg_name, server_params)

    @classmethod
    def from_docker(
        cls,
        image_name: str,
        volume_mounts: List[str] = [],
        docker_args: List[str] = [],
        docker_env: Dict[str, str] = None,
    ):
        server_params = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",  # Remove container after exit
                "-i",  # Interactive mode
                *volume_mounts,
                image_name,  # Use SQLite MCP image
                *docker_args,
            ],
            env=docker_env,
        )
        # CONSOLE.log(f"Use MCP: {image_name} from docker")
        return cls(image_name, server_params)

    def __init__(self, name: str, server_params: StdioServerParameters):
        """Initialize the MCP client with server parameters"""
        self.__name = name
        self.server_params = server_params
        self.session = None
        self.read = None
        self.write = None
        self._exit_stack = None
        self._connected = False

    @property
    def name(self) -> str:
        return self.__name

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.disconnect()

    async def connect(self, exit_stack: AsyncExitStack):
        if self._connected:
            return

        CONSOLE.print(f"Connect to MCP: {self.name}")

        # Create an AsyncExitStack to properly manage all async context managers
        # self._exit_stack = AsyncExitStack()

        # Use the exit stack to enter the stdio_client context
        # This ensures proper cleanup when we exit the stack
        client_ctx = stdio_client(self.server_params)
        self.read, self.write = await exit_stack.enter_async_context(client_ctx)

        # Create and enter the session context
        session_ctx = ClientSession(self.read, self.write)
        self.session = await exit_stack.enter_async_context(session_ctx)

        # Initialize the session
        await self.session.initialize()
        self._connected = True

    async def disconnect(self):
        if not self._connected or not self._exit_stack:
            return
        # Use the exit stack to properly clean up all contexts
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            CONSOLE.log(f"Error during disconnect for {self.name}: {e}")

        # Clear all references
        self.session = None
        self.read = None
        self.write = None
        self._exit_stack = None
        self._connected = False

        CONSOLE.log(f"Disconnected from {self.name}")

    async def get_available_tools(self) -> List:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        return tools.tools

    def call_tool(self, tool_name: str) -> Callable[..., Awaitable[str]]:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable

    async def hint(self) -> str:
        tools = await self.get_available_tools()
        hint = ""
        for tool in tools:
            hint += f"- {tool.name}: {tool.description}\n"
        return hint

    async def tool_schemas(self) -> List[dict]:
        openai_tools = []
        tools = await self.get_available_tools()
        for tool in tools:
            schema = tool.inputSchema
            schema.pop("$schema")
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                }
            )
        return openai_tools