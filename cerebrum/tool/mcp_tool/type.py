from typing import List, Callable, Awaitable
from mcp import Tool
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack


class BaseMCPClient(ABC):
    @abstractmethod
    async def connect(self, exit_stack: AsyncExitStack):
        """Establishes connection to MCP server"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the MCP client"""
        pass

    @abstractmethod
    async def get_available_tools(self) -> List[Tool]:
        """
        Retrieve a list of available tools from the MCP server.
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str) -> Callable[..., Awaitable[str]]:
        """
        Create a callable function for a specific tool.
        This allows us to execute database operations through the MCP server.

        Args:
            tool_name: The name of the tool to create a callable for

        Returns:
            A callable async function that executes the specified tool
        """
        pass

    @abstractmethod
    async def hint(self) -> str:
        """
        Retrieve a hint for the MCP server.
        """
        pass

    @abstractmethod
    async def tool_schemas(self) -> List[dict]:
        """
        Retrieve a list of tool schemas from the MCP server.
        """
        pass