"""MCPRegistry — tool discovery, routing, and health check."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.mcp.config import MCPConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPToolTimeoutError(TimeoutError):
    pass


class MCPToolNotFoundError(KeyError):
    pass


class MCPRegistry:
    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._servers: dict[str, Any] = {}
        self._tool_index: dict[str, str] = {}

    def register_server(self, name: str, server: Any) -> None:
        tools = server.list_tools()
        for tool in tools:
            tool_name = tool.name if hasattr(tool, "name") else str(tool)
            if tool_name in self._tool_index:
                raise ValueError(
                    f"Tool name conflict: '{tool_name}' already registered by "
                    f"'{self._tool_index[tool_name]}', cannot register for '{name}'"
                )
            self._tool_index[tool_name] = name
        self._servers[name] = server

    def list_tools(self) -> list[str]:
        return sorted(self._tool_index.keys())

    def find_server(self, tool_name: str) -> Any | None:
        server_name = self._tool_index.get(tool_name)
        if server_name is None:
            return None
        return self._servers.get(server_name)

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        server = self.find_server(tool_name)
        if server is None:
            raise MCPToolNotFoundError(tool_name)
        timeout = self._config.call_timeout_s
        try:
            result = await asyncio.wait_for(
                server.call_tool(tool_name, args),
                timeout=timeout,
            )
            if isinstance(result, dict):
                return result
            return {"result": result}
        except TimeoutError as exc:
            raise MCPToolTimeoutError(f"Tool '{tool_name}' timed out after {timeout}s") from exc

    async def health_check(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for name, server in self._servers.items():
            try:
                server.list_tools()
                status[name] = True
            except Exception:
                status[name] = False
        return status

    @classmethod
    def from_config(cls, config: MCPConfig) -> MCPRegistry:
        registry = cls(config)
        server_configs: list[MCPServerConfig] = config.servers
        for sc in server_configs:
            if not sc.enabled:
                continue
            server = _create_server(sc.name)
            if server is not None:
                registry.register_server(sc.name, server)
        return registry


def _create_server(name: str) -> Any | None:
    try:
        if name == "cryptotrader-binance":
            from cryptotrader.mcp.servers.binance import mcp

            return mcp
        if name == "cryptotrader-macro":
            from cryptotrader.mcp.servers.macro import mcp

            return mcp
        if name == "cryptotrader-onchain":
            from cryptotrader.mcp.servers.onchain import mcp

            return mcp
        if name == "cryptotrader-news":
            from cryptotrader.mcp.servers.news import mcp

            return mcp
    except Exception:
        logger.warning("Failed to create MCP server '%s'", name, exc_info=True)
    return None
