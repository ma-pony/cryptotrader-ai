"""MCP (Model Context Protocol) standardized data layer.

Exports are lazy-loaded to avoid importing fastmcp at module level
(fastmcp may not be installed or may have compatibility issues).
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "MCPAdapter":
        from cryptotrader.mcp.adapter import MCPAdapter

        return MCPAdapter
    if name in ("MCPConfig", "MCPServerConfig"):
        from cryptotrader.mcp import config as _cfg

        return getattr(_cfg, name)
    if name == "MCPRegistry":
        from cryptotrader.mcp.registry import MCPRegistry

        return MCPRegistry
    if name in ("MCPToolNotFoundError", "MCPToolTimeoutError"):
        from cryptotrader.mcp import registry as _reg

        return getattr(_reg, name)
    raise AttributeError(f"module 'cryptotrader.mcp' has no attribute {name!r}")
