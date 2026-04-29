"""MCP configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MCPServerConfig:
    name: str = ""
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "localhost"
    port: int = 8080
    enabled: bool = True
    tools: list[str] = field(default_factory=list)


@dataclass
class MCPConfig:
    enabled: bool = False
    transport: Literal["stdio", "http"] = "stdio"
    fallback_on_error: bool = True
    call_timeout_s: float = 5.0
    servers: list[MCPServerConfig] = field(default_factory=list)
