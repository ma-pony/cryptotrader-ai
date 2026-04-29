"""FastMCP compatibility layer.

Imports the real FastMCP if available, otherwise provides a minimal stub
for tool registration and discovery (no actual MCP transport).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

try:
    from fastmcp import FastMCP  # type: ignore[import-untyped]
except Exception:
    logger.debug("fastmcp not available; using stub implementation")

    class _ToolInfo:
        def __init__(self, name: str, fn: Callable) -> None:
            self.name = name
            self.fn = fn

    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, name: str = "", **kwargs: Any) -> None:
            self.name = name
            self._tools: dict[str, _ToolInfo] = {}

        def tool(self, **kwargs: Any):
            def decorator(fn: Callable) -> Callable:
                self._tools[fn.__name__] = _ToolInfo(fn.__name__, fn)
                return fn

            return decorator

        def list_tools(self) -> list[_ToolInfo]:
            return list(self._tools.values())

        async def call_tool(self, name: str, args: dict) -> Any:
            if name not in self._tools:
                raise KeyError(f"Unknown tool: {name}")
            return await self._tools[name].fn(**args)

        def run(self, **kwargs: Any) -> None:
            pass
