"""MCP server registry — single ``FastMCP`` class with a stable shape.

The project uses two surfaces:

* ``MCPRegistry`` (registry.py) needs ``server.list_tools()`` (sync, returns
  objects with ``.name``) and ``server.call_tool(name, args)`` (async, returns
  dict-or-scalar) for in-process dispatch.
* Tests import the decorated functions directly and call them as plain
  callables: ``from servers.binance import binance_derivatives; await ...``.

Upstream ``fastmcp>=2.0`` no longer satisfies either surface — its decorator
returns a ``FunctionTool`` wrapper that is not directly callable, and tool
listing / invocation are async with different names. Rather than monkey-patch
a third-party class, we own a small ``FastMCP`` here that is purpose-built for
those two surfaces. If a future feature needs to actually expose tools over
the MCP wire protocol (stdio / HTTP), wrap an instance with the real
``fastmcp`` server at *that* call site (not at decoration time).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _ToolInfo:
    __slots__ = ("fn", "name")

    def __init__(self, name: str, fn: Callable) -> None:
        self.name = name
        self.fn = fn


class FastMCP:
    """In-process tool registry — keeps decorated functions directly callable."""

    def __init__(self, name: str = "", **_: Any) -> None:
        self.name = name
        self._tools: dict[str, _ToolInfo] = {}

    def tool(self, **_: Any):
        """Decorator that registers ``fn`` under ``fn.__name__`` and returns it unchanged.

        Returning the original function (rather than a wrapper) is deliberate:
        tests and other call sites import the decorated symbol directly and
        invoke it with positional / keyword args.
        """

        def decorator(fn: Callable) -> Callable:
            self._tools[fn.__name__] = _ToolInfo(fn.__name__, fn)
            return fn

        return decorator

    def list_tools(self) -> list[_ToolInfo]:
        return list(self._tools.values())

    async def call_tool(self, name: str, args: dict) -> Any:
        info = self._tools.get(name)
        if info is None:
            raise KeyError(f"Unknown tool: {name}")
        return await info.fn(**args)

    def run(self, **_: Any) -> None:
        """Placeholder — wire up real fastmcp transport here when needed."""
        return
