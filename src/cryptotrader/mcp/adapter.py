"""MCPAdapter — three-level routing: MCP → Python fallback → zero-value."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cryptotrader.mcp.registry import MCPToolNotFoundError, MCPToolTimeoutError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from cryptotrader.mcp.config import MCPConfig
    from cryptotrader.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)


class MCPAdapter:
    def __init__(self, registry: MCPRegistry, config: MCPConfig) -> None:
        self._registry = registry
        self._config = config

    async def call(
        self,
        tool_name: str,
        args: dict,
        *,
        backtest_mode: bool = False,
        python_fallback: Callable[..., Awaitable[dict]] | None = None,
        fallback_args: dict[str, Any] | None = None,
        zero_value: dict | None = None,
    ) -> dict:
        _zero = zero_value or {}
        _fb_args = fallback_args or {}

        if backtest_mode:
            if python_fallback is not None:
                return await python_fallback(**_fb_args)
            return _zero

        if not self._config.enabled:
            if python_fallback is not None:
                return await python_fallback(**_fb_args)
            return _zero

        try:
            return await self._registry.call_tool(tool_name, args)
        except (MCPToolNotFoundError, MCPToolTimeoutError, Exception) as exc:
            logger.warning("[MCP fallback] %s failed: %s", tool_name, exc)
            if self._config.fallback_on_error and python_fallback is not None:
                return await python_fallback(**_fb_args)
            return _zero
