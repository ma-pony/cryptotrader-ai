"""Tests for MCPAdapter — three-level routing, backtest short-circuit, fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.mcp.adapter import MCPAdapter
from cryptotrader.mcp.config import MCPConfig
from cryptotrader.mcp.registry import MCPRegistry, MCPToolTimeoutError


def _make_adapter(*, enabled: bool = True, fallback_on_error: bool = True) -> MCPAdapter:
    config = MCPConfig(enabled=enabled, fallback_on_error=fallback_on_error)
    registry = MagicMock(spec=MCPRegistry)
    return MCPAdapter(registry, config)


class TestBacktestShortCircuit:
    @pytest.mark.asyncio
    async def test_backtest_uses_python_fallback(self):
        adapter = _make_adapter()
        fallback = AsyncMock(return_value={"source": "python"})
        result = await adapter.call("any_tool", {}, backtest_mode=True, python_fallback=fallback)
        assert result == {"source": "python"}
        fallback.assert_awaited_once()
        adapter._registry.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_backtest_no_fallback_returns_zero(self):
        adapter = _make_adapter()
        result = await adapter.call("any_tool", {}, backtest_mode=True, zero_value={"v": 0})
        assert result == {"v": 0}


class TestMCPDisabled:
    @pytest.mark.asyncio
    async def test_disabled_uses_python_fallback(self):
        adapter = _make_adapter(enabled=False)
        fallback = AsyncMock(return_value={"source": "direct"})
        result = await adapter.call("tool", {}, python_fallback=fallback)
        assert result == {"source": "direct"}
        adapter._registry.call_tool.assert_not_called()


class TestMCPEnabled:
    @pytest.mark.asyncio
    async def test_success_path(self):
        adapter = _make_adapter()
        adapter._registry.call_tool = AsyncMock(return_value={"data": "mcp"})
        result = await adapter.call("tool", {"k": "v"})
        assert result == {"data": "mcp"}
        adapter._registry.call_tool.assert_awaited_once_with("tool", {"k": "v"})

    @pytest.mark.asyncio
    async def test_timeout_fallback(self):
        adapter = _make_adapter()
        adapter._registry.call_tool = AsyncMock(side_effect=MCPToolTimeoutError("timeout"))
        fallback = AsyncMock(return_value={"source": "fallback"})
        result = await adapter.call("tool", {}, python_fallback=fallback)
        assert result == {"source": "fallback"}

    @pytest.mark.asyncio
    async def test_error_fallback(self):
        adapter = _make_adapter()
        adapter._registry.call_tool = AsyncMock(side_effect=RuntimeError("crash"))
        fallback = AsyncMock(return_value={"source": "fallback"})
        result = await adapter.call("tool", {}, python_fallback=fallback)
        assert result == {"source": "fallback"}

    @pytest.mark.asyncio
    async def test_no_fallback_on_error_returns_zero(self):
        adapter = _make_adapter(fallback_on_error=False)
        adapter._registry.call_tool = AsyncMock(side_effect=RuntimeError("crash"))
        result = await adapter.call("tool", {}, zero_value={"v": 0})
        assert result == {"v": 0}


class TestFallbackLogging:
    @pytest.mark.asyncio
    async def test_fallback_logs_warning(self, caplog):
        import logging

        adapter = _make_adapter()
        adapter._registry.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        fallback = AsyncMock(return_value={})
        with caplog.at_level(logging.WARNING):
            await adapter.call("broken_tool", {}, python_fallback=fallback)
        assert "[MCP fallback]" in caplog.text
        assert "broken_tool" in caplog.text
