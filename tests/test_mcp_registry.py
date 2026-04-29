"""Tests for MCPRegistry — tool discovery, routing, health check."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cryptotrader.mcp.config import MCPConfig
from cryptotrader.mcp.registry import MCPRegistry, MCPToolNotFoundError


def _mock_server(tools: list[str]) -> MagicMock:
    server = MagicMock()
    mock_tools = [MagicMock(name=t) for t in tools]
    for mt, name in zip(mock_tools, tools, strict=False):
        mt.name = name
    server.list_tools.return_value = mock_tools
    return server


class TestRegisterServer:
    def test_register_builds_index(self):
        registry = MCPRegistry(MCPConfig())
        server = _mock_server(["tool_a", "tool_b"])
        registry.register_server("test", server)
        assert "tool_a" in registry.list_tools()
        assert "tool_b" in registry.list_tools()

    def test_tool_name_conflict_raises(self):
        registry = MCPRegistry(MCPConfig())
        s1 = _mock_server(["conflict_tool"])
        s2 = _mock_server(["conflict_tool"])
        registry.register_server("server1", s1)
        with pytest.raises(ValueError, match="conflict_tool"):
            registry.register_server("server2", s2)


class TestListTools:
    def test_sorted_output(self):
        registry = MCPRegistry(MCPConfig())
        registry.register_server("s1", _mock_server(["zzz_tool", "aaa_tool"]))
        tools = registry.list_tools()
        assert tools == ["aaa_tool", "zzz_tool"]

    def test_total_tools_at_least_13(self):
        registry = MCPRegistry(MCPConfig())
        registry.register_server(
            "binance", _mock_server(["binance_derivatives", "binance_funding_rate", "binance_klines"])
        )
        registry.register_server(
            "macro", _mock_server(["macro_fear_greed", "macro_btc_dominance", "macro_fred_series", "macro_etf_flow"])
        )
        registry.register_server(
            "onchain",
            _mock_server(
                ["onchain_defi_tvl", "onchain_derivatives", "onchain_exchange_netflow", "onchain_whale_transfers"]
            ),
        )
        registry.register_server("news", _mock_server(["news_rss", "news_sosovalue"]))
        assert len(registry.list_tools()) >= 13


class TestFindServer:
    def test_found(self):
        registry = MCPRegistry(MCPConfig())
        server = _mock_server(["my_tool"])
        registry.register_server("test", server)
        assert registry.find_server("my_tool") is server

    def test_not_found(self):
        registry = MCPRegistry(MCPConfig())
        assert registry.find_server("nonexistent") is None


class TestCallTool:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        registry = MCPRegistry(MCPConfig())
        server = _mock_server(["test_tool"])

        async def _call(name, args):
            return {"data": "ok"}

        server.call_tool = _call
        registry.register_server("s", server)
        result = await registry.call_tool("test_tool", {})
        assert result == {"data": "ok"}

    @pytest.mark.asyncio
    async def test_not_found_raises(self):
        registry = MCPRegistry(MCPConfig())
        with pytest.raises(MCPToolNotFoundError):
            await registry.call_tool("no_tool", {})


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_server(self):
        registry = MCPRegistry(MCPConfig())
        server = _mock_server(["t"])
        registry.register_server("healthy", server)
        status = await registry.health_check()
        assert status["healthy"] is True

    @pytest.mark.asyncio
    async def test_unhealthy_server(self):
        registry = MCPRegistry(MCPConfig())
        server = _mock_server(["t"])
        registry.register_server("broken", server)
        server.list_tools.side_effect = RuntimeError("down")
        status = await registry.health_check()
        assert status["broken"] is False
