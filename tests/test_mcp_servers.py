"""Tests for MCP Server tool discovery and invocation (mocked providers)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestBinanceMCPServer:
    def test_discovers_three_tools(self):
        from cryptotrader.mcp.servers.binance import mcp

        tools = mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "binance_derivatives" in tool_names
        assert "binance_funding_rate" in tool_names
        assert "binance_klines" in tool_names
        assert len(tool_names) == 3

    @pytest.mark.asyncio
    async def test_derivatives_returns_expected_fields(self):
        from cryptotrader.mcp.servers.binance import binance_derivatives

        mock_data = {"open_interest": 100.0, "long_short_ratio": 1.5, "taker_buy_sell_ratio": 0.8}
        with patch(
            "cryptotrader.data.providers.binance.fetch_derivatives_binance",
            new_callable=AsyncMock,
            return_value=mock_data,
        ):
            result = await binance_derivatives("BTC")
        assert result["open_interest"] == 100.0
        assert result["long_short_ratio"] == 1.5


class TestMacroMCPServer:
    def test_discovers_four_tools(self):
        from cryptotrader.mcp.servers.macro import mcp

        tools = mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert len(tool_names) == 4
        assert "macro_fear_greed" in tool_names
        assert "macro_btc_dominance" in tool_names
        assert "macro_fred_series" in tool_names
        assert "macro_etf_flow" in tool_names

    @pytest.mark.asyncio
    async def test_fear_greed_classification(self):
        from cryptotrader.mcp.servers.macro import macro_fear_greed

        with patch(
            "cryptotrader.data.macro._fetch_fear_greed",
            new_callable=AsyncMock,
            return_value=(25, [20, 22, 25, 28, 30, 32, 35]),
        ):
            result = await macro_fear_greed()
        assert result["value"] == 25
        assert result["classification"] == "Fear"


class TestOnchainMCPServer:
    def test_discovers_four_tools(self):
        from cryptotrader.mcp.servers.onchain import mcp

        tools = mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert len(tool_names) == 4
        assert "onchain_defi_tvl" in tool_names

    @pytest.mark.asyncio
    async def test_no_api_key_returns_unavailable(self):
        from cryptotrader.mcp.servers.onchain import onchain_derivatives

        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.providers.coinglass_api_key = ""
            result = await onchain_derivatives("BTC")
        assert result["data_available"] is False
        assert result["open_interest"] == 0.0


class TestNewsMCPServer:
    def test_discovers_two_tools(self):
        from cryptotrader.mcp.servers.news import mcp

        tools = mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert len(tool_names) == 2
        assert "news_rss" in tool_names
        assert "news_sosovalue" in tool_names

    @pytest.mark.asyncio
    async def test_rss_wraps_sync_function(self):
        from cryptotrader.mcp.servers.news import news_rss

        articles = [{"title": "BTC hits 100k", "source": "coindesk", "published": "2026-01-01"}]
        with patch(
            "cryptotrader.data.providers.rss_news.fetch_crypto_news",
            return_value=articles,
        ):
            result = await news_rss(5)
        assert result["count"] == 1
        assert result["articles"][0]["title"] == "BTC hits 100k"


class TestTruncateResponse:
    def test_small_response_unchanged(self):
        from cryptotrader.mcp.utils import truncate_response

        data = {"key": "value"}
        assert truncate_response(data) == data

    def test_large_response_truncated(self):
        from cryptotrader.mcp.utils import truncate_response

        large_list = [{"data": "x" * 100} for _ in range(1000)]
        data = {"items": large_list}
        result = truncate_response(data, max_bytes=1024)
        assert result.get("truncated") is True
        assert len(result["items"]) < 1000
