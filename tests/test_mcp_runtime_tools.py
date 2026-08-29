"""Database-runtime MCP products retain their explicit credential boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def test_runtime_mcp_servers_register_every_product_tool_without_import_side_effects():
    from cryptotrader.mcp.servers.macro import mcp as macro
    from cryptotrader.mcp.servers.news import mcp as news
    from cryptotrader.mcp.servers.onchain import mcp as onchain

    assert {tool.name for tool in macro.list_tools()} == {
        "macro_fear_greed",
        "macro_btc_dominance",
        "macro_fred_series",
        "macro_etf_flow",
    }
    assert {tool.name for tool in news.list_tools()} == {"news_rss", "news_sosovalue"}
    assert {tool.name for tool in onchain.list_tools()} == {
        "onchain_defi_tvl",
        "onchain_derivatives",
        "onchain_exchange_netflow",
        "onchain_whale_transfers",
    }


@pytest.mark.asyncio
async def test_credentialed_mcp_tools_do_not_read_global_config_and_require_explicit_keys():
    from cryptotrader.mcp.servers.macro import macro_fred_series
    from cryptotrader.mcp.servers.news import news_sosovalue
    from cryptotrader.mcp.servers.onchain import onchain_derivatives

    assert await macro_fred_series("VIXCLS") == {"value": 0.0, "series_id": "VIXCLS", "data_available": False}
    assert await news_sosovalue() == {"articles": [], "count": 0, "data_available": False}
    assert await onchain_derivatives() == {"open_interest": 0.0, "liquidations_24h": {}, "data_available": False}

    with patch(
        "cryptotrader.data.providers.coinglass.fetch_derivatives",
        new=AsyncMock(return_value={"open_interest": 42}),
    ):
        assert await onchain_derivatives("BTC", provider_key="explicit-key") == {
            "open_interest": 42,
            "data_available": True,
        }
