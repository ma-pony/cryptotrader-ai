"""Tests for data collectors, debate researchers, and snapshot aggregator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── MarketCollector ──


def test_closed_ohlcv_excludes_forming_candle():
    from cryptotrader.data.market import _closed_ohlcv

    hour = 3_600_000
    now_ms = 10 * hour + 2 * 60_000
    rows = [
        [8 * hour, 1, 1, 1, 1, 1],
        [9 * hour, 2, 2, 2, 2, 2],
        [10 * hour, 3, 3, 3, 3, 3],
    ]

    assert _closed_ohlcv(rows, "1h", now_ms) == rows[:2]


def test_fetch_market_types_match_ccxt_exchange_names():
    from cryptotrader.ccxt_options import fetch_market_types

    assert fetch_market_types("binance") == ["spot", "linear"]
    assert fetch_market_types("okx") == ["spot", "swap"]
    assert fetch_market_types("coinbase") == ["spot"]


@pytest.mark.asyncio
async def test_market_collector():
    """MarketCollector assembles MarketData from ccxt."""
    from cryptotrader.data.market import MarketCollector

    mock_exchange = MagicMock()
    mock_exchange.load_markets = AsyncMock()
    mock_exchange.fetch_ohlcv = AsyncMock(
        return_value=[
            [1700000000000, 50000, 51000, 49000, 50500, 100],
            [1700003600000, 50500, 52000, 50000, 51000, 120],
            [1700007200000, 51000, 51500, 50500, 51200, 110],
        ]
    )
    mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 51200, "baseVolume": 5000})
    mock_exchange.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0003})
    mock_exchange.fetch_order_book = AsyncMock(
        return_value={
            "bids": [[51000, 10], [50900, 5]],
            "asks": [[51100, 8], [51200, 6]],
        }
    )
    mock_exchange.close = AsyncMock()

    with (
        patch("cryptotrader.data.market.ccxt") as mock_ccxt,
        patch("cryptotrader.data.market.get_cached_or_none", return_value=None),
        patch("cryptotrader.data.market.cache_result"),
    ):
        mock_ccxt.binance.return_value = mock_exchange
        collector = MarketCollector()
        result = await collector.collect("BTC/USDT", "binance", "1h", 100)

    assert result.pair == "BTC/USDT"
    assert result.ticker["last"] == 51200
    assert result.funding_rate == 0.0003
    assert result.volatility >= 0
    assert len(result.ohlcv) == 3
    mock_ccxt.binance.assert_called_once_with({"options": {"fetchMarkets": ["spot", "linear"]}})
    mock_exchange.close.assert_called_once()


@pytest.mark.asyncio
async def test_market_collector_paginates_large_ohlcv_request():
    """Kronos-sized requests must bypass exchange single-page limits."""
    from cryptotrader.data.market import MarketCollector

    start_ms = 1_700_000_000_000
    rows = [[start_ms + i * 14_400_000, 1, 2, 0.5, 1.5, 10] for i in range(513)]
    mock_exchange = MagicMock()
    mock_exchange.load_markets = AsyncMock()
    mock_exchange.fetch_ohlcv = AsyncMock(return_value=rows)
    mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 1.5})
    mock_exchange.fetch_funding_rate = AsyncMock(return_value={"fundingRate": 0.0})
    mock_exchange.fetch_order_book = AsyncMock(return_value={"bids": [[1, 1]], "asks": [[2, 1]]})
    mock_exchange.close = AsyncMock()

    with (
        patch("cryptotrader.data.market.ccxt") as mock_ccxt,
        patch("cryptotrader.data.market.get_cached_or_none", return_value=None),
        patch("cryptotrader.data.market.cache_result"),
    ):
        mock_ccxt.okx.return_value = mock_exchange
        result = await MarketCollector().collect("BTC/USDT", "okx", "4h", 512)

    mock_exchange.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", "4h", limit=513, params={"paginate": True})
    assert len(result.ohlcv) == 512


@pytest.mark.asyncio
async def test_market_collector_funding_rate_fallback():
    """MarketCollector handles missing funding rate."""
    from cryptotrader.data.market import MarketCollector

    mock_exchange = MagicMock()
    mock_exchange.load_markets = AsyncMock()
    mock_exchange.fetch_ohlcv = AsyncMock(return_value=[[1700000000000, 50000, 51000, 49000, 50500, 100]])
    mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50500})
    mock_exchange.fetch_funding_rate = AsyncMock(side_effect=Exception("Not supported"))
    mock_exchange.fetch_order_book = AsyncMock(return_value={"bids": [[50000, 1]], "asks": [[51000, 1]]})
    mock_exchange.close = AsyncMock()

    with (
        patch("cryptotrader.data.market.ccxt") as mock_ccxt,
        patch("cryptotrader.data.market.get_cached_or_none", return_value=None),
        patch("cryptotrader.data.market.cache_result"),
    ):
        mock_ccxt.binance.return_value = mock_exchange
        collector = MarketCollector()
        result = await collector.collect("ETH/USDT", "binance")

    assert result.funding_rate == 0.0


# ── OnchainCollector ──


@pytest.mark.asyncio
async def test_onchain_collector_all_providers():
    """OnchainCollector merges data from multiple providers."""
    from cryptotrader.data.onchain import OnchainCollector

    with (
        patch(
            "cryptotrader.data.providers.binance.fetch_derivatives_binance",
            new_callable=AsyncMock,
            return_value={
                "open_interest": 1000000,
                "long_short_ratio": 1.2,
                "top_trader_ratio": 1.1,
                "taker_buy_sell_ratio": 0.95,
            },
        ),
        patch(
            "cryptotrader.data.providers.defillama.fetch_tvl",
            new_callable=AsyncMock,
            return_value={"defi_tvl": 50e9, "defi_tvl_change_7d": 2.5},
        ),
        patch(
            "cryptotrader.data.providers.coinglass.fetch_derivatives",
            new_callable=AsyncMock,
            return_value={"open_interest": 1200000, "liquidations_24h": {"total": 5000000}},
        ),
        patch(
            "cryptotrader.data.providers.cryptoquant.fetch_exchange_netflow",
            new_callable=AsyncMock,
            return_value=-500.0,
        ),
        patch(
            "cryptotrader.data.providers.whale_alert.fetch_whale_transfers",
            new_callable=AsyncMock,
            return_value=[{"hash": "abc", "amount": 1000}],
        ),
    ):
        collector = OnchainCollector(providers_config=None)
        result = await collector.collect("BTC/USDT", 0.01)

    assert result.open_interest == 1200000  # Prefers CoinGlass
    assert result.exchange_netflow == -500.0
    assert result.defi_tvl == 50e9
    assert len(result.whale_transfers) == 1


@pytest.mark.asyncio
async def test_onchain_collector_provider_failure():
    """OnchainCollector handles provider failures gracefully."""
    from cryptotrader.data.onchain import OnchainCollector

    with (
        patch(
            "cryptotrader.data.providers.binance.fetch_derivatives_binance",
            new_callable=AsyncMock,
            return_value={"open_interest": 500000},
        ),
        patch(
            "cryptotrader.data.providers.defillama.fetch_tvl",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ),
        patch(
            "cryptotrader.data.providers.coinglass.fetch_derivatives",
            new_callable=AsyncMock,
            side_effect=Exception("Rate limited"),
        ),
        patch(
            "cryptotrader.data.providers.cryptoquant.fetch_exchange_netflow",
            new_callable=AsyncMock,
            side_effect=Exception("Timeout"),
        ),
        patch(
            "cryptotrader.data.providers.whale_alert.fetch_whale_transfers",
            new_callable=AsyncMock,
            side_effect=Exception("Auth failed"),
        ),
    ):
        collector = OnchainCollector(providers_config=None)
        result = await collector.collect("ETH/USDT")

    # Should not raise, returns defaults
    assert result.open_interest == 500000  # Falls back to Binance
    assert result.exchange_netflow == 0.0
    assert result.defi_tvl == 0.0
    assert result.whale_transfers == []
    assert result.data_quality["binance"] is True
    assert result.data_quality["defillama"] is False


# ── MacroCollector ──


@pytest.mark.asyncio
async def test_macro_collector():
    """MacroCollector aggregates macro data sources."""
    from cryptotrader.data.macro import MacroCollector

    with (
        patch("cryptotrader.data.macro._fetch_fred", new_callable=AsyncMock, return_value=5.25),
        patch("cryptotrader.data.macro._fetch_fear_greed", new_callable=AsyncMock, return_value=72),
        patch("cryptotrader.data.macro._fetch_btc_dominance", new_callable=AsyncMock, return_value=54.3),
    ):
        collector = MacroCollector(providers_config=None)
        result = await collector.collect()

    assert result.fear_greed_index == 72
    assert result.btc_dominance == 54.3
    # With no config/key, FRED tasks use noop (0.0)
    assert result.fed_rate == 0.0


@pytest.mark.asyncio
async def test_fetch_fear_greed_fallback():
    """_fetch_fear_greed returns 50 on API failure."""
    from cryptotrader.data.macro import _fetch_fear_greed

    with (
        patch("cryptotrader.data.macro.get_cached_or_none", return_value=None),
        patch("cryptotrader.data.macro.cache_result"),
        patch("cryptotrader.data.macro.httpx.AsyncClient") as mock_client,
    ):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
        mock_client.return_value = mock_instance
        result = await _fetch_fear_greed()

    assert result == (50, [])
