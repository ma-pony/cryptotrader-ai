"""Tests for data collectors, debate researchers, and snapshot aggregator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── MarketCollector ──


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

    with patch("cryptotrader.data.market.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = mock_exchange
        collector = MarketCollector()
        result = await collector.collect("BTC/USDT", "binance", "1h", 100)

    assert result.pair == "BTC/USDT"
    assert result.ticker["last"] == 51200
    assert result.funding_rate == 0.0003
    assert result.volatility >= 0
    assert len(result.ohlcv) == 3
    mock_exchange.close.assert_called_once()


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

    with patch("cryptotrader.data.market.ccxt") as mock_ccxt:
        mock_ccxt.binance.return_value = mock_exchange
        collector = MarketCollector()
        result = await collector.collect("ETH/USDT")

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

    with patch("cryptotrader.data.macro.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
        MockClient.return_value = mock_instance
        result = await _fetch_fear_greed()

    assert result == 50


# ── Debate researchers ──


@pytest.mark.asyncio
async def test_run_debate():
    """run_debate produces bull/bear history."""
    from cryptotrader.debate.researchers import run_debate

    analyses = {
        "tech_agent": {"direction": "bullish", "confidence": 0.8, "reasoning": "RSI oversold at 28"},
        "macro_agent": {"direction": "bearish", "confidence": 0.7, "reasoning": "Fear index at 80"},
    }

    from langchain_core.messages import AIMessage

    mock_ai_msg = AIMessage(content="Strong argument here.")

    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, return_value=mock_ai_msg):
        result = await run_debate(analyses, rounds=2)

    assert len(result["bull_history"]) == 2
    assert len(result["bear_history"]) == 2
    assert result["rounds"] == 2
    assert "full_debate" in result
    assert "Round 1" in result["full_debate"]


@pytest.mark.asyncio
async def test_run_debate_llm_failure():
    """run_debate handles LLM failures gracefully."""
    from cryptotrader.debate.researchers import run_debate

    analyses = {"tech_agent": {"direction": "bullish", "confidence": 0.5, "reasoning": "test"}}

    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, side_effect=Exception("API error")):
        result = await run_debate(analyses, rounds=1)

    assert "Unable to generate" in result["bull_history"][0]
    assert "Unable to generate" in result["bear_history"][0]


@pytest.mark.asyncio
async def test_judge_debate_long():
    """judge_debate parses long verdict."""
    from cryptotrader.debate.researchers import judge_debate

    debate = {
        "full_debate": "BULL: RSI at 28 oversold\nBEAR: Funding at 0.05% crowded",
        "bull_history": ["argument"],
        "bear_history": ["argument"],
    }

    from langchain_core.messages import AIMessage

    mock_ai_msg = AIMessage(content='{"action": "long", "confidence": 0.75, "reasoning": "Bull won with RSI data"}')

    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, return_value=mock_ai_msg):
        result = await judge_debate(debate, "BTC/USDT")

    assert result["action"] == "long"
    assert result["confidence"] == 0.75


@pytest.mark.asyncio
async def test_judge_debate_normalizes_action():
    """judge_debate normalizes 'buy'→'long', 'sell'→'short'."""
    from cryptotrader.debate.researchers import judge_debate

    debate = {"full_debate": "test", "bull_history": [], "bear_history": []}

    for raw_action, expected in [("buy", "long"), ("bullish", "long"), ("sell", "short"), ("bearish", "short")]:
        from langchain_core.messages import AIMessage

        mock_ai_msg = AIMessage(content=f'{{"action": "{raw_action}", "confidence": 0.6, "reasoning": "test"}}')
        with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, return_value=mock_ai_msg):
            result = await judge_debate(debate, "BTC/USDT")
        assert result["action"] == expected, f"Expected {expected} for {raw_action}"


@pytest.mark.asyncio
async def test_judge_debate_fallback():
    """judge_debate returns hold on LLM failure."""
    from cryptotrader.debate.researchers import judge_debate

    debate = {"full_debate": "test", "bull_history": [], "bear_history": []}

    with patch("langchain_openai.ChatOpenAI.ainvoke", new_callable=AsyncMock, side_effect=Exception("fail")):
        result = await judge_debate(debate, "ETH/USDT")

    assert result["action"] == "hold"
    assert result["confidence"] == 0.2


def test_format_reports():
    """_format_reports produces readable analyst summary."""
    from cryptotrader.debate.researchers import _format_reports

    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.8, "reasoning": "Strong RSI"},
        "chain": {"direction": "bearish", "confidence": 0.6, "reasoning": "High funding"},
    }
    result = _format_reports(analyses)
    assert "[tech]" in result
    assert "[chain]" in result
    assert "Strong RSI" in result
