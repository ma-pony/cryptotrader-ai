"""Data tools for AI agents — wrap existing providers as LangChain tools.

Each tool is a thin async wrapper around existing data providers,
giving agents the ability to actively query data during analysis.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Chain / Derivatives Tools ──


@tool
async def get_derivatives_data(pair: str) -> str:
    """Fetch real-time derivatives data from Binance (OI, long/short ratio, taker ratio).

    Use when you need current futures positioning data.
    Args:
        pair: Trading pair like "BTC/USDT"
    """
    from cryptotrader.data.providers.binance import fetch_derivatives_binance

    symbol = pair.split("/")[0]
    data = await fetch_derivatives_binance(symbol)
    return json.dumps(data, default=str)


@tool
async def get_funding_rate_history(pair: str, periods: int = 8) -> str:
    """Fetch recent funding rate snapshots from Binance futures.

    Use to detect persistent funding bias (sustained positive = crowded long).
    Args:
        pair: Trading pair like "BTC/USDT"
        periods: Number of 8h periods to fetch (default 8 = last 2.67 days)
    """
    import httpx

    symbol = pair.split("/")[0] + "USDT"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": min(periods, 100)},
            )
            r.raise_for_status()
            rows = r.json()
            return json.dumps(
                [{"time": r["fundingTime"], "rate": float(r["fundingRate"])} for r in rows],
                default=str,
            )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def get_liquidation_data(pair: str) -> str:
    """Fetch 24h liquidation data (long vs short liquidation amounts).

    Use to assess leverage flush risk and cascade potential.
    Args:
        pair: Trading pair like "BTC/USDT"
    """
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.coinglass import fetch_derivatives

    cfg = load_config().providers
    symbol = pair.split("/")[0]
    data = await fetch_derivatives(cfg.coinglass_api_key, symbol)
    return json.dumps(data, default=str)


@tool
async def get_whale_transfers(pair: str) -> str:
    """Fetch recent large whale transfers for a given asset.

    Use to detect smart money accumulation or distribution.
    Args:
        pair: Trading pair like "BTC/USDT"
    """
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.whale_alert import fetch_whale_transfers as _fetch

    cfg = load_config().providers
    transfers = await _fetch(cfg.whale_alert_api_key)
    # Filter by asset if possible
    symbol = pair.split("/")[0].upper()
    relevant = [t for t in transfers if t.get("symbol", "").upper() == symbol] or transfers
    return json.dumps(relevant[:10], default=str)


@tool
async def get_exchange_netflow(pair: str) -> str:
    """Fetch exchange net inflow/outflow data.

    Positive = net inflow (sell pressure), Negative = net outflow (accumulation).
    Args:
        pair: Trading pair like "BTC/USDT"
    """
    from cryptotrader.config import load_config
    from cryptotrader.data.providers.cryptoquant import fetch_exchange_netflow as _fetch

    cfg = load_config().providers
    netflow = await _fetch(cfg.cryptoquant_api_key)
    label = "inflow (sell pressure)" if netflow > 0 else "outflow (accumulation)"
    return json.dumps({"netflow": netflow, "interpretation": label})


@tool
async def get_defi_tvl() -> str:
    """Fetch total DeFi TVL and 7-day change from DefiLlama.

    Use to assess capital flows in the DeFi ecosystem.
    """
    from cryptotrader.data.providers.defillama import fetch_tvl

    data = await fetch_tvl()
    return json.dumps(data, default=str)


# ── News / Sentiment Tools ──


@tool
async def search_crypto_news(query: str, limit: int = 10) -> str:
    """Search recent crypto news headlines from CoinDesk, CoinTelegraph, Decrypt.

    Returns headlines with keyword sentiment score.
    Args:
        query: Search keyword (e.g. "BTC", "ethereum", "SEC")
        limit: Max headlines to return (default 10)
    """
    from cryptotrader.data.news import NewsCollector

    collector = NewsCollector()
    headlines, score, key_events = await collector._collect_rss(query.lower())
    return json.dumps(
        {
            "headlines": headlines[:limit],
            "sentiment_score": round(score, 3),
            "key_events": key_events,
            "query": query,
        }
    )


@tool
async def get_social_buzz(pair: str) -> str:
    """Fetch social buzz metrics from CoinGecko (Twitter followers, Reddit, sentiment).

    Use to gauge community engagement and sentiment momentum.
    Args:
        pair: Trading pair like "BTC/USDT"
    """
    from cryptotrader.data.news import _fetch_social_buzz

    symbol = pair.split("/")[0]
    buzz = await _fetch_social_buzz(symbol)
    return json.dumps({"social_buzz": buzz, "symbol": symbol})


@tool
async def get_fear_greed_index() -> str:
    """Fetch the current Crypto Fear & Greed Index (0-100).

    Extreme Fear (<25) is a contrarian buy signal.
    Extreme Greed (>75) is a contrarian sell signal.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.alternative.me/fng/?limit=7")
            r.raise_for_status()
            entries = r.json().get("data", [])
            return json.dumps(
                [
                    {"value": int(e["value"]), "classification": e["value_classification"], "date": e["timestamp"]}
                    for e in entries
                ]
            )
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool sets per agent type ──

CHAIN_TOOLS = [
    get_derivatives_data,
    get_funding_rate_history,
    get_liquidation_data,
    get_whale_transfers,
    get_exchange_netflow,
    get_defi_tvl,
]

NEWS_TOOLS = [
    search_crypto_news,
    get_social_buzz,
    get_fear_greed_index,
]
