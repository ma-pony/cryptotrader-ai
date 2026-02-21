"""Binance Futures free API — OI, long/short ratio, taker volume."""

from __future__ import annotations

import logging
import httpx

logger = logging.getLogger(__name__)
BASE = "https://fapi.binance.com"


async def fetch_derivatives_binance(symbol: str = "BTC") -> dict:
    """Fetch OI + long/short + taker volume from Binance (free, no key)."""
    pair = f"{symbol}USDT"
    result = {"open_interest": 0.0, "open_interest_value": 0.0,
              "long_short_ratio": 1.0, "top_trader_ratio": 1.0,
              "taker_buy_sell_ratio": 1.0, "liquidations_24h": {}}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            oi, ls, top, taker = await _gather(c, pair)
            if oi:
                result["open_interest"] = float(oi.get("openInterest", 0))
            if ls:
                result["long_short_ratio"] = float(ls.get("longShortRatio", 1))
            if top:
                result["top_trader_ratio"] = float(top.get("longShortRatio", 1))
            if taker:
                result["taker_buy_sell_ratio"] = float(taker.get("buySellRatio", 1))
    except Exception:
        logger.warning("Binance derivatives fetch failed", exc_info=True)
    return result


async def _gather(c: httpx.AsyncClient, pair: str):
    import asyncio
    async def _get(url, params=None):
        try:
            r = await c.get(url, params=params)
            r.raise_for_status()
            d = r.json()
            return d[0] if isinstance(d, list) else d
        except Exception:
            return None

    return await asyncio.gather(
        _get(f"{BASE}/fapi/v1/openInterest", {"symbol": pair}),
        _get(f"{BASE}/futures/data/globalLongShortAccountRatio", {"symbol": pair, "period": "1d", "limit": 1}),
        _get(f"{BASE}/futures/data/topLongShortPositionRatio", {"symbol": pair, "period": "1d", "limit": 1}),
        _get(f"{BASE}/futures/data/takerlongshortRatio", {"symbol": pair, "period": "1d", "limit": 1}),
    )
