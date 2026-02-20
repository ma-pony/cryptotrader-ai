"""Macro data collector — FRED, CoinGecko, Alternative.me."""

from __future__ import annotations

import logging
import os

import httpx

from cryptotrader.models import MacroData

logger = logging.getLogger(__name__)


async def _fetch_fred(series: str, api_key: str) -> float:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series, "api_key": api_key,
                        "sort_order": "desc", "limit": 1, "file_type": "json"},
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                return float(obs[0]["value"])
    except Exception:
        logger.warning("FRED fetch failed for %s", series, exc_info=True)
    return 0.0


async def _fetch_fear_greed() -> int:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.alternative.me/fng/?limit=1")
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                return int(data[0]["value"])
    except Exception:
        logger.warning("Fear & Greed fetch failed", exc_info=True)
    return 50


async def _fetch_btc_dominance() -> float:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception:
        logger.warning("CoinGecko dominance fetch failed", exc_info=True)
    return 0.0


class MacroCollector:

    async def collect(self) -> MacroData:
        fred_key = os.environ.get("FRED_API_KEY", "")
        fed_rate = await _fetch_fred("DFF", fred_key) if fred_key else 0.0
        dxy = await _fetch_fred("DTWEXBGS", fred_key) if fred_key else 0.0
        fear_greed = await _fetch_fear_greed()
        btc_dom = await _fetch_btc_dominance()

        return MacroData(
            fed_rate=fed_rate,
            dxy=dxy,
            btc_dominance=btc_dom,
            fear_greed_index=fear_greed,
        )
