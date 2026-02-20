"""CoinGlass API provider — free tier, 1000 req/month."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

BASE = "https://open-api.coinglass.com/public/v2"


async def fetch_derivatives(api_key: str | None = None, symbol: str = "BTC") -> dict:
    """Return open_interest and liquidations_24h."""
    if not api_key:
        logger.warning("CoinGlass API key not set, using defaults")
        return {"open_interest": 0.0, "liquidations_24h": {}}
    headers = {"coinglassSecret": api_key}
    result: dict = {"open_interest": 0.0, "liquidations_24h": {}}
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as c:
            r = await c.get(f"{BASE}/open_interest", params={"symbol": symbol})
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                result["open_interest"] = float(data[0].get("openInterest", 0))

            r2 = await c.get(f"{BASE}/liquidation_history", params={"symbol": symbol})
            r2.raise_for_status()
            liq = r2.json().get("data", [])
            if liq:
                result["liquidations_24h"] = {
                    "long": float(liq[0].get("longLiquidationUsd", 0)),
                    "short": float(liq[0].get("shortLiquidationUsd", 0)),
                }
    except Exception:
        logger.warning("CoinGlass fetch failed", exc_info=True)
    return result
