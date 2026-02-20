"""CryptoQuant API provider — free tier, daily."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.cryptoquant.com/v1"


async def fetch_exchange_netflow(api_key: str | None = None) -> float:
    """Return BTC exchange netflow."""
    if not api_key:
        logger.warning("CryptoQuant API key not set, using default")
        return 0.0
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{BASE}/btc/exchange-flows/netflow",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()
            data = r.json().get("result", {}).get("data", [])
            if data:
                return float(data[-1].get("netflow", 0.0))
    except Exception:
        logger.warning("CryptoQuant fetch failed", exc_info=True)
    return 0.0
