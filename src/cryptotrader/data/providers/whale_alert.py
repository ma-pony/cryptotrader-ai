"""Whale Alert API provider — free tier, 10 req/min."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.whale-alert.io/v1"


async def fetch_whale_transfers(api_key: str | None = None, min_usd: int = 500000) -> list[dict]:
    """Return recent large transfers."""
    if not api_key:
        logger.warning("Whale Alert API key not set, using default")
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{BASE}/transactions",
                params={"api_key": api_key, "min_value": min_usd, "currency": "btc"},
            )
            r.raise_for_status()
            txs = r.json().get("transactions", [])
            return [
                {
                    "hash": t.get("hash", ""),
                    "from": t.get("from", {}).get("owner", "unknown"),
                    "to": t.get("to", {}).get("owner", "unknown"),
                    "amount_usd": t.get("amount_usd", 0),
                }
                for t in txs[:20]
            ]
    except Exception:
        logger.warning("Whale Alert fetch failed", exc_info=True)
    return []
