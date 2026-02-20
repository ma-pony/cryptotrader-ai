"""DefiLlama API provider — free, no key required."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.llama.fi"
YIELDS = "https://yields.llama.fi"


async def fetch_tvl(chain: str = "Ethereum") -> dict:
    """Return defi_tvl and defi_tvl_change_7d for a chain."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{BASE}/v2/historicalChainTvl/{chain}")
            r.raise_for_status()
            data = r.json()
            if len(data) < 2:
                return {"defi_tvl": 0.0, "defi_tvl_change_7d": 0.0}
            current = data[-1].get("tvl", 0.0)
            week_ago = data[-8].get("tvl", current) if len(data) >= 8 else current
            change = (current - week_ago) / week_ago if week_ago else 0.0
            return {"defi_tvl": current, "defi_tvl_change_7d": change}
    except Exception:
        logger.warning("DefiLlama fetch failed", exc_info=True)
        return {"defi_tvl": 0.0, "defi_tvl_change_7d": 0.0}
