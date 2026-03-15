"""Whale Alert API provider — free tier, 10 req/min."""

from __future__ import annotations

import logging

import httpx
from pydantic import ValidationError

from cryptotrader.models import OnchainMetricResponse

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
            result: list[dict] = []
            for t in txs[:20]:
                tx_hash = t.get("hash", "")
                if not tx_hash or not tx_hash.strip():
                    logger.warning("Whale Alert transaction schema validation failed, skipping row: hash is empty")
                    continue
                raw_amount = t.get("amount_usd", 0)
                try:
                    validated = OnchainMetricResponse(
                        metric_name="amount_usd", value=float(raw_amount), source="whale_alert"
                    )
                    amount_usd = validated.value
                except ValidationError as exc:
                    logger.warning("Whale Alert amount_usd schema validation failed, skipping row: %s", exc)
                    continue
                result.append(
                    {
                        "hash": tx_hash.strip(),
                        "from": t.get("from", {}).get("owner", "unknown"),
                        "to": t.get("to", {}).get("owner", "unknown"),
                        "amount_usd": amount_usd,
                    }
                )
            return result
    except Exception:
        logger.warning("Whale Alert fetch failed", exc_info=True)
    return []
