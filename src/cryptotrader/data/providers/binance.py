"""Binance Futures free API — OI, long/short ratio, taker volume."""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)
BASE = "https://fapi.binance.com"


async def fetch_derivatives_binance(symbol: str = "BTC") -> dict:
    """Fetch OI + long/short + taker volume from Binance (free, no key)."""
    pair = f"{symbol}USDT"
    result = {
        "open_interest": 0.0,
        "open_interest_value": 0.0,
        "long_short_ratio": 1.0,
        "top_trader_ratio": 1.0,
        "taker_buy_sell_ratio": 1.0,
        "liquidations_24h": {},
    }
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
    async def _get(url, params=None):
        try:
            r = await c.get(url, params=params)
            r.raise_for_status()
            d = r.json()
            return d[0] if isinstance(d, list) else d
        except Exception:
            logger.debug("Binance API request failed", exc_info=True)
            return None

    return await asyncio.gather(
        _get(f"{BASE}/fapi/v1/openInterest", {"symbol": pair}),
        _get(f"{BASE}/futures/data/globalLongShortAccountRatio", {"symbol": pair, "period": "1d", "limit": 1}),
        _get(f"{BASE}/futures/data/topLongShortPositionRatio", {"symbol": pair, "period": "1d", "limit": 1}),
        _get(f"{BASE}/futures/data/takerlongshortRatio", {"symbol": pair, "period": "1d", "limit": 1}),
    )


async def fetch_funding_rate_binance(symbol: str = "BTC") -> dict:
    """Fetch latest funding rate via ccxt ``fetch_funding_rate``.

    ``symbol`` is the bare base ("BTC", "ETH"); the perp symbol is built as
    ``BASE/USDT:USDT`` (linear). Returns ``{funding_rate, next_funding_time}``
    so existing callers don't need to change their dict-key reads.
    """
    perp = _perp_symbol_for(symbol)
    result: dict = {"funding_rate": 0.0, "next_funding_time": 0}
    try:
        client = await _open_market_client()
        try:
            data = await client.fetch_funding_rate(perp)
        finally:
            await client.close()
        if data:
            rate = data.get("fundingRate")
            if rate is not None:
                result["funding_rate"] = float(rate)
            next_ts = data.get("fundingTimestamp") or data.get("nextFundingTimestamp")
            if next_ts is not None:
                result["next_funding_time"] = int(next_ts)
    except Exception:
        logger.warning("Binance funding rate fetch failed", exc_info=True)
    return result


async def fetch_funding_history_ccxt(
    pair: str,
    *,
    since_ms: int | None = None,
    limit: int | None = None,
    page_size: int = 1000,
    pause_seconds: float = 0.1,
) -> list[dict]:
    """Paginated funding-rate history via ccxt ``fetch_funding_rate_history``.

    Designed to be a drop-in for the ``fapi/v1/fundingRate`` REST loop that
    appeared in 4 places before this refactor (agents/data_tools.py,
    backtest/historical_data.py, data/sync.py twice). Returns dicts shaped
    like the legacy Binance response (``fundingTime`` ms int, ``fundingRate``
    float) so existing aggregation code still works without rewrites.

    ``pair`` accepts any of ``"BTC"``, ``"BTCUSDT"``, ``"BTC/USDT"`` or the
    canonical perp ``"BTC/USDT:USDT"`` — ccxt needs the canonical form.
    """
    perp = _perp_symbol_for(pair)
    out: list[dict] = []
    cursor = since_ms
    client = await _open_market_client()
    try:
        while True:
            raw_batch = await client.fetch_funding_rate_history(perp, since=cursor, limit=page_size)
            batch: list[dict] = [e for e in (raw_batch or []) if isinstance(e, dict)]
            if not batch:
                break
            for entry in batch:
                ts = entry.get("timestamp")
                rate = entry.get("fundingRate")
                if ts is None or rate is None:
                    continue
                out.append({"fundingTime": int(ts), "fundingRate": float(rate)})
            if limit is not None and len(out) >= limit:
                return out[:limit]
            if len(batch) < page_size:
                break
            last_ts = batch[-1].get("timestamp")
            if last_ts is None:
                break
            cursor = int(last_ts) + 1
            if pause_seconds > 0:
                await asyncio.sleep(pause_seconds)
    finally:
        await client.close()
    return out


async def _open_market_client():
    """Lazy import + construct a public-only ccxt async Binance client.

    No credentials — these endpoints (klines, funding, premium index) are
    public. Caller is responsible for ``await client.close()`` to release
    the aiohttp connector.
    """
    import ccxt.async_support as ccxt_async

    return ccxt_async.binance({"enableRateLimit": True})


def _perp_symbol_for(pair: str) -> str:
    """Coerce caller input to a ccxt unified linear-perp symbol.

    Accepts:
      - bare base: ``"BTC"`` → ``"BTC/USDT:USDT"``
      - Binance native: ``"BTCUSDT"`` → ``"BTC/USDT:USDT"``
      - spot canonical: ``"BTC/USDT"`` → ``"BTC/USDT:USDT"``
      - already-perp: ``"BTC/USDT:USDT"`` → unchanged
    """
    p = pair.strip()
    if ":" in p:
        return p
    if "/" in p:
        return f"{p}:{p.split('/')[1]}"
    upper = p.upper()
    for quote in ("USDT", "USDC", "BUSD"):
        if upper.endswith(quote) and len(upper) > len(quote):
            base = upper[: -len(quote)]
            return f"{base}/{quote}:{quote}"
    # Bare base symbol — assume USDT linear perp.
    return f"{upper}/USDT:USDT"
