"""Historical OHLCV cache using SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path


CACHE_DB = Path.home() / ".cryptotrader" / "ohlcv_cache.db"


def _ensure_db() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ohlcv "
        "(pair TEXT, timeframe TEXT, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, "
        "PRIMARY KEY (pair, timeframe, ts))"
    )
    return conn


def get_cached(pair: str, timeframe: str, since: int, until: int) -> list[list]:
    conn = _ensure_db()
    rows = conn.execute(
        "SELECT ts, o, h, l, c, v FROM ohlcv WHERE pair=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
        (pair, timeframe, since, until),
    ).fetchall()
    conn.close()
    return [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]


def store_ohlcv(pair: str, timeframe: str, candles: list[list]) -> None:
    conn = _ensure_db()
    conn.executemany(
        "INSERT OR REPLACE INTO ohlcv (pair, timeframe, ts, o, h, l, c, v) VALUES (?,?,?,?,?,?,?,?)",
        [(pair, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles],
    )
    conn.commit()
    conn.close()


async def fetch_historical(pair: str, timeframe: str, since_ms: int, until_ms: int) -> list[list]:
    """Fetch OHLCV with cache. Uses ccxt pagination for large ranges."""
    cached = get_cached(pair, timeframe, since_ms, until_ms)
    # Only use cache if it covers the full range (first candle near start)
    if cached and cached[0][0] <= since_ms + 86_400_000:
        return cached

    import ccxt.async_support as ccxt_async

    exchange = ccxt_async.binance({"enableRateLimit": True})
    all_candles = []
    cursor = since_ms

    try:
        while cursor < until_ms:
            candles = await exchange.fetch_ohlcv(pair, timeframe, since=cursor, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            cursor = candles[-1][0] + 1
    finally:
        await exchange.close()

    if all_candles:
        store_ohlcv(pair, timeframe, all_candles)
    return [c for c in all_candles if c[0] <= until_ms]
