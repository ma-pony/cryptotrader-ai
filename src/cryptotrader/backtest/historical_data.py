"""Fetch and cache historical macro/news data for backtesting."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx

from cryptotrader.backtest.cache import CACHE_DB

UTC = timezone.utc


def _ensure_tables():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fear_greed "
        "(date TEXT PRIMARY KEY, value INTEGER, classification TEXT)"
    )
    conn.commit()
    conn.close()


async def fetch_fear_greed(start_date: str, end_date: str) -> dict[str, int]:
    """Fetch Fear & Greed Index history. Returns {date_str: value}."""
    _ensure_tables()
    conn = sqlite3.connect(str(CACHE_DB))
    cached = dict(conn.execute(
        "SELECT date, value FROM fear_greed WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchall())
    conn.close()

    if len(cached) > 180:  # good enough coverage
        return cached

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.alternative.me/fng/",
            params={"limit": "400", "format": "json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", [])

    conn = sqlite3.connect(str(CACHE_DB))
    for d in data:
        ts = int(d["timestamp"])
        dt = datetime.fromtimestamp(ts, UTC)
        date_str = dt.strftime("%Y-%m-%d")
        val = int(d["value"])
        conn.execute(
            "INSERT OR REPLACE INTO fear_greed (date, value, classification) VALUES (?,?,?)",
            (date_str, val, d.get("value_classification", "")),
        )
        cached[date_str] = val
    conn.commit()
    conn.close()
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


def derive_news_sentiment(candles: list[list], idx: int) -> tuple[float, list[str]]:
    """Derive proxy news sentiment from recent price action.
    Returns (sentiment_score -1..1, key_events list)."""
    if idx < 7:
        return 0.0, []

    # 7-day return as sentiment proxy
    c_now = candles[idx][4]
    c_7d = candles[max(0, idx - 7)][4]
    ret_7d = (c_now - c_7d) / c_7d

    # Volatility spike detection
    recent_returns = []
    for i in range(max(1, idx - 14), idx + 1):
        recent_returns.append(abs(candles[i][4] - candles[i-1][4]) / candles[i-1][4])
    avg_vol = sum(recent_returns) / len(recent_returns) if recent_returns else 0.01
    today_vol = recent_returns[-1] if recent_returns else 0

    events = []
    if abs(ret_7d) > 0.10:
        events.append(f"BTC {'surged' if ret_7d > 0 else 'crashed'} {ret_7d:.1%} in 7 days")
    if today_vol > avg_vol * 2:
        events.append(f"Volatility spike: {today_vol:.2%} vs avg {avg_vol:.2%}")

    # Clamp sentiment to [-1, 1]
    sentiment = max(-1.0, min(1.0, ret_7d * 5))
    return sentiment, events
