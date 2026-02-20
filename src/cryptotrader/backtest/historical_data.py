"""Fetch and cache historical macro/news data for backtesting."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx

from cryptotrader.backtest.cache import CACHE_DB

UTC = timezone.utc


FRED_API_KEY = "f2a21b61924b9ede4c094dd27acecf39"


def _ensure_tables():
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fear_greed "
        "(date TEXT PRIMARY KEY, value INTEGER, classification TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS funding_rate "
        "(date TEXT PRIMARY KEY, rate REAL, count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS btc_dominance "
        "(date TEXT PRIMARY KEY, dominance REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fred_series "
        "(series_id TEXT, date TEXT, value REAL, PRIMARY KEY (series_id, date))"
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


async def fetch_funding_rate(symbol: str, start_date: str, end_date: str) -> dict[str, float]:
    """Fetch daily avg funding rate from Binance. Returns {date_str: avg_rate}."""
    _ensure_tables()
    conn = sqlite3.connect(str(CACHE_DB))
    cached = dict(conn.execute(
        "SELECT date, rate FROM funding_rate WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchall())
    conn.close()

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    expected = (end_dt - start_dt).days
    if len(cached) >= expected * 0.9:
        return cached

    # Paginate: Binance returns max 1000 records, 3 per day (8h intervals)
    pair = symbol.replace("/", "") + "T" if "/" in symbol else symbol + "USDT"
    all_records: list[dict] = []
    cursor_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000) + 86400000

    async with httpx.AsyncClient() as client:
        while cursor_ms < end_ms:
            r = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": pair, "startTime": cursor_ms, "limit": 1000},
                timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_records.extend(batch)
            cursor_ms = batch[-1]["fundingTime"] + 1

    # Aggregate to daily average
    daily: dict[str, list[float]] = {}
    for rec in all_records:
        dt = datetime.fromtimestamp(rec["fundingTime"] / 1000, UTC)
        d = dt.strftime("%Y-%m-%d")
        daily.setdefault(d, []).append(float(rec["fundingRate"]))

    conn = sqlite3.connect(str(CACHE_DB))
    for d, rates in daily.items():
        avg = sum(rates) / len(rates)
        conn.execute(
            "INSERT OR REPLACE INTO funding_rate (date, rate, count) VALUES (?,?,?)",
            (d, avg, len(rates)),
        )
        cached[d] = avg
    conn.commit()
    conn.close()
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


async def fetch_btc_dominance(start_date: str, end_date: str) -> dict[str, float]:
    """Fetch BTC dominance from CoinGecko. Estimates total mcap from BTC+ETH."""
    _ensure_tables()
    conn = sqlite3.connect(str(CACHE_DB))
    cached = dict(conn.execute(
        "SELECT date, dominance FROM btc_dominance WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchall())
    conn.close()

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    expected = (datetime.fromisoformat(end_date).replace(tzinfo=UTC) - start_dt).days
    if len(cached) >= expected * 0.9:
        return cached

    days = (datetime.now(UTC) - start_dt).days + 1

    async with httpx.AsyncClient() as client:
        btc_r = await client.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=30,
        )
        btc_r.raise_for_status()
        btc_caps = {
            datetime.fromtimestamp(p[0] / 1000, UTC).strftime("%Y-%m-%d"): p[1]
            for p in btc_r.json().get("market_caps", []) if p[1]
        }

        eth_r = await client.get(
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=30,
        )
        eth_r.raise_for_status()
        eth_caps = {
            datetime.fromtimestamp(p[0] / 1000, UTC).strftime("%Y-%m-%d"): p[1]
            for p in eth_r.json().get("market_caps", []) if p[1]
        }

    # BTC+ETH ≈ 66.4% of total market (stable ratio). Estimate total, then compute dominance.
    BTC_ETH_SHARE = 0.664
    conn = sqlite3.connect(str(CACHE_DB))
    for d, btc_cap in btc_caps.items():
        if start_date <= d <= end_date:
            eth_cap = eth_caps.get(d, 0)
            if eth_cap > 0:
                est_total = (btc_cap + eth_cap) / BTC_ETH_SHARE
                dom = round(btc_cap / est_total * 100, 2)
            else:
                dom = 56.0  # fallback
            conn.execute(
                "INSERT OR REPLACE INTO btc_dominance (date, dominance) VALUES (?,?)",
                (d, dom),
            )
            cached[d] = dom
    conn.commit()
    conn.close()
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


async def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> dict[str, float]:
    """Fetch a FRED time series. Returns {date_str: value}. Fills weekends/holidays forward."""
    _ensure_tables()
    conn = sqlite3.connect(str(CACHE_DB))
    cached = dict(conn.execute(
        "SELECT date, value FROM fred_series WHERE series_id=? AND date>=? AND date<=?",
        (series_id, start_date, end_date),
    ).fetchall())
    conn.close()

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    expected = (datetime.fromisoformat(end_date).replace(tzinfo=UTC) - start_dt).days
    if len(cached) >= expected * 0.6:  # FRED has no weekends, so ~60% coverage is full
        return cached

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "observation_start": start_date,
                    "observation_end": end_date},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])

    conn = sqlite3.connect(str(CACHE_DB))
    for o in obs:
        if o["value"] != ".":  # FRED uses "." for missing
            val = float(o["value"])
            conn.execute(
                "INSERT OR REPLACE INTO fred_series (series_id, date, value) VALUES (?,?,?)",
                (series_id, o["date"], val),
            )
            cached[o["date"]] = val
    conn.commit()
    conn.close()

    # Forward-fill weekends/holidays
    if cached:
        all_dates = sorted(cached.keys())
        last_val = cached[all_dates[0]]
        from datetime import timedelta
        cur = start_dt
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
        filled = {}
        while cur <= end_dt:
            d = cur.strftime("%Y-%m-%d")
            if d in cached:
                last_val = cached[d]
            filled[d] = last_val
            cur += timedelta(days=1)
        return filled
    return cached


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
