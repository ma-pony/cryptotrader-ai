"""Fetch and cache historical macro/news data for backtesting."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime

import httpx

from cryptotrader.backtest.cache import CACHE_DB

# BTC+ETH combined share of total crypto market cap (approximate stable ratio)
_BTC_ETH_MARKET_SHARE = 0.664
# Fallback BTC dominance percentage when ETH market cap data is unavailable
_BTC_DOMINANCE_FALLBACK = 56.0


def _ensure_tables():
    with sqlite3.connect(str(CACHE_DB)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fear_greed (date TEXT PRIMARY KEY, value INTEGER, classification TEXT)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS funding_rate (date TEXT PRIMARY KEY, rate REAL, count INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS btc_dominance (date TEXT PRIMARY KEY, dominance REAL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fred_series"
            " (series_id TEXT, date TEXT, value REAL, PRIMARY KEY (series_id, date))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS futures_volume (date TEXT PRIMARY KEY, volume REAL, quote_volume REAL)"
        )
        conn.commit()


async def fetch_fear_greed(start_date: str, end_date: str) -> dict[str, int]:
    """Fetch Fear & Greed Index history. Returns {date_str: value}."""
    _ensure_tables()
    with sqlite3.connect(str(CACHE_DB)) as conn:
        cached = dict(
            conn.execute(
                "SELECT date, value FROM fear_greed WHERE date >= ? AND date <= ?",
                (start_date, end_date),
            ).fetchall()
        )

    # Check coverage against actual date range, not hardcoded 180 days
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    expected_days = (end_dt - start_dt).days
    if len(cached) >= expected_days * 0.9:
        return cached

    # Calculate limit based on how far back start_date is from today
    days_to_today = (datetime.now(UTC) - start_dt).days
    limit = max(400, days_to_today + 30)

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.alternative.me/fng/",
            params={"limit": str(limit), "format": "json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", [])

    with sqlite3.connect(str(CACHE_DB)) as conn:
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
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


async def fetch_funding_rate(symbol: str, start_date: str, end_date: str) -> dict[str, float]:
    """Fetch daily avg funding rate from Binance. Returns {date_str: avg_rate}."""
    _ensure_tables()
    with sqlite3.connect(str(CACHE_DB)) as conn:
        cached = dict(
            conn.execute(
                "SELECT date, rate FROM funding_rate WHERE date >= ? AND date <= ?",
                (start_date, end_date),
            ).fetchall()
        )

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    expected = (end_dt - start_dt).days
    if len(cached) >= expected * 0.9:
        return cached

    # Paginate: Binance returns max 1000 records, 3 per day (8h intervals)
    pair = symbol.replace("/", "") if "/" in symbol else symbol + "USDT"
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

    with sqlite3.connect(str(CACHE_DB)) as conn:
        for d, rates in daily.items():
            avg = sum(rates) / len(rates)
            conn.execute(
                "INSERT OR REPLACE INTO funding_rate (date, rate, count) VALUES (?,?,?)",
                (d, avg, len(rates)),
            )
            cached[d] = avg
        conn.commit()
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


async def fetch_btc_dominance(start_date: str, end_date: str) -> dict[str, float]:
    """Fetch BTC dominance from CoinGecko. Estimates total mcap from BTC+ETH."""
    _ensure_tables()
    with sqlite3.connect(str(CACHE_DB)) as conn:
        cached = dict(
            conn.execute(
                "SELECT date, dominance FROM btc_dominance WHERE date >= ? AND date <= ?",
                (start_date, end_date),
            ).fetchall()
        )

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
            for p in btc_r.json().get("market_caps", [])
            if p[1]
        }

        eth_r = await client.get(
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=30,
        )
        eth_r.raise_for_status()
        eth_caps = {
            datetime.fromtimestamp(p[0] / 1000, UTC).strftime("%Y-%m-%d"): p[1]
            for p in eth_r.json().get("market_caps", [])
            if p[1]
        }

    # BTC+ETH ≈ 66.4% of total market (stable ratio). Estimate total, then compute dominance.
    with sqlite3.connect(str(CACHE_DB)) as conn:
        for d, btc_cap in btc_caps.items():
            if start_date <= d <= end_date:
                eth_cap = eth_caps.get(d, 0)
                if eth_cap > 0:
                    est_total = (btc_cap + eth_cap) / _BTC_ETH_MARKET_SHARE
                    dom = round(btc_cap / est_total * 100, 2)
                else:
                    dom = _BTC_DOMINANCE_FALLBACK
                conn.execute(
                    "INSERT OR REPLACE INTO btc_dominance (date, dominance) VALUES (?,?)",
                    (d, dom),
                )
                cached[d] = dom
        conn.commit()
    return {k: v for k, v in cached.items() if start_date <= k <= end_date}


async def fetch_fred_series(series_id: str, start_date: str, end_date: str, api_key: str = "") -> dict[str, float]:
    """Fetch a FRED time series. Returns {date_str: value}. Fills weekends/holidays forward."""
    _ensure_tables()
    if not api_key:
        api_key = os.environ.get("FRED_API_KEY", "")
    with sqlite3.connect(str(CACHE_DB)) as conn:
        cached = dict(
            conn.execute(
                "SELECT date, value FROM fred_series WHERE series_id=? AND date>=? AND date<=?",
                (series_id, start_date, end_date),
            ).fetchall()
        )

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    expected = (datetime.fromisoformat(end_date).replace(tzinfo=UTC) - start_dt).days
    if len(cached) >= expected * 0.6:  # FRED has no weekends, so ~60% coverage is full
        return cached

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            },
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])

    with sqlite3.connect(str(CACHE_DB)) as conn:
        for o in obs:
            if o["value"] != ".":  # FRED uses "." for missing
                val = float(o["value"])
                conn.execute(
                    "INSERT OR REPLACE INTO fred_series (series_id, date, value) VALUES (?,?,?)",
                    (series_id, o["date"], val),
                )
                cached[o["date"]] = val
        conn.commit()

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
    """Derive proxy news sentiment from recent price action and momentum quality.
    Returns (sentiment_score -1..1, key_events list)."""
    if idx < 14:
        return 0.0, []

    c_now = candles[idx][4]
    c_7d = candles[max(0, idx - 7)][4]
    c_14d = candles[max(0, idx - 14)][4]
    ret_7d = (c_now - c_7d) / c_7d
    ret_14d = (c_now - c_14d) / c_14d

    # Trend acceleration: compare recent 7d pace vs prior 7d pace
    # If 7d return > half of 14d return, trend is accelerating
    prior_7d_ret = ret_14d - ret_7d  # approximate return of the 7d before the recent 7d
    accelerating = abs(ret_7d) > abs(prior_7d_ret) and ret_7d * ret_14d > 0

    # Volatility spike detection
    recent_returns = [
        abs(candles[i][4] - candles[i - 1][4]) / candles[i - 1][4] for i in range(max(1, idx - 14), idx + 1)
    ]
    avg_vol = sum(recent_returns) / len(recent_returns) if recent_returns else 0.01
    today_vol = recent_returns[-1] if recent_returns else 0

    events = []
    if abs(ret_7d) > 0.10:
        events.append(f"BTC {'surged' if ret_7d > 0 else 'crashed'} {ret_7d:.1%} in 7 days")
    if abs(ret_14d) > 0.15:
        events.append(f"BTC 14d move: {ret_14d:+.1%} ({'accelerating' if accelerating else 'decelerating'})")
    if today_vol > avg_vol * 2:
        events.append(f"Volatility spike: {today_vol:.2%} vs avg {avg_vol:.2%}")
    if accelerating and abs(ret_7d) > 0.05:
        events.append(f"Momentum accelerating: 7d {ret_7d:+.1%} vs prior 7d {prior_7d_ret:+.1%}")

    # Blend 7d return with acceleration signal for richer sentiment
    accel_bonus = 0.15 if accelerating else -0.05
    raw_sentiment = ret_7d * 4 + (accel_bonus if ret_7d > 0 else -accel_bonus)
    sentiment = max(-1.0, min(1.0, raw_sentiment))
    return sentiment, events


async def fetch_futures_volume(symbol: str, start_date: str, end_date: str) -> dict[str, dict]:
    """Fetch daily futures volume from Binance. Returns {date: {volume, quote_volume}}."""
    _ensure_tables()
    with sqlite3.connect(str(CACHE_DB)) as conn:
        cached = {
            r[0]: {"volume": r[1], "quote_volume": r[2]}
            for r in conn.execute(
                "SELECT date, volume, quote_volume FROM futures_volume WHERE date >= ? AND date <= ?",
                (start_date, end_date),
            ).fetchall()
        }

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    if len(cached) >= (end_dt - start_dt).days * 0.9:
        return cached

    import ccxt.async_support as ccxt_async

    exchange = ccxt_async.binance({"options": {"defaultType": "future"}})
    try:
        pair = f"{symbol}/USDT"
        all_candles = []
        cursor = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000) + 86400000
        while cursor < end_ms:
            batch = await exchange.fetch_ohlcv(pair, "1d", since=cursor, limit=500)
            if not batch:
                break
            all_candles.extend(batch)
            cursor = batch[-1][0] + 86400000
            if len(batch) < 500:
                break

        result = {}
        with sqlite3.connect(str(CACHE_DB)) as conn:
            for c in all_candles:
                d = datetime.fromtimestamp(c[0] / 1000, UTC).strftime("%Y-%m-%d")
                if start_date <= d <= end_date:
                    result[d] = {"volume": c[5], "quote_volume": c[5] * c[4]}
                    conn.execute(
                        "INSERT OR REPLACE INTO futures_volume VALUES (?, ?, ?)",
                        (d, c[5], c[5] * c[4]),
                    )
            conn.commit()
        return result
    finally:
        await exchange.close()
