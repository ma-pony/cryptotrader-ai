"""Unified market data store — SQLite-backed persistence for all data sources.

Stores time-series data from all providers (FRED, Fear&Greed, SoSoValue ETF, funding rate, etc.)
with intelligent caching to minimize API calls and avoid rate limiting.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DB = Path.home() / ".cryptotrader" / "market_data.db"

# Module-level singleton connection (lazy init, check_same_thread=False for multi-thread use)
_conn: sqlite3.Connection | None = None

# Minimum interval between API fetches per source (seconds)
_RATE_LIMITS = {
    "sosovalue_etf_metrics": 300,  # 5 min
    "sosovalue_etf_history": 3600,  # 1 hour
    "sosovalue_news": 600,  # 10 min
    "fred_DFF": 3600,  # 1 hour
    "fred_DTWEXBGS": 3600,  # 1 hour
    "fear_greed": 3600,  # 1 hour
    "btc_dominance": 600,  # 10 min
    "funding_rate": 300,  # 5 min
    "cryptoquant": 600,  # 10 min
    "binance_derivatives": 3600,  # 1 hour
    "binance_eth_derivatives": 3600,  # 1 hour
    "binance_funding": 3600,  # 1 hour
    "defillama": 3600,  # 1 hour
    "defillama_extra": 3600,  # 1 hour
    "coingecko": 3600,  # 1 hour
    "coingecko_eth": 3600,  # 1 hour
    "blockchain_info": 3600,  # 1 hour
    "fred_multi": 3600,  # 1 hour
    "sosovalue_eth_etf": 3600,  # 1 hour
    "stablecoin_total": 3600,  # 1 hour
    "blockchain_extra": 3600,  # 1 hour
    "blockchain_extended": 3600,  # 1 hour
    "mempool_space": 3600,  # 1 hour
    "defillama_chains": 3600,  # 1 hour
    "defillama_perps": 3600,  # 1 hour
    "defillama_options": 3600,  # 1 hour
    "binance_funding_full": 3600,  # 1 hour
    "coinpaprika": 3600,  # 1 hour
    # Live collector cache keys
    "live_news_rss": 600,  # 10 min
    "live_social_buzz": 600,  # 10 min
    "live_coinglass": 600,  # 10 min
    "live_whale_alert": 600,  # 10 min
    "live_cryptoquant": 600,  # 10 min
}


def get_cached_or_none(source: str, date: str | None = None) -> dict | list | float | None:
    """Return cached data if available, else None.

    Two modes:
    - date=None (live): TTL-based check, returns latest data if still fresh
    - date="2025-07-15" (backtest): exact date lookup, no TTL — historical data never expires

    Use this in collectors to check the store before calling external APIs.
    """
    if date is not None:
        # Backtest mode: look up by exact date, no TTL
        return get_data(source, date)

    # Live mode: TTL-based check
    if not _should_fetch(source):
        latest = get_latest(source, 1)
        if latest:
            return latest[0][1]
    return None


def cache_result(source: str, data: dict | list | float, date: str | None = None) -> None:
    """Store data and record fetch timestamp.

    Use this after a successful API call to persist the result.
    """
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
    store_data(source, date, data)
    _record_fetch(source)


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    DATA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATA_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS market_data (
            source TEXT NOT NULL,
            date TEXT NOT NULL,
            data TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (source, date)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS fetch_log (
            source TEXT PRIMARY KEY,
            last_fetch REAL NOT NULL
        )"""
    )
    _conn = conn
    return _conn


def _should_fetch(source: str) -> bool:
    """Check if enough time has passed since last fetch for this source."""
    conn = _get_conn()
    row = conn.execute("SELECT last_fetch FROM fetch_log WHERE source=?", (source,)).fetchone()
    if not row:
        return True
    min_interval = _RATE_LIMITS.get(source, 300)
    return (time.time() - row[0]) >= min_interval


def _record_fetch(source: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO fetch_log (source, last_fetch) VALUES (?, ?)",
        (source, time.time()),
    )
    conn.commit()


def store_data(source: str, date: str, data: dict | list | float) -> None:
    """Store a single data point."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO market_data (source, date, data, updated_at) VALUES (?, ?, ?, ?)",
        (source, date, json.dumps(data, default=str), time.time()),
    )
    conn.commit()


def store_batch(
    source: str,
    records: list[tuple[str, dict | list | float]],
    *,
    forward_fill: bool = False,
) -> None:
    """Store multiple data points efficiently. records = [(date, data), ...]

    Args:
        forward_fill: If True, fill gaps (weekends/holidays) by carrying the
                      last known value forward so every calendar day has data.
    """
    if not records:
        return
    if forward_fill:
        records = _forward_fill(records)
    conn = _get_conn()
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO market_data (source, date, data, updated_at) VALUES (?, ?, ?, ?)",
        [(source, date, json.dumps(data, default=str), now) for date, data in records],
    )
    conn.commit()
    logger.info("Stored %d records for source=%s", len(records), source)


def _forward_fill(records: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Fill date gaps by repeating the previous value for each missing day."""
    if len(records) < 2:
        return records
    sorted_recs = sorted(records, key=lambda r: r[0])
    filled: list[tuple[str, Any]] = []
    for i, (date_str, data) in enumerate(sorted_recs):
        filled.append((date_str, data))
        if i + 1 < len(sorted_recs):
            cur = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            nxt = datetime.strptime(sorted_recs[i + 1][0], "%Y-%m-%d").replace(tzinfo=UTC)
            gap = nxt - cur
            for d in range(1, gap.days):
                fill_date = (cur + timedelta(days=d)).strftime("%Y-%m-%d")
                filled.append((fill_date, data))
    return filled


def get_data(source: str, date: str) -> dict | list | float | None:
    """Get a single data point."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM market_data WHERE source=? AND date=?", (source, date)).fetchone()
    if row:
        return json.loads(row[0])
    return None


def get_range(source: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Get data points in a date range. Returns {date: data}."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, data FROM market_data WHERE source=? AND date>=? AND date<=? ORDER BY date",
        (source, start_date, end_date),
    ).fetchall()
    return {r[0]: json.loads(r[1]) for r in rows}


def get_latest(source: str, limit: int = 1) -> list[tuple[str, Any]]:
    """Get most recent data points. Returns [(date, data), ...]."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, data FROM market_data WHERE source=? ORDER BY date DESC LIMIT ?",
        (source, limit),
    ).fetchall()
    return [(r[0], json.loads(r[1])) for r in rows]


def count_records(source: str) -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM market_data WHERE source=?", (source,)).fetchone()
    return row[0] if row else 0
