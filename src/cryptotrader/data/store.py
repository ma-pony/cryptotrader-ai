"""Unified market data store — SQLite-backed persistence for all data sources.

Stores time-series data from all providers (FRED, Fear&Greed, SoSoValue ETF, funding rate, etc.)
with intelligent caching to minimize API calls and avoid rate limiting.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DB = Path.home() / ".cryptotrader" / "market_data.db"

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
}


def _get_conn() -> sqlite3.Connection:
    DATA_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATA_DB))
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
    return conn


def _should_fetch(source: str) -> bool:
    """Check if enough time has passed since last fetch for this source."""
    conn = _get_conn()
    row = conn.execute("SELECT last_fetch FROM fetch_log WHERE source=?", (source,)).fetchone()
    conn.close()
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
    conn.close()


def store_data(source: str, date: str, data: dict | list | float) -> None:
    """Store a single data point."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO market_data (source, date, data, updated_at) VALUES (?, ?, ?, ?)",
        (source, date, json.dumps(data, default=str), time.time()),
    )
    conn.commit()
    conn.close()


def store_batch(source: str, records: list[tuple[str, dict | list | float]]) -> None:
    """Store multiple data points efficiently. records = [(date, data), ...]"""
    if not records:
        return
    conn = _get_conn()
    now = time.time()
    conn.executemany(
        "INSERT OR REPLACE INTO market_data (source, date, data, updated_at) VALUES (?, ?, ?, ?)",
        [(source, date, json.dumps(data, default=str), now) for date, data in records],
    )
    conn.commit()
    conn.close()
    logger.info("Stored %d records for source=%s", len(records), source)


def get_data(source: str, date: str) -> dict | list | float | None:
    """Get a single data point."""
    conn = _get_conn()
    row = conn.execute("SELECT data FROM market_data WHERE source=? AND date=?", (source, date)).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def get_range(source: str, start_date: str, end_date: str) -> dict[str, any]:
    """Get data points in a date range. Returns {date: data}."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, data FROM market_data WHERE source=? AND date>=? AND date<=? ORDER BY date",
        (source, start_date, end_date),
    ).fetchall()
    conn.close()
    return {r[0]: json.loads(r[1]) for r in rows}


def get_latest(source: str, limit: int = 1) -> list[tuple[str, any]]:
    """Get most recent data points. Returns [(date, data), ...]."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT date, data FROM market_data WHERE source=? ORDER BY date DESC LIMIT ?",
        (source, limit),
    ).fetchall()
    conn.close()
    return [(r[0], json.loads(r[1])) for r in rows]


def count_records(source: str) -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM market_data WHERE source=?", (source,)).fetchone()
    conn.close()
    return row[0] if row else 0
