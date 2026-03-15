"""Dashboard data loading layer.

Centralizes all cached data access for the dashboard pages.
The run_async() helper executes async coroutines from synchronous Streamlit code
using a dedicated module-level event loop (separate from the main app loop).

All @st.cache_data(ttl=N) decorated functions live here. Page modules must NOT
access data sources directly — they call these functions instead.

TTL tiers:
- 10 s  — portfolio, journal, commit detail, risk status (near-real-time)
- 30 s  — scheduler status, metrics summary (cross-process via HTTP)
- 300 s — backtest sessions (immutable historical data)

Error policy:
- DB exceptions propagate to the caller (page layer catches them).
- HTTP exceptions are swallowed and return None (graceful degradation).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
import streamlit as st

if TYPE_CHECKING:
    from cryptotrader.models import DecisionCommit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified config resolution — single source of truth for all pages.
# Priority: environment variable > config file > default.
# ---------------------------------------------------------------------------


@st.cache_resource
def get_dashboard_config() -> dict[str, Any]:
    """Return a dict with db_url, redis_url, and api_base_url for all pages.

    Resolution order for each key:
    1. Application config file (load_config().infrastructure) — same source as API
    2. Environment variable (DATABASE_URL, REDIS_URL, API_BASE_URL)
    3. Sensible default

    Returns:
        Dict with keys: db_url, redis_url, api_base_url.
    """
    db_url: str | None = None
    redis_url: str | None = None
    api_base_url: str = os.environ.get("API_BASE_URL", "")

    # Primary source: config file (same as API server uses).
    try:
        from cryptotrader.config import load_config

        cfg = load_config()
        infra = cfg.infrastructure
        db_url = infra.database_url or None
        redis_url = infra.redis_url or None
    except Exception:
        logger.debug("Failed to load config for dashboard", exc_info=True)

    # Fall back to environment variables if config is empty.
    if not db_url:
        db_url = os.environ.get("DATABASE_URL") or None
    if not redis_url:
        redis_url = os.environ.get("REDIS_URL") or None

    # Default API base URL uses the CLI serve default port (8003).
    if not api_base_url:
        api_base_url = "http://localhost:8003"

    return {"db_url": db_url, "redis_url": redis_url, "api_base_url": api_base_url}


# ---------------------------------------------------------------------------
# Module-level event loop — reused across calls to avoid loop-creation overhead.
# Kept separate from the main application event loop to avoid pool conflicts.
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run an async coroutine synchronously using a persistent module-level loop.

    Creates a fresh event loop if none exists or if the current one has been closed.
    Re-raises any exception the coroutine raises.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The coroutine's return value.
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10)
def load_portfolio(db_url: str | None) -> dict[str, Any]:
    """Load portfolio summary from PortfolioManager.

    DB exceptions propagate to the caller.

    Args:
        db_url: Database URL or None for in-memory mode.

    Returns:
        Portfolio summary dict with keys: account_id, positions, cash, total_value.
    """
    from cryptotrader.portfolio.manager import PortfolioManager

    pm = PortfolioManager(database_url=db_url)
    return run_async(pm.get_portfolio())


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10)
def load_journal(
    db_url: str | None,
    limit: int = 20,
    pair: str | None = None,
    offset: int = 0,
) -> list[DecisionCommit]:
    """Load decision commit history from JournalStore.

    DB exceptions propagate to the caller.

    Args:
        db_url: Database URL or None for in-memory mode.
        limit: Maximum number of commits to return.
        pair: Optional trading pair filter (e.g. "BTC/USDT").
        offset: Number of most-recent commits to skip before applying limit.

    Returns:
        List of DecisionCommit objects, newest-first.
    """
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(database_url=db_url)
    # JournalStore.log() does not support offset natively; we replicate it here
    # by fetching limit + offset and discarding the first `offset` entries.
    fetch_limit = limit + offset
    commits = run_async(store.log(limit=fetch_limit, pair=pair))
    return commits[offset : offset + limit]


@st.cache_data(ttl=10)
def load_commit_detail(db_url: str | None, commit_hash: str) -> DecisionCommit | None:
    """Load a single DecisionCommit by hash.

    DB exceptions propagate to the caller.

    Args:
        db_url: Database URL or None for in-memory mode.
        commit_hash: The hex hash of the commit to retrieve.

    Returns:
        The matching DecisionCommit or None if not found.
    """
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(database_url=db_url)
    return run_async(store.show(commit_hash))


# ---------------------------------------------------------------------------
# Risk status
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10)
def load_risk_status(redis_url: str | None) -> dict[str, Any] | None:
    """Load risk state from RedisStateManager.

    Returns None when Redis is unavailable or no URL is configured.
    DB/Redis exceptions are handled internally; returns None on failure.

    Args:
        redis_url: Redis connection URL or None.

    Returns:
        Risk status dict or None if Redis is not reachable.
    """
    from cryptotrader.risk.state import RedisStateManager

    if not redis_url:
        return None

    rsm = RedisStateManager(redis_url=redis_url)
    if not rsm.available:
        return None

    try:
        hourly, daily = run_async(rsm.get_trade_counts())
        circuit_breaker_active = run_async(rsm.is_circuit_breaker_active())
        return {
            "hourly_trade_count": hourly,
            "daily_trade_count": daily,
            "circuit_breaker_active": circuit_breaker_active,
        }
    except Exception:
        logger.debug("Failed to read risk status from Redis", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# HTTP-based cross-process data (Scheduler status, Metrics summary)
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT_SECONDS = 5


@st.cache_data(ttl=30)
def load_scheduler_status(api_base_url: str) -> dict[str, Any] | None:
    """Load Scheduler status from FastAPI /scheduler/status endpoint.

    Swallows all HTTP/network exceptions and returns None on failure.

    Args:
        api_base_url: Base URL of the FastAPI server (e.g. "http://localhost:8000").

    Returns:
        Parsed JSON dict or None on any error.
    """
    url = f"{api_base_url.rstrip('/')}/scheduler/status"
    try:
        resp = httpx.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.debug("load_scheduler_status failed for %s", url, exc_info=True)
        return None


@st.cache_data(ttl=30)
def load_metrics_summary(api_base_url: str) -> dict[str, Any] | None:
    """Load Prometheus metrics snapshot from FastAPI /metrics/summary endpoint.

    Swallows all HTTP/network exceptions and returns None on failure.

    Args:
        api_base_url: Base URL of the FastAPI server (e.g. "http://localhost:8000").

    Returns:
        Parsed JSON dict or None on any error.
    """
    url = f"{api_base_url.rstrip('/')}/metrics/summary"
    try:
        resp = httpx.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.debug("load_metrics_summary failed for %s", url, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Backtest sessions
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def list_backtest_sessions() -> list[str]:
    """List all backtest session IDs from the sessions directory.

    Returns:
        Sorted list of session ID strings.
    """
    from cryptotrader.backtest.session import list_sessions

    return list_sessions()


@st.cache_data(ttl=300)
def load_backtest_session(session_id: str) -> list[dict]:
    """Load commit records from a backtest session's JSONL file.

    Args:
        session_id: Session identifier returned by list_backtest_sessions().

    Returns:
        List of raw commit dicts (as stored in commits.jsonl), or [] if not found.
    """
    from cryptotrader.backtest.session import load_commits

    return load_commits(session_id)
