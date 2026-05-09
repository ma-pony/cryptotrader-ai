"""spec 022 FR-D13 -- Evolution daemon sliding-window metrics aggregators.

Three in-process aggregators mirroring spec 020a CacheMetricsAggregator pattern
(deque + Lock + sliding window).  The metrics endpoint (FR-D14) reads these
singletons and exposes them as Prometheus Gauges.

Because the daemon runs as a *separate docker-compose service* from the API,
cross-process state is communicated via Redis (sorted set
``evolution_daemon:events``).  The aggregators here are used by:

  * The daemon process -- to record events after each run_once() call.
  * The API /metrics process -- to read events from Redis on each scrape and
    compute the gauge values lazily.

The Redis-backed helpers (record_run_event / get_run_count_24h / etc.) are
thin wrappers so the Prometheus endpoint never imports daemon internals directly.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from threading import Lock
from time import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process aggregators (spec 020a pattern)
# ---------------------------------------------------------------------------


class DaemonRunCountAggregator:
    """24h sliding window count of daemon runs.

    Thread-safe deque ring buffer; evicts entries older than 24h on every
    record() / count() call.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[float] = deque()
        self._lock = Lock()

    def record(self) -> None:
        """Record one daemon run at current timestamp."""
        with self._lock:
            now = time()
            self._buffer.append(now)
            self._evict(now)

    def count(self) -> int:
        """Return number of runs in the sliding window."""
        with self._lock:
            self._evict(time())
            return len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0] < cutoff:
            self._buffer.popleft()


class DaemonLLMFailureAggregator:
    """24h sliding window LLM failure rate.

    Each record() call pushes (timestamp, failed: bool).
    failure_rate() returns successes/total over the window.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[tuple[float, bool]] = deque()
        self._lock = Lock()

    def record(self, *, failed: bool) -> None:
        """Record one skill_proposal outcome (failed=True if LLM raised)."""
        with self._lock:
            now = time()
            self._buffer.append((now, failed))
            self._evict(now)

    def failure_rate(self) -> float:
        """Return fraction of failed calls in the sliding window (0.0 if empty)."""
        with self._lock:
            self._evict(time())
            if not self._buffer:
                return 0.0
            return sum(1 for _, f in self._buffer if f) / len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()


class SkillProposalDraftAggregator:
    """7-day sliding window total draft files created."""

    def __init__(self, window_seconds: int = 7 * 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[float] = deque()
        self._lock = Lock()

    def record(self) -> None:
        """Record one draft creation event at current timestamp."""
        with self._lock:
            now = time()
            self._buffer.append(now)
            self._evict(now)

    def total(self) -> int:
        """Return total drafts created in the sliding window."""
        with self._lock:
            self._evict(time())
            return len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0] < cutoff:
            self._buffer.popleft()


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_run_count_agg: DaemonRunCountAggregator | None = None
_llm_failure_agg: DaemonLLMFailureAggregator | None = None
_draft_agg: SkillProposalDraftAggregator | None = None


def get_run_count_aggregator() -> DaemonRunCountAggregator:
    global _run_count_agg
    if _run_count_agg is None:
        _run_count_agg = DaemonRunCountAggregator()
    return _run_count_agg


def get_llm_failure_aggregator() -> DaemonLLMFailureAggregator:
    global _llm_failure_agg
    if _llm_failure_agg is None:
        _llm_failure_agg = DaemonLLMFailureAggregator()
    return _llm_failure_agg


def get_draft_aggregator() -> SkillProposalDraftAggregator:
    global _draft_agg
    if _draft_agg is None:
        _draft_agg = SkillProposalDraftAggregator()
    return _draft_agg


# ---------------------------------------------------------------------------
# Redis-backed cross-process helpers (FR-D13 Decision 4)
# ---------------------------------------------------------------------------

_REDIS_KEY_RUNS = "evolution_daemon:events:runs"
_REDIS_KEY_LLM_FAILURES = "evolution_daemon:events:llm_failures"
_REDIS_KEY_DRAFTS = "evolution_daemon:events:drafts"

_24H = 86400.0
_7D = 7 * 86400.0


def record_run_event(redis_client: object | None = None) -> None:
    """Write a daemon run event to Redis sorted set (score=timestamp).

    Falls back silently when Redis is unavailable.
    """
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return
    try:
        now = time()
        redis_client.zadd(_REDIS_KEY_RUNS, {json.dumps({"ts": now}): now})
        redis_client.zremrangebyscore(_REDIS_KEY_RUNS, "-inf", now - _24H)
    except Exception:
        logger.debug("record_run_event: redis write failed", exc_info=True)


def record_llm_failure_event(*, failed: bool, redis_client: object | None = None) -> None:
    """Write an LLM outcome event to Redis sorted set."""
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return
    try:
        now = time()
        redis_client.zadd(_REDIS_KEY_LLM_FAILURES, {json.dumps({"ts": now, "failed": failed}): now})
        redis_client.zremrangebyscore(_REDIS_KEY_LLM_FAILURES, "-inf", now - _24H)
    except Exception:
        logger.debug("record_llm_failure_event: redis write failed", exc_info=True)


def record_draft_event(redis_client: object | None = None) -> None:
    """Write a draft creation event to Redis sorted set."""
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return
    try:
        now = time()
        redis_client.zadd(_REDIS_KEY_DRAFTS, {json.dumps({"ts": now}): now})
        redis_client.zremrangebyscore(_REDIS_KEY_DRAFTS, "-inf", now - _7D)
    except Exception:
        logger.debug("record_draft_event: redis write failed", exc_info=True)


def get_run_count_24h_from_redis(redis_client: object | None = None) -> float:
    """Return count of daemon runs in last 24h from Redis sorted set."""
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return 0.0
    try:
        now = time()
        return float(redis_client.zcount(_REDIS_KEY_RUNS, now - _24H, "+inf"))
    except Exception:
        logger.debug("get_run_count_24h_from_redis: redis read failed", exc_info=True)
        return 0.0


def get_llm_failure_rate_24h_from_redis(redis_client: object | None = None) -> float:
    """Return LLM failure rate in last 24h from Redis sorted set."""
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return 0.0
    try:
        now = time()
        members = redis_client.zrangebyscore(_REDIS_KEY_LLM_FAILURES, now - _24H, "+inf")
        if not members:
            return 0.0
        total = len(members)
        failures = sum(1 for m in members if json.loads(m).get("failed", False))
        return failures / total
    except Exception:
        logger.debug("get_llm_failure_rate_24h_from_redis: redis read failed", exc_info=True)
        return 0.0


def get_draft_count_7d_from_redis(redis_client: object | None = None) -> float:
    """Return skill proposal draft count in last 7 days from Redis sorted set."""
    if redis_client is None:
        redis_client = _get_redis()
    if redis_client is None:
        return 0.0
    try:
        now = time()
        return float(redis_client.zcount(_REDIS_KEY_DRAFTS, now - _7D, "+inf"))
    except Exception:
        logger.debug("get_draft_count_7d_from_redis: redis read failed", exc_info=True)
        return 0.0


def _get_redis() -> object | None:
    """Lazily obtain Redis client from config. Returns None on any error."""
    try:
        import redis as redis_lib

        from cryptotrader.config import load_config

        cfg = load_config()
        redis_url = cfg.infrastructure.redis_url or "redis://localhost:6379/0"
        return redis_lib.from_url(redis_url, decode_responses=True)
    except Exception:
        logger.debug("daemon_metrics: Redis unavailable", exc_info=True)
        return None
