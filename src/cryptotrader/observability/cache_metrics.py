"""spec 020a FR-Z18 — LLM prompt cache hit rate 24h sliding window aggregator.

CacheMetricsAggregator: thread-safe deque-based ring buffer.
- record(hit_rate): push (timestamp, hit_rate) into buffer
- average(): return 24h sliding window average hit rate
- Evicts entries older than window_seconds on every read/write
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import time


class CacheMetricsAggregator:
    """LLM prompt cache hit rate aggregator (24h sliding window).

    Thread-safe. Uses a deque as a ring buffer; evicts expired entries
    on every record() / average() call. Memory bounded:
    ~5 LLM/cycle * 24 cycles/day = ~120 entries max.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def record(self, hit_rate: float) -> None:
        """Push a new hit_rate observation into the sliding window."""
        with self._lock:
            now = time()
            self._buffer.append((now, float(hit_rate)))
            self._evict(now)

    def average(self) -> float:
        """Return the average hit_rate over the sliding window. Returns 0.0 if empty."""
        with self._lock:
            self._evict(time())
            if not self._buffer:
                return 0.0
            return sum(r for _, r in self._buffer) / len(self._buffer)

    def _evict(self, now: float) -> None:
        """Remove entries older than window_seconds (must hold lock)."""
        cutoff = now - self._window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()


# Module-level singleton (lazy-initialized by metrics endpoint)
_cache_metrics_aggregator: CacheMetricsAggregator | None = None


def get_cache_metrics_aggregator() -> CacheMetricsAggregator:
    """Return the process-global CacheMetricsAggregator instance."""
    global _cache_metrics_aggregator
    if _cache_metrics_aggregator is None:
        _cache_metrics_aggregator = CacheMetricsAggregator()
    return _cache_metrics_aggregator
