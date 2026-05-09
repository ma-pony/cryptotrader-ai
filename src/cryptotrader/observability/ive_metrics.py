"""spec 020a FR-Z18 — IVE classify_case failure rate 1h sliding window aggregator.

IveMetricsAggregator: thread-safe deque-based ring buffer.
- record(success): push (timestamp, success:bool) into buffer
- failure_rate(): return 1h sliding window failure rate (0.0 = no failures)
- Evicts entries older than window_seconds on every read/write
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import time


class IveMetricsAggregator:
    """IVE classify_case failure rate aggregator (1h sliding window).

    Thread-safe. Stores (timestamp, success) pairs; computes failure_rate
    as (failed_count / total_count) over the window.
    Memory bounded: ~10 IVE/cycle * 1h/cycle = ~10 entries max.
    """

    def __init__(self, window_seconds: int = 3600) -> None:
        self._window = window_seconds
        self._buffer: deque[tuple[float, bool]] = deque()
        self._lock = Lock()

    def record(self, success: bool) -> None:
        """Push a new success/failure observation into the sliding window."""
        with self._lock:
            now = time()
            self._buffer.append((now, bool(success)))
            self._evict(now)

    def failure_rate(self) -> float:
        """Return the failure rate over the sliding window. Returns 0.0 if empty."""
        with self._lock:
            self._evict(time())
            if not self._buffer:
                return 0.0
            total = len(self._buffer)
            failed = sum(1 for _, ok in self._buffer if not ok)
            return failed / total

    def _evict(self, now: float) -> None:
        """Remove entries older than window_seconds (must hold lock)."""
        cutoff = now - self._window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()


# Module-level singleton (lazy-initialized by metrics endpoint)
_ive_metrics_aggregator: IveMetricsAggregator | None = None


def get_ive_metrics_aggregator() -> IveMetricsAggregator:
    """Return the process-global IveMetricsAggregator instance."""
    global _ive_metrics_aggregator
    if _ive_metrics_aggregator is None:
        _ive_metrics_aggregator = IveMetricsAggregator()
    return _ive_metrics_aggregator
