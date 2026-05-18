"""spec 022 FR-022-21 — Heartbeat events + external skill fetch 24h sliding-window aggregators.

Three aggregators:
- HeartbeatPollAggregator: count of heartbeat polls in the last 24h.
- HeartbeatPollLagAggregator: lag (seconds between events) sliding-window average.
- ExternalSkillFetchAggregator: count of external skill fetches in the last 24h.

All use the same deque + threading.Lock pattern as CacheMetricsAggregator (spec 020a).
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import time


class HeartbeatPollAggregator:
    """Count of heartbeat poll requests in a 24h sliding window.

    Thread-safe. Each call to record() pushes a timestamp; count() returns
    the number of polls within the window.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[float] = deque()
        self._lock = Lock()

    def record(self, client_identifier: str = "") -> None:
        """Record a heartbeat poll event (client_identifier is stored separately)."""
        with self._lock:
            now = time()
            self._buffer.append(now)
            self._evict(now)

    def count(self) -> int:
        """Return the number of polls in the current window."""
        with self._lock:
            self._evict(time())
            return len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0] < cutoff:
            self._buffer.popleft()


class HeartbeatPollLagAggregator:
    """Average lag (seconds) between consecutive heartbeat polls in a 24h window.

    Thread-safe. Each call to record(lag_seconds) pushes a lag observation.
    average() returns the sliding-window mean.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def record(self, lag_seconds: float) -> None:
        """Push a lag observation into the sliding window."""
        with self._lock:
            now = time()
            self._buffer.append((now, float(lag_seconds)))
            self._evict(now)

    def average(self) -> float:
        """Return average lag over the sliding window. Returns 0.0 if empty."""
        with self._lock:
            self._evict(time())
            if not self._buffer:
                return 0.0
            return sum(lag for _, lag in self._buffer) / len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()


class ExternalSkillFetchAggregator:
    """Count of external skill fetch requests in a 24h sliding window.

    Thread-safe. Each call to record() pushes a timestamp observation.
    """

    def __init__(self, window_seconds: int = 86400) -> None:
        self._window = window_seconds
        self._buffer: deque[float] = deque()
        self._lock = Lock()

    def record(self, skill_name: str = "", client_identifier: str = "", response_status: int = 200) -> None:
        """Record an external skill fetch event."""
        with self._lock:
            now = time()
            self._buffer.append(now)
            self._evict(now)

    def count(self) -> int:
        """Return the number of fetches in the current window."""
        with self._lock:
            self._evict(time())
            return len(self._buffer)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0] < cutoff:
            self._buffer.popleft()


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized by metrics endpoint)
# ---------------------------------------------------------------------------

_heartbeat_poll_aggregator: HeartbeatPollAggregator | None = None
_heartbeat_poll_lag_aggregator: HeartbeatPollLagAggregator | None = None
_external_skill_fetch_aggregator: ExternalSkillFetchAggregator | None = None


def get_heartbeat_poll_aggregator() -> HeartbeatPollAggregator:
    """Return the process-global HeartbeatPollAggregator instance."""
    global _heartbeat_poll_aggregator
    if _heartbeat_poll_aggregator is None:
        _heartbeat_poll_aggregator = HeartbeatPollAggregator()
    return _heartbeat_poll_aggregator


def get_heartbeat_poll_lag_aggregator() -> HeartbeatPollLagAggregator:
    """Return the process-global HeartbeatPollLagAggregator instance."""
    global _heartbeat_poll_lag_aggregator
    if _heartbeat_poll_lag_aggregator is None:
        _heartbeat_poll_lag_aggregator = HeartbeatPollLagAggregator()
    return _heartbeat_poll_lag_aggregator


def get_external_skill_fetch_aggregator() -> ExternalSkillFetchAggregator:
    """Return the process-global ExternalSkillFetchAggregator instance."""
    global _external_skill_fetch_aggregator
    if _external_skill_fetch_aggregator is None:
        _external_skill_fetch_aggregator = ExternalSkillFetchAggregator()
    return _external_skill_fetch_aggregator
