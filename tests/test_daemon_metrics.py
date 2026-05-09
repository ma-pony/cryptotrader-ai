"""spec 022 T023 -- Unit tests for daemon_metrics.py aggregators.

Tests cover:
- DaemonRunCountAggregator: record / count / sliding window eviction
- DaemonLLMFailureAggregator: record / failure_rate / eviction
- SkillProposalDraftAggregator: record / total / eviction
"""

from __future__ import annotations

from time import time
from unittest.mock import patch

from cryptotrader.observability.daemon_metrics import (
    DaemonLLMFailureAggregator,
    DaemonRunCountAggregator,
    SkillProposalDraftAggregator,
    get_draft_aggregator,
    get_llm_failure_aggregator,
    get_run_count_aggregator,
)

# ---------------------------------------------------------------------------
# DaemonRunCountAggregator
# ---------------------------------------------------------------------------


def test_run_count_record_and_count():
    agg = DaemonRunCountAggregator()
    assert agg.count() == 0
    agg.record()
    assert agg.count() == 1
    agg.record()
    agg.record()
    assert agg.count() == 3


def test_run_count_sliding_window_eviction():
    agg = DaemonRunCountAggregator(window_seconds=10)
    old_ts = time() - 20  # 20s ago, outside 10s window
    with agg._lock:
        agg._buffer.append(old_ts)  # inject stale entry directly
    agg.record()  # this triggers _evict
    assert agg.count() == 1  # stale entry evicted, only fresh one remains


def test_run_count_empty_window():
    agg = DaemonRunCountAggregator(window_seconds=1)
    agg.record()
    # Fake time forward past the window
    future = time() + 2
    with patch("cryptotrader.observability.daemon_metrics.time", return_value=future):
        assert agg.count() == 0


def test_run_count_multiple_eviction():
    agg = DaemonRunCountAggregator(window_seconds=60)
    agg.record()
    agg.record()
    # Both within window
    assert agg.count() == 2


# ---------------------------------------------------------------------------
# DaemonLLMFailureAggregator
# ---------------------------------------------------------------------------


def test_llm_failure_rate_empty():
    agg = DaemonLLMFailureAggregator()
    assert agg.failure_rate() == 0.0


def test_llm_failure_rate_all_success():
    agg = DaemonLLMFailureAggregator()
    agg.record(failed=False)
    agg.record(failed=False)
    assert agg.failure_rate() == 0.0


def test_llm_failure_rate_all_failed():
    agg = DaemonLLMFailureAggregator()
    agg.record(failed=True)
    agg.record(failed=True)
    assert agg.failure_rate() == 1.0


def test_llm_failure_rate_mixed():
    agg = DaemonLLMFailureAggregator()
    agg.record(failed=False)
    agg.record(failed=False)
    agg.record(failed=True)
    assert abs(agg.failure_rate() - 1 / 3) < 0.01


def test_llm_failure_rate_sliding_window_eviction():
    agg = DaemonLLMFailureAggregator(window_seconds=10)
    old_ts = time() - 20
    with agg._lock:
        agg._buffer.append((old_ts, True))  # inject stale failed entry
    agg.record(failed=False)  # triggers evict
    # Only fresh success entry remains -> 0.0 failure rate
    assert agg.failure_rate() == 0.0


def test_llm_failure_rate_window_expired():
    agg = DaemonLLMFailureAggregator(window_seconds=1)
    agg.record(failed=True)
    future = time() + 2
    with patch("cryptotrader.observability.daemon_metrics.time", return_value=future):
        assert agg.failure_rate() == 0.0


# ---------------------------------------------------------------------------
# SkillProposalDraftAggregator
# ---------------------------------------------------------------------------


def test_draft_total_empty():
    agg = SkillProposalDraftAggregator()
    assert agg.total() == 0


def test_draft_total_records():
    agg = SkillProposalDraftAggregator()
    agg.record()
    agg.record()
    agg.record()
    assert agg.total() == 3


def test_draft_total_sliding_window_eviction():
    agg = SkillProposalDraftAggregator(window_seconds=10)
    old_ts = time() - 20
    with agg._lock:
        agg._buffer.append(old_ts)
    agg.record()
    assert agg.total() == 1  # stale evicted


def test_draft_total_window_expired():
    agg = SkillProposalDraftAggregator(window_seconds=1)
    agg.record()
    future = time() + 2
    with patch("cryptotrader.observability.daemon_metrics.time", return_value=future):
        assert agg.total() == 0


# ---------------------------------------------------------------------------
# Singleton getters
# ---------------------------------------------------------------------------


def test_get_run_count_aggregator_singleton():
    a = get_run_count_aggregator()
    b = get_run_count_aggregator()
    assert a is b


def test_get_llm_failure_aggregator_singleton():
    a = get_llm_failure_aggregator()
    b = get_llm_failure_aggregator()
    assert a is b


def test_get_draft_aggregator_singleton():
    a = get_draft_aggregator()
    b = get_draft_aggregator()
    assert a is b


# ---------------------------------------------------------------------------
# Redis-backed helpers (mocked Redis)
# ---------------------------------------------------------------------------


def test_get_run_count_24h_from_redis():
    from cryptotrader.observability.daemon_metrics import get_run_count_24h_from_redis

    class FakeRedis:
        def zcount(self, key, min_score, max_score):
            return 5

    result = get_run_count_24h_from_redis(redis_client=FakeRedis())
    assert result == 5.0


def test_get_llm_failure_rate_24h_from_redis():
    import json

    from cryptotrader.observability.daemon_metrics import get_llm_failure_rate_24h_from_redis

    class FakeRedis:
        def zrangebyscore(self, key, min_score, max_score):
            return [
                json.dumps({"failed": False}),
                json.dumps({"failed": True}),
                json.dumps({"failed": False}),
                json.dumps({"failed": True}),
            ]

    result = get_llm_failure_rate_24h_from_redis(redis_client=FakeRedis())
    assert abs(result - 0.5) < 0.01


def test_get_draft_count_7d_from_redis():
    from cryptotrader.observability.daemon_metrics import get_draft_count_7d_from_redis

    class FakeRedis:
        def zcount(self, key, min_score, max_score):
            return 3

    result = get_draft_count_7d_from_redis(redis_client=FakeRedis())
    assert result == 3.0


def test_redis_helpers_return_zero_on_none():
    from cryptotrader.observability.daemon_metrics import (
        get_draft_count_7d_from_redis,
        get_llm_failure_rate_24h_from_redis,
        get_run_count_24h_from_redis,
    )

    with patch("cryptotrader.observability.daemon_metrics._get_redis", return_value=None):
        assert get_run_count_24h_from_redis() == 0.0
        assert get_llm_failure_rate_24h_from_redis() == 0.0
        assert get_draft_count_7d_from_redis() == 0.0
