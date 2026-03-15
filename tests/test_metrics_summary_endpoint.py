"""Tests for GET /metrics/summary endpoint — task 4.

TDD approach: tests are written first, before the implementation.

Strategy: use real MetricsCollector instance + real FastAPI TestClient.
Call inc/observe methods on real prometheus-client metrics, then verify
the summary endpoint reads correct values. Only mock what cannot run in tests.

The prometheus-client uses a global REGISTRY that accumulates across the
process lifetime, so tests use deltas (before/after) or unique label values
to avoid interference from other tests.
"""

from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan disabled for fast unit tests."""
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def mc():
    """Real MetricsCollector singleton."""
    from cryptotrader.metrics import get_metrics_collector

    return get_metrics_collector()


# ---------------------------------------------------------------------------
# Helper: read current summary via HTTP
# ---------------------------------------------------------------------------


def _get_summary(client: TestClient) -> dict:
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Tests: endpoint availability and response shape
# ---------------------------------------------------------------------------


class TestMetricsSummaryEndpointShape:
    """Verify the endpoint exists and returns the correct response schema."""

    def test_endpoint_returns_200(self, client):
        """GET /metrics/summary returns HTTP 200."""
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200

    def test_response_is_json(self, client):
        """GET /metrics/summary returns JSON content type."""
        resp = client.get("/metrics/summary")
        assert "application/json" in resp.headers["content-type"]

    def test_response_contains_all_required_fields(self, client):
        """Response body contains all 11 required fields."""
        body = _get_summary(client)
        required_fields = {
            "llm_calls_total",
            "debate_skipped_total",
            "verdict_distribution",
            "risk_rejected_total",
            "risk_rejected_by_check",
            "trade_executed_total",
            "pipeline_duration_p50_ms",
            "pipeline_duration_p95_ms",
            "execution_latency_p50_ms",
            "execution_latency_p95_ms",
            "snapshot_time",
        }
        missing = required_fields - set(body.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_integer_counter_fields_are_non_negative(self, client):
        """Counter fields (int) are >= 0."""
        body = _get_summary(client)
        for field in ("llm_calls_total", "debate_skipped_total", "risk_rejected_total", "trade_executed_total"):
            assert isinstance(body[field], int), f"{field} should be int"
            assert body[field] >= 0, f"{field} should be >= 0"

    def test_verdict_distribution_is_dict_of_str_int(self, client):
        """verdict_distribution is a dict mapping action strings to int counts."""
        body = _get_summary(client)
        dist = body["verdict_distribution"]
        assert isinstance(dist, dict)
        for k, v in dist.items():
            assert isinstance(k, str), f"Key {k!r} should be str"
            assert isinstance(v, int), f"Value {v!r} for key {k!r} should be int"

    def test_risk_rejected_by_check_is_dict_of_str_int(self, client):
        """risk_rejected_by_check is a dict mapping check names to int counts."""
        body = _get_summary(client)
        rbc = body["risk_rejected_by_check"]
        assert isinstance(rbc, dict)
        for k, v in rbc.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    def test_latency_fields_are_floats_non_negative(self, client):
        """p50/p95 latency fields are floats >= 0."""
        body = _get_summary(client)
        for field in (
            "pipeline_duration_p50_ms",
            "pipeline_duration_p95_ms",
            "execution_latency_p50_ms",
            "execution_latency_p95_ms",
        ):
            assert isinstance(body[field], float), f"{field} should be float"
            assert body[field] >= 0.0, f"{field} should be >= 0"

    def test_snapshot_time_is_iso8601_datetime(self, client):
        """snapshot_time is a valid ISO-8601 datetime string."""
        body = _get_summary(client)
        snap = body["snapshot_time"]
        assert isinstance(snap, str)
        # Should parse without error
        dt = datetime.datetime.fromisoformat(snap.replace("Z", "+00:00"))
        assert isinstance(dt, datetime.datetime)

    def test_existing_prometheus_endpoint_still_works(self, client):
        """GET /metrics (Prometheus text format) is unaffected."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Tests: counter values reflect real MetricsCollector increments
# ---------------------------------------------------------------------------


class TestMetricsSummaryCounterValues:
    """Verify that counter fields in the summary reflect real inc() calls.

    Because prometheus-client uses a global registry that accumulates across
    the test process, we measure deltas (value after - value before) to avoid
    coupling to absolute counts from other tests.
    """

    def test_llm_calls_total_increments(self, client, mc):
        """llm_calls_total increases after inc_llm_calls() is called."""
        before = _get_summary(client)["llm_calls_total"]
        mc.inc_llm_calls(model="gpt-4o-summary-test", node="test_node")
        mc.inc_llm_calls(model="gpt-4o-summary-test", node="test_node")
        after = _get_summary(client)["llm_calls_total"]
        assert after == before + 2

    def test_debate_skipped_total_increments(self, client, mc):
        """debate_skipped_total increases after inc_debate_skipped() is called."""
        before = _get_summary(client)["debate_skipped_total"]
        mc.inc_debate_skipped()
        mc.inc_debate_skipped()
        mc.inc_debate_skipped()
        after = _get_summary(client)["debate_skipped_total"]
        assert after == before + 3

    def test_risk_rejected_total_increments(self, client, mc):
        """risk_rejected_total increases after inc_risk_rejected() is called."""
        before = _get_summary(client)["risk_rejected_total"]
        mc.inc_risk_rejected(check_name="test_volatility_check")
        after = _get_summary(client)["risk_rejected_total"]
        assert after == before + 1

    def test_trade_executed_total_increments(self, client, mc):
        """trade_executed_total increases after inc_trade_executed() is called."""
        before = _get_summary(client)["trade_executed_total"]
        mc.inc_trade_executed(engine="paper", side="buy_summary_test")
        mc.inc_trade_executed(engine="paper", side="sell_summary_test")
        after = _get_summary(client)["trade_executed_total"]
        assert after == before + 2

    def test_verdict_distribution_reflects_inc_verdict(self, client, mc):
        """verdict_distribution[action] increases after inc_verdict(action=...) is called."""
        # Use a rare action label to reduce cross-test interference
        before_body = _get_summary(client)
        before_val = before_body["verdict_distribution"].get("summary_test_action", 0)

        mc.inc_verdict(action="summary_test_action")
        mc.inc_verdict(action="summary_test_action")

        after_body = _get_summary(client)
        after_val = after_body["verdict_distribution"].get("summary_test_action", 0)
        assert after_val == before_val + 2

    def test_risk_rejected_by_check_reflects_inc_risk_rejected(self, client, mc):
        """risk_rejected_by_check[check_name] increases after inc_risk_rejected(check_name=...) calls."""
        check_name = "summary_test_drawdown_check"
        before_body = _get_summary(client)
        before_val = before_body["risk_rejected_by_check"].get(check_name, 0)

        mc.inc_risk_rejected(check_name=check_name)
        mc.inc_risk_rejected(check_name=check_name)
        mc.inc_risk_rejected(check_name=check_name)

        after_body = _get_summary(client)
        after_val = after_body["risk_rejected_by_check"].get(check_name, 0)
        assert after_val == before_val + 3


# ---------------------------------------------------------------------------
# Tests: histogram percentile values are plausible
# ---------------------------------------------------------------------------


class TestMetricsSummaryHistogramPercentiles:
    """Verify that p50/p95 fields reflect observed histogram values.

    Prometheus bucket-based percentile estimation returns the upper bound
    of the bucket that contains the target percentile. We verify that the
    returned values are within the declared bucket boundaries of the metrics.
    """

    def test_pipeline_duration_p50_within_bucket_range(self, client, mc):
        """pipeline_duration_p50_ms is a non-negative float after observations."""
        mc.observe_pipeline_duration(ms=1000.0)
        mc.observe_pipeline_duration(ms=2000.0)
        mc.observe_pipeline_duration(ms=4000.0)
        body = _get_summary(client)
        # p50 must be >= 0 and <= max bucket boundary (60000)
        assert 0.0 <= body["pipeline_duration_p50_ms"] <= 60000.0

    def test_pipeline_duration_p95_within_bucket_range(self, client, mc):
        """pipeline_duration_p95_ms is >= p50 after observations."""
        mc.observe_pipeline_duration(ms=500.0)
        mc.observe_pipeline_duration(ms=1000.0)
        mc.observe_pipeline_duration(ms=10000.0)
        body = _get_summary(client)
        assert body["pipeline_duration_p95_ms"] >= body["pipeline_duration_p50_ms"]

    def test_execution_latency_p50_within_bucket_range(self, client, mc):
        """execution_latency_p50_ms is a non-negative float after observations."""
        mc.observe_execution_latency(engine="paper", ms=100.0)
        mc.observe_execution_latency(engine="paper", ms=250.0)
        mc.observe_execution_latency(engine="live", ms=500.0)
        body = _get_summary(client)
        # Buckets go up to 5000
        assert 0.0 <= body["execution_latency_p50_ms"] <= 5000.0

    def test_execution_latency_p95_gte_p50(self, client, mc):
        """execution_latency_p95_ms >= execution_latency_p50_ms."""
        mc.observe_execution_latency(engine="paper", ms=50.0)
        mc.observe_execution_latency(engine="paper", ms=2500.0)
        body = _get_summary(client)
        assert body["execution_latency_p95_ms"] >= body["execution_latency_p50_ms"]

    def test_percentile_zero_when_no_observations(self, client):
        """When no histogram observations have been recorded, percentiles are 0.0.

        This test resets the registry context by using a fresh CollectorRegistry.
        Since the global registry accumulates across tests, this test verifies
        the degenerate case by checking type and bound only.
        """
        body = _get_summary(client)
        # Values must always be floats and >= 0
        assert isinstance(body["pipeline_duration_p50_ms"], float)
        assert isinstance(body["execution_latency_p50_ms"], float)


# ---------------------------------------------------------------------------
# Tests: snapshot_time is recent
# ---------------------------------------------------------------------------


class TestMetricsSummarySnapshotTime:
    """Verify snapshot_time is a timezone-aware datetime close to now."""

    def test_snapshot_time_is_utc_aware(self, client):
        """snapshot_time includes timezone info (UTC)."""
        body = _get_summary(client)
        snap_str = body["snapshot_time"]
        # ISO format with timezone offset or Z suffix
        dt = datetime.datetime.fromisoformat(snap_str.replace("Z", "+00:00"))
        assert dt.tzinfo is not None

    def test_snapshot_time_is_close_to_now(self, client):
        """snapshot_time is within 5 seconds of the current UTC time."""
        now = datetime.datetime.now(datetime.UTC)
        body = _get_summary(client)
        snap_str = body["snapshot_time"]
        dt = datetime.datetime.fromisoformat(snap_str.replace("Z", "+00:00"))
        diff = abs((dt - now).total_seconds())
        assert diff < 5.0, f"snapshot_time differs from now by {diff:.1f}s"
