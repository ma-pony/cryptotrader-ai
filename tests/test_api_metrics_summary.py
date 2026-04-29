"""Tests for GET /api/metrics/summary (contract shape, FR-808).

The legacy ``/metrics/summary`` returns a flat field list. The contract
endpoint at ``/api/metrics/summary`` returns the data-model shape used by
the React Metrics page:

```
{
  "counters": {
    "trades_total": 142,
    "orders_placed": 138,
    "orders_failed": 4,
    "risk_rejections": 12,
    "debate_skipped_total": 23
  },
  "percentiles": {
    "pipeline_p50_ms": 1250,
    "pipeline_p95_ms": 4800,
    "execution_p50_ms": 320,
    "execution_p95_ms": 880
  },
  "collected_at": "2026-04-16T13:30:00Z"
}
```
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _patch_summary_helpers(
    *,
    trades_total: int = 0,
    risk_rejections: int = 0,
    debate_skipped: int = 0,
    pipeline_p50: float = 0.0,
    pipeline_p95: float = 0.0,
    execution_p50: float = 0.0,
    execution_p95: float = 0.0,
):
    """Patch the registry-walking helpers so tests don't depend on real metrics."""

    def _sum(name: str) -> int:
        if name == "ct_trade_executed_total":
            return trades_total
        if name == "ct_risk_rejected_total":
            return risk_rejections
        if name == "ct_debate_skipped_total":
            return debate_skipped
        return 0

    def _hist(name: str, q: float, *, filter_labels=None) -> float:
        if name == "ct_pipeline_duration_ms":
            return pipeline_p50 if q == 0.5 else pipeline_p95
        if name == "ct_execution_latency_ms":
            return execution_p50 if q == 0.5 else execution_p95
        return 0.0

    return (
        patch("api.routes.metrics._sum_counter_samples", side_effect=_sum),
        patch("api.routes.metrics._histogram_quantile", side_effect=_hist),
    )


class TestMetricsSummaryV2:
    """GET /api/metrics/summary returns the contract shape."""

    def test_returns_contract_shape(self, client: TestClient) -> None:
        sum_patch, hist_patch = _patch_summary_helpers(
            trades_total=142,
            risk_rejections=12,
            debate_skipped=23,
            pipeline_p50=1250,
            pipeline_p95=4800,
            execution_p50=320,
            execution_p95=880,
        )
        with sum_patch, hist_patch:
            resp = client.get("/api/metrics/summary")

        assert resp.status_code == 200
        data = resp.json()
        # Spec: frontend-prototype-alignment (2026-04-24) extends MetricsSummary with
        # llm accounting + cost series + latency histogram. Contract is now superset.
        assert {"counters", "percentiles", "collected_at"}.issubset(set(data.keys()))
        assert {
            "llm_calls_24h",
            "llm_cost_24h",
            "cache_hit_rate",
            "decisions_per_day",
            "latency_histogram",
            "cost_14d",
        }.issubset(set(data.keys()))

        counters = data["counters"]
        assert counters["trades_total"] == 142
        assert counters["orders_placed"] == 142
        assert counters["orders_failed"] == 0  # Not yet tracked
        assert counters["risk_rejections"] == 12
        assert counters["debate_skipped_total"] == 23

        pct = data["percentiles"]
        assert pct["pipeline_p50_ms"] == 1250
        assert pct["pipeline_p95_ms"] == 4800
        assert pct["execution_p50_ms"] == 320
        assert pct["execution_p95_ms"] == 880

        # collected_at should be ISO-8601 UTC
        assert "T" in data["collected_at"]

    def test_zero_when_no_observations(self, client: TestClient) -> None:
        sum_patch, hist_patch = _patch_summary_helpers()
        with sum_patch, hist_patch:
            resp = client.get("/api/metrics/summary")

        assert resp.status_code == 200
        data = resp.json()
        assert data["counters"]["trades_total"] == 0
        assert data["percentiles"]["pipeline_p95_ms"] == 0
