"""spec 020a T020 — test_metrics_endpoint_cache.py

Tests: /metrics endpoint lazy-updates LLM_CACHE_HIT_RATE_GAUGE and
IVE_CLASSIFY_FAILURE_RATE_GAUGE from in-process aggregators.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _make_app() -> TestClient:
    from fastapi import FastAPI

    from api.routes.metrics import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPrometheusMetricsGaugeUpdate:
    def test_cache_gauge_updated_on_metrics_request(self):
        """LLM_CACHE_HIT_RATE_GAUGE is set from CacheMetricsAggregator on /metrics call."""
        mock_cache_agg = MagicMock()
        mock_cache_agg.average.return_value = 0.42
        mock_ive_agg = MagicMock()
        mock_ive_agg.failure_rate.return_value = 0.05

        # Patch at source module (lazy-imported inside endpoint function)
        with (
            patch(
                "cryptotrader.observability.cache_metrics.get_cache_metrics_aggregator",
                return_value=mock_cache_agg,
            ),
            patch(
                "cryptotrader.observability.ive_metrics.get_ive_metrics_aggregator",
                return_value=mock_ive_agg,
            ),
        ):
            client = _make_app()
            resp = client.get("/metrics")
            assert resp.status_code == 200

            mock_cache_agg.average.assert_called()
            mock_ive_agg.failure_rate.assert_called()

    def test_ive_gauge_updated_on_metrics_request(self):
        """IVE_CLASSIFY_FAILURE_RATE_GAUGE is set from IveMetricsAggregator."""
        mock_cache_agg = MagicMock()
        mock_cache_agg.average.return_value = 0.0
        mock_ive_agg = MagicMock()
        mock_ive_agg.failure_rate.return_value = 0.20

        with (
            patch(
                "cryptotrader.observability.cache_metrics.get_cache_metrics_aggregator",
                return_value=mock_cache_agg,
            ),
            patch(
                "cryptotrader.observability.ive_metrics.get_ive_metrics_aggregator",
                return_value=mock_ive_agg,
            ),
        ):
            client = _make_app()
            resp = client.get("/metrics")
            assert resp.status_code == 200
            # gauge names must appear in Prometheus text output
            body = resp.text
            assert "llm_cache_hit_rate_24h_avg" in body
            assert "ive_classify_failure_rate_1h_avg" in body

    def test_aggregator_exception_does_not_break_metrics(self):
        """If aggregator raises, /metrics still returns 200 (non-blocking)."""
        with patch(
            "cryptotrader.observability.cache_metrics.get_cache_metrics_aggregator",
            side_effect=RuntimeError("agg unavailable"),
        ):
            client = _make_app()
            resp = client.get("/metrics")
            assert resp.status_code == 200

    def test_gauge_names_in_prometheus_output(self):
        """Both gauge metric names appear in raw Prometheus text output."""
        client = _make_app()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "llm_cache_hit_rate_24h_avg" in body
        assert "ive_classify_failure_rate_1h_avg" in body


class TestIveMetricsAggregator:
    def test_record_success(self):
        from cryptotrader.observability.ive_metrics import IveMetricsAggregator

        agg = IveMetricsAggregator()
        agg.record(success=True)
        agg.record(success=True)
        assert agg.failure_rate() == 0.0

    def test_record_failure(self):
        from cryptotrader.observability.ive_metrics import IveMetricsAggregator

        agg = IveMetricsAggregator()
        agg.record(success=False)
        agg.record(success=True)
        assert abs(agg.failure_rate() - 0.5) < 1e-9

    def test_empty_returns_zero(self):
        from cryptotrader.observability.ive_metrics import IveMetricsAggregator

        agg = IveMetricsAggregator()
        assert agg.failure_rate() == 0.0

    def test_evicts_old_entries(self):
        import time

        from cryptotrader.observability.ive_metrics import IveMetricsAggregator

        agg = IveMetricsAggregator(window_seconds=1)
        agg.record(success=False)  # will be evicted
        time.sleep(1.1)
        agg.record(success=True)  # stays
        assert agg.failure_rate() == 0.0
