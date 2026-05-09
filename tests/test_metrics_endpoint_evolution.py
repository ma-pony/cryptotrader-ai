"""spec 022 T026 -- /metrics endpoint returns 3 evolution daemon Prometheus Gauges.

Tests that the 3 new Gauges registered in metrics.py appear in Prometheus output
when Redis events are mocked.
"""

from __future__ import annotations

from unittest.mock import patch


def test_evolution_gauge_names_registered():
    """All 3 spec 022 Gauge names are registered in the Prometheus registry."""
    from prometheus_client import REGISTRY

    from api.routes.metrics import (
        EVOLUTION_DAEMON_LLM_FAILURE_RATE_GAUGE,
        EVOLUTION_DAEMON_RUN_COUNT_GAUGE,
        SKILL_PROPOSAL_DRAFT_COUNT_GAUGE,
    )

    registered_names = {m.name for m in REGISTRY.collect()}
    assert "evolution_daemon_run_count_24h" in registered_names
    assert "evolution_daemon_llm_failure_rate_24h" in registered_names
    assert "skill_proposal_draft_count_7d" in registered_names

    # Verify the objects are Gauge instances
    from prometheus_client import Gauge

    assert isinstance(EVOLUTION_DAEMON_RUN_COUNT_GAUGE, Gauge)
    assert isinstance(EVOLUTION_DAEMON_LLM_FAILURE_RATE_GAUGE, Gauge)
    assert isinstance(SKILL_PROPOSAL_DRAFT_COUNT_GAUGE, Gauge)


def test_prometheus_metrics_endpoint_contains_evolution_gauges():
    """GET /metrics text output contains all 3 evolution daemon gauge names."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)

    with (
        patch(
            "cryptotrader.observability.daemon_metrics.get_run_count_24h_from_redis",
            return_value=3.0,
        ),
        patch(
            "cryptotrader.observability.daemon_metrics.get_llm_failure_rate_24h_from_redis",
            return_value=0.25,
        ),
        patch(
            "cryptotrader.observability.daemon_metrics.get_draft_count_7d_from_redis",
            return_value=7.0,
        ),
    ):
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "evolution_daemon_run_count_24h" in body
    assert "evolution_daemon_llm_failure_rate_24h" in body
    assert "skill_proposal_draft_count_7d" in body


def test_prometheus_metrics_gauge_values_from_redis():
    """Gauge values in /metrics output reflect mocked Redis data."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)

    with (
        patch(
            "cryptotrader.observability.daemon_metrics.get_run_count_24h_from_redis",
            return_value=5.0,
        ),
        patch(
            "cryptotrader.observability.daemon_metrics.get_llm_failure_rate_24h_from_redis",
            return_value=0.5,
        ),
        patch(
            "cryptotrader.observability.daemon_metrics.get_draft_count_7d_from_redis",
            return_value=2.0,
        ),
    ):
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "evolution_daemon_run_count_24h 5.0" in body
    assert "evolution_daemon_llm_failure_rate_24h 0.5" in body
    assert "skill_proposal_draft_count_7d 2.0" in body


def test_prometheus_metrics_degrades_gracefully_when_redis_unavailable():
    """If Redis is unavailable, /metrics still returns 200 (gauges default to 0.0)."""
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)

    with patch(
        "cryptotrader.observability.daemon_metrics._get_redis",
        return_value=None,
    ):
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    # Gauges default to 0.0 when Redis unavailable
    assert "evolution_daemon_run_count_24h" in body
    assert "skill_proposal_draft_count_7d" in body
