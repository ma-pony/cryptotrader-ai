"""Tests for GET /api/risk/status + POST /api/risk/circuit-breaker/reset — FR-807 + FR-404.

Risk status response follows data-model §4 RiskStatus shape. Reset returns
confirmation, 409 when already inactive, 503 when Redis unavailable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.infrastructure.database_url = None
    cfg.infrastructure.redis_url = "redis://localhost:6379"
    # RiskConfig thresholds — names match the real ``cryptotrader.config`` dataclasses
    cfg.risk.position.max_single_pct = 0.30
    cfg.risk.loss.max_daily_loss_pct = 0.05
    cfg.risk.max_stop_loss_pct = 0.02
    cfg.risk.rate_limit.max_trades_per_hour = 10
    cfg.risk.rate_limit.max_trades_per_day = 50
    cfg.risk.cooldown.post_loss_minutes = 30  # 30 * 60 = 1800s
    return cfg


class TestRiskStatusShape:
    def test_returns_full_risk_status(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = True
        mock_rs.ping = AsyncMock(return_value=True)
        mock_rs.get_trade_counts = AsyncMock(return_value=(3, 12))
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=True)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            resp = client.get("/api/risk/status")

        assert resp.status_code == 200
        body = resp.json()
        for key in (
            "trade_count_hour",
            "trade_count_day",
            "circuit_breaker",
            "thresholds",
            "redis_available",
        ):
            assert key in body

        assert body["trade_count_hour"] == 3
        assert body["trade_count_day"] == 12
        assert body["redis_available"] is True
        assert body["circuit_breaker"]["state"] == "active"

    def test_inactive_circuit_breaker_shape(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = True
        mock_rs.ping = AsyncMock(return_value=True)
        mock_rs.get_trade_counts = AsyncMock(return_value=(0, 0))
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=False)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            body = client.get("/api/risk/status").json()

        assert body["circuit_breaker"]["state"] == "inactive"

    def test_thresholds_exposed(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = True
        mock_rs.ping = AsyncMock(return_value=True)
        mock_rs.get_trade_counts = AsyncMock(return_value=(0, 0))
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=False)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            body = client.get("/api/risk/status").json()

        t = body["thresholds"]
        for key in (
            "max_position_pct",
            "max_daily_loss_pct",
            "max_stop_loss_pct",
            "max_trades_per_hour",
            "max_trades_per_day",
            "post_loss_cooldown_seconds",
        ):
            assert key in t

    def test_redis_unavailable_returns_200_with_null_counts(self, client: TestClient) -> None:
        """per contract §5 — 200 with redis_available=false, counts null."""
        mock_rs = MagicMock()
        mock_rs.available = False
        mock_rs.ping = AsyncMock(return_value=False)
        mock_rs.get_trade_counts = AsyncMock(return_value=(0, 0))
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=False)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            resp = client.get("/api/risk/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["redis_available"] is False
        assert body["trade_count_hour"] is None
        assert body["trade_count_day"] is None


class TestCircuitBreakerReset:
    def test_reset_active_returns_success(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = True
        mock_rs.ping = AsyncMock(return_value=True)
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=True)
        mock_rs.reset_circuit_breaker = AsyncMock(return_value=None)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            resp = client.post("/api/risk/circuit-breaker/reset")

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "message" in body

    def test_409_when_already_inactive(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = True
        mock_rs.ping = AsyncMock(return_value=True)
        mock_rs.is_circuit_breaker_active = AsyncMock(return_value=False)
        mock_rs.reset_circuit_breaker = AsyncMock(return_value=None)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            resp = client.post("/api/risk/circuit-breaker/reset")

        assert resp.status_code == 409

    def test_503_when_redis_unavailable(self, client: TestClient) -> None:
        mock_rs = MagicMock()
        mock_rs.available = False
        mock_rs.ping = AsyncMock(return_value=False)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rs),
        ):
            resp = client.post("/api/risk/circuit-breaker/reset")

        assert resp.status_code == 503
