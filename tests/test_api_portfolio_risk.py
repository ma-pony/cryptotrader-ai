"""Tests for GET /portfolio and GET /risk/status API endpoints.

Uses FastAPI TestClient with mocked PortfolioManager and RedisStateManager
so that no actual database or Redis connection is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /portfolio
# ---------------------------------------------------------------------------


class TestPortfolioEndpoint:
    """GET /portfolio returns portfolio summary."""

    def _mock_config(self):
        cfg = MagicMock()
        cfg.infrastructure.database_url = None
        cfg.infrastructure.redis_url = None
        return cfg

    def test_portfolio_returns_200_with_mocked_data(self) -> None:
        """GET /portfolio returns 200 and a valid PortfolioOut structure."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(
            return_value={
                "total_value": 100000.0,
                "positions": {
                    "BTC/USDT": {"amount": 0.5, "avg_price": 50000.0},
                },
            }
        )
        mock_pm.get_daily_pnl = AsyncMock(return_value=500.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.02)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/portfolio")

        assert resp.status_code == 200
        data = resp.json()
        assert "total_value" in data
        assert "daily_pnl" in data
        assert "drawdown" in data
        assert "positions" in data

    def test_portfolio_returns_correct_total_value(self) -> None:
        """GET /portfolio total_value matches PortfolioManager output."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 75000.0, "positions": {}})
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.0)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/portfolio")

        assert resp.status_code == 200
        assert resp.json()["total_value"] == pytest.approx(75000.0)

    def test_portfolio_positions_list(self) -> None:
        """GET /portfolio positions includes pair, amount, avg_price, value."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(
            return_value={
                "total_value": 50000.0,
                "positions": {
                    "ETH/USDT": {"amount": 2.0, "avg_price": 3000.0},
                },
            }
        )
        mock_pm.get_daily_pnl = AsyncMock(return_value=100.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.01)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/portfolio")

        assert resp.status_code == 200
        positions = resp.json()["positions"]
        assert len(positions) == 1
        pos = positions[0]
        assert pos["pair"] == "ETH/USDT"
        assert pos["amount"] == pytest.approx(2.0)
        assert pos["avg_price"] == pytest.approx(3000.0)
        assert pos["value"] == pytest.approx(6000.0)  # 2.0 * 3000.0

    def test_portfolio_empty_positions(self) -> None:
        """GET /portfolio with no positions returns empty list."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 10000.0, "positions": {}})
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.0)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/portfolio")

        assert resp.status_code == 200
        assert resp.json()["positions"] == []

    def test_portfolio_returns_daily_pnl(self) -> None:
        """GET /portfolio includes daily_pnl from PortfolioManager."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 0.0, "positions": {}})
        mock_pm.get_daily_pnl = AsyncMock(return_value=-250.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.025)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/portfolio")

        assert resp.status_code == 200
        assert resp.json()["daily_pnl"] == pytest.approx(-250.0)
        assert resp.json()["drawdown"] == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# GET /risk/status
# ---------------------------------------------------------------------------


class TestRiskStatusEndpoint:
    """GET /risk/status returns risk state (trades, circuit breaker)."""

    def _mock_config(self, redis_url: str = "redis://localhost:6379"):
        cfg = MagicMock()
        cfg.infrastructure.redis_url = redis_url
        return cfg

    def test_risk_status_returns_200_when_redis_unavailable(self) -> None:
        """GET /risk/status returns 200 with zeros when Redis is unavailable."""
        mock_rsm = MagicMock()
        mock_rsm.available = False

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        ):
            resp = client.get("/risk/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["hourly_trades"] == 0
        assert data["daily_trades"] == 0
        assert data["circuit_breaker_active"] is False

    def test_risk_status_returns_counts_when_redis_available(self) -> None:
        """GET /risk/status returns actual trade counts when Redis is reachable."""
        mock_rsm = MagicMock()
        mock_rsm.available = True
        mock_rsm.get_trade_counts = AsyncMock(return_value=(3, 12))
        mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=False)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        ):
            resp = client.get("/risk/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["hourly_trades"] == 3
        assert data["daily_trades"] == 12
        assert data["circuit_breaker_active"] is False

    def test_risk_status_circuit_breaker_active(self) -> None:
        """GET /risk/status reflects circuit_breaker_active=True."""
        mock_rsm = MagicMock()
        mock_rsm.available = True
        mock_rsm.get_trade_counts = AsyncMock(return_value=(0, 0))
        mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=True)

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        ):
            resp = client.get("/risk/status")

        assert resp.status_code == 200
        assert resp.json()["circuit_breaker_active"] is True

    def test_risk_status_response_shape(self) -> None:
        """GET /risk/status response has exactly the required fields."""
        mock_rsm = MagicMock()
        mock_rsm.available = False

        with (
            patch("cryptotrader.config.load_config", return_value=self._mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        ):
            resp = client.get("/risk/status")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"hourly_trades", "daily_trades", "circuit_breaker_active"}
