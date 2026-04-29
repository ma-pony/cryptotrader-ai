"""Tests for GET /api/portfolio/snapshot — FR-800.

Returns the current portfolio Snapshot (equity, cash, positions, pnl_24h, drawdown).
Replaces the legacy GET /portfolio with a richer schema aligned to data-model §1.

Strategy: mock PortfolioManager + read_portfolio_from_exchange (live exchange call) —
we never hit a real exchange in unit tests.  503 path triggered by both helpers
returning falsy values.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cryptotrader._compat import UTC


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.infrastructure.database_url = None
    cfg.infrastructure.redis_url = None
    return cfg


class TestPortfolioSnapshotShape:
    def test_returns_200_and_required_fields(self, client: TestClient) -> None:
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(
            return_value={
                "cash": 2103.10,
                "positions": {
                    "BTC/USDT": {"amount": 0.05, "avg_price": 65000.0, "side": "long"},
                },
                "total_value": 5353.10,
            }
        )
        mock_pm.get_daily_pnl = AsyncMock(return_value=215.80)
        mock_pm.get_drawdown = AsyncMock(return_value=-0.012)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        for key in ("equity", "cash", "positions", "pnl_24h", "pnl_24h_pct", "drawdown", "updated_at"):
            assert key in body, f"missing key: {key}"

    def test_drawdown_is_absolute_value_in_unit_interval(self, client: TestClient) -> None:
        """drawdown ∈ [0, 1] per data-model §1 (PortfolioManager returns negative)."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"cash": 1000.0, "positions": {}, "total_value": 1000.0})
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=-0.025)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        body = resp.json()
        assert body["drawdown"] == pytest.approx(0.025)
        assert 0 <= body["drawdown"] <= 1

    def test_positions_have_unrealized_pnl_fields(self, client: TestClient) -> None:
        """Each position must include unrealized_pnl + side per data-model §1."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(
            return_value={
                "cash": 0.0,
                "positions": {
                    "ETH/USDT": {"amount": 2.0, "avg_price": 3000.0, "side": "long"},
                },
                "total_value": 6000.0,
            }
        )
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.0)

        # Exchange overrides with current price for unrealized PnL math
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(
                    return_value={
                        "cash": 0.0,
                        "positions": {
                            "ETH/USDT": {
                                "amount": 2.0,
                                "avg_price": 3000.0,
                                "side": "long",
                                "current_price": 3100.0,
                                "unrealized_pnl": 200.0,
                            },
                        },
                        "total_value": 6200.0,
                    }
                ),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        body = resp.json()
        assert len(body["positions"]) == 1
        pos = body["positions"][0]
        assert pos["pair"] == "ETH/USDT"
        assert pos["side"] == "long"
        assert pos["size"] == pytest.approx(2.0)
        assert pos["avg_price"] == pytest.approx(3000.0)
        assert "unrealized_pnl" in pos
        assert "unrealized_pnl_pct" in pos
        assert "opened_at" in pos

    def test_empty_positions_returns_empty_list(self, client: TestClient) -> None:
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"cash": 10000.0, "positions": {}, "total_value": 10000.0})
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.0)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        assert resp.status_code == 200
        assert resp.json()["positions"] == []
        assert resp.json()["equity"] == pytest.approx(10000.0)

    def test_updated_at_is_iso8601(self, client: TestClient) -> None:
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(return_value={"cash": 100.0, "positions": {}, "total_value": 100.0})
        mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
        mock_pm.get_drawdown = AsyncMock(return_value=0.0)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        ts = resp.json()["updated_at"]
        # Must parse cleanly (allow trailing Z)
        datetime.fromisoformat(ts.replace("Z", "+00:00"))


class TestPortfolioSnapshotErrors:
    def test_503_when_portfolio_manager_raises(self, client: TestClient) -> None:
        """Both PortfolioManager and exchange unavailable → 503 with error envelope."""
        mock_pm = MagicMock()
        mock_pm.get_portfolio = AsyncMock(side_effect=RuntimeError("DB connection refused"))

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
            patch(
                "cryptotrader.portfolio.manager.read_portfolio_from_exchange",
                AsyncMock(return_value=None),
            ),
        ):
            resp = client.get("/api/portfolio/snapshot")

        assert resp.status_code == 503
        body = resp.json()
        assert "detail" in body


def test_24h_pnl_pct_is_proportional() -> None:
    """pnl_24h_pct == pnl_24h / (equity - pnl_24h) — guards against div-by-zero."""
    from api.routes.portfolio_v2 import _compute_pnl_pct

    # Standard case
    assert _compute_pnl_pct(equity=10215.80, pnl_24h=215.80) == pytest.approx(0.0216, rel=1e-2)
    # Zero baseline → 0
    assert _compute_pnl_pct(equity=0.0, pnl_24h=0.0) == 0.0
    # Negative baseline guard
    assert _compute_pnl_pct(equity=100.0, pnl_24h=200.0) == 0.0


# Avoid unused-import lint
_ = UTC
