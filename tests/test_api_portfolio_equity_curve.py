"""Tests for GET /api/portfolio/equity-curve — FR-801.

Returns historical equity points for a given range. Supports range param
24h | 7d | 30d | all (data-model §1).  400 on invalid range.

Backed by PortfolioManager._load_snapshots → equity-curve transformation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
    return cfg


def _make_snapshots(n: int, hours_apart: int = 1) -> list[dict]:
    """Build a list of synthetic snapshots descending in time."""
    now = datetime.now(UTC)
    return [
        {
            "account_id": "default",
            "total_value": 10000 + i * 5,
            "cash": 1000.0,
            "timestamp": now - timedelta(hours=(n - 1 - i) * hours_apart),
        }
        for i in range(n)
    ]


class TestEquityCurveShape:
    @pytest.mark.parametrize("rng", ["24h", "7d", "30d", "all"])
    def test_valid_range_returns_200(self, client: TestClient, rng: str) -> None:
        mock_pm = MagicMock()
        mock_pm._load_snapshots = AsyncMock(return_value=_make_snapshots(5))

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get(f"/api/portfolio/equity-curve?range={rng}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["range"] == rng
        assert isinstance(body["points"], list)

    def test_each_point_has_ts_and_equity(self, client: TestClient) -> None:
        mock_pm = MagicMock()
        mock_pm._load_snapshots = AsyncMock(return_value=_make_snapshots(3))

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/api/portfolio/equity-curve?range=24h")

        body = resp.json()
        assert len(body["points"]) == 3
        for pt in body["points"]:
            assert "ts" in pt
            assert "equity" in pt
            datetime.fromisoformat(pt["ts"].replace("Z", "+00:00"))

    def test_24h_range_filters_by_window(self, client: TestClient) -> None:
        """24h range only returns snapshots within last 24 hours."""
        snaps = _make_snapshots(50, hours_apart=1)  # 50 hours of data
        mock_pm = MagicMock()
        mock_pm._load_snapshots = AsyncMock(return_value=snaps)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/api/portfolio/equity-curve?range=24h")

        body = resp.json()
        # Should not exceed 25 (24 hours of hourly samples)
        assert len(body["points"]) <= 25

    def test_points_capped_at_1000(self, client: TestClient) -> None:
        """data-model §1 — at most 1000 points to honour NFR-P-004."""
        snaps = _make_snapshots(2000, hours_apart=1)
        mock_pm = MagicMock()
        mock_pm._load_snapshots = AsyncMock(return_value=snaps)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/api/portfolio/equity-curve?range=all")

        body = resp.json()
        assert len(body["points"]) <= 1000

    def test_empty_snapshots_returns_empty_points(self, client: TestClient) -> None:
        mock_pm = MagicMock()
        mock_pm._load_snapshots = AsyncMock(return_value=[])

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        ):
            resp = client.get("/api/portfolio/equity-curve?range=7d")

        assert resp.status_code == 200
        assert resp.json()["points"] == []


class TestEquityCurveErrors:
    @pytest.mark.parametrize("rng", ["", "1h", "12h", "abc", "all_time", "1y"])
    def test_invalid_range_returns_400_or_422(self, client: TestClient, rng: str) -> None:
        """Invalid range value rejected with 4xx (400 explicit or 422 from FastAPI Literal)."""
        resp = client.get(f"/api/portfolio/equity-curve?range={rng}")
        assert resp.status_code in (400, 422)

    def test_missing_range_returns_4xx(self, client: TestClient) -> None:
        """range query is required — missing returns 422 from FastAPI."""
        resp = client.get("/api/portfolio/equity-curve")
        assert resp.status_code == 422
