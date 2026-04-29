"""Tests for GET /api/scheduler/status (contract shape, FR-802).

The legacy ``/scheduler/status`` returns an APScheduler-flavored payload
(``running``/``jobs``/``cycle_count``/…). The contract endpoint at
``/api/scheduler/status`` returns the data-model shape consumed by the
React Dashboard:

```
{
  "enabled": true,
  "next_pair": "BTC/USDT",
  "next_run_at": "2026-04-16T13:35:00Z",
  "redis_available": true
}
```
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from cryptotrader._compat import UTC


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _mock_config(scheduler_enabled: bool = True, pairs: list[str] | None = None):
    cfg = MagicMock()
    cfg.scheduler.enabled = scheduler_enabled
    cfg.scheduler.pairs = pairs if pairs is not None else ["BTC/USDT", "ETH/USDT"]
    cfg.scheduler.interval_minutes = 240
    cfg.infrastructure.redis_url = "redis://localhost:6379"
    return cfg


class TestSchedulerStatusV2:
    """GET /api/scheduler/status returns the contract-shape payload."""

    def test_returns_contract_shape(self, client: TestClient) -> None:
        next_run = datetime(2026, 4, 16, 13, 35, 0, tzinfo=UTC)
        mock_sched = MagicMock()
        mock_sched.pairs = ["BTC/USDT", "ETH/USDT"]
        mock_sched.jobs = [
            {
                "id": "trading_cycle",
                "name": "Trading cycle",
                "next_run_time": next_run.isoformat(),
            }
        ]
        # APScheduler-running probe — endpoint shouldn't depend on it
        mock_sched._scheduler.running = True

        mock_rsm = MagicMock()
        mock_rsm.available = True

        # Pre-stash the scheduler on app.state so the endpoint can read it
        app.state.scheduler = mock_sched
        try:
            with (
                patch("cryptotrader.config.load_config", return_value=_mock_config()),
                patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
            ):
                resp = client.get("/api/scheduler/status")
        finally:
            del app.state.scheduler

        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"enabled", "next_pair", "next_run_at", "redis_available"}
        assert data["enabled"] is True
        assert data["next_pair"] == "BTC/USDT"
        assert data["next_run_at"].startswith("2026-04-16T13:35:00")
        assert data["redis_available"] is True

    def test_disabled_when_config_disabled(self, client: TestClient) -> None:
        with (
            patch(
                "cryptotrader.config.load_config",
                return_value=_mock_config(scheduler_enabled=False),
            ),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=MagicMock(available=True)),
        ):
            resp = client.get("/api/scheduler/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        # next_pair / next_run_at may be null when scheduler not registered
        assert data["next_pair"] is None
        assert data["next_run_at"] is None

    def test_redis_unavailable_reports_false(self, client: TestClient) -> None:
        mock_rsm = MagicMock()
        mock_rsm.available = False
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        ):
            resp = client.get("/api/scheduler/status")

        assert resp.status_code == 200
        assert resp.json()["redis_available"] is False

    def test_no_scheduler_registered_returns_null_fields(self, client: TestClient) -> None:
        """API-only deployment: app.state.scheduler is not set → next_* are null."""
        # Ensure no leftover state
        if hasattr(app.state, "scheduler"):
            del app.state.scheduler

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.risk.state.RedisStateManager", return_value=MagicMock(available=True)),
        ):
            resp = client.get("/api/scheduler/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True  # Config says enabled, just not running here
        assert data["next_pair"] is None
        assert data["next_run_at"] is None
