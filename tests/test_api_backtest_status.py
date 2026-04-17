"""Tests for GET/DELETE /api/backtest/runs/{run_id} — FR-805 + FR-302.

Status polling returns BacktestRun with progress + (when complete) result.
DELETE cancels in-flight run; 409 if already terminated.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _running_run(run_id: str = "run_a1b2c3") -> dict:
    return {
        "run_id": run_id,
        "params": {
            "start": "2026-01-01",
            "end": "2026-04-01",
            "pair": "BTC/USDT",
            "initial_capital": 10000,
            "mode": "rules",
        },
        "status": "running",
        "progress": 0.42,
        "started_at": "2026-04-16T13:00:00Z",
    }


def _completed_run(run_id: str = "run_done") -> dict:
    return {
        "run_id": run_id,
        "params": {
            "start": "2026-01-01",
            "end": "2026-04-01",
            "pair": "BTC/USDT",
            "initial_capital": 10000,
            "mode": "rules",
        },
        "status": "completed",
        "progress": 1.0,
        "started_at": "2026-04-16T13:00:00Z",
        "finished_at": "2026-04-16T13:08:42Z",
        "result": {
            "metrics": {
                "total_return_pct": 0.085,
                "sharpe": 1.42,
                "max_drawdown_pct": 0.12,
                "win_rate": 0.61,
                "trades_count": 38,
            },
            "equity_curve": [{"ts": "2026-01-01T00:00:00Z", "equity": 10000.0}],
            "decisions": [],
        },
    }


class TestBacktestStatus:
    def test_running_returns_progress(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._get_run", return_value=_running_run()),
        ):
            resp = client.get("/api/backtest/runs/run_a1b2c3")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run_a1b2c3"
        assert body["status"] == "running"
        assert 0 <= body["progress"] <= 1
        assert "result" not in body or body["result"] is None

    def test_completed_returns_result_block(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._get_run", return_value=_completed_run()),
        ):
            resp = client.get("/api/backtest/runs/run_done")

        body = resp.json()
        assert body["status"] == "completed"
        assert body["progress"] == 1.0
        assert "finished_at" in body
        result = body["result"]
        for k in ("metrics", "equity_curve", "decisions"):
            assert k in result
        for m in ("total_return_pct", "sharpe", "max_drawdown_pct", "win_rate", "trades_count"):
            assert m in result["metrics"]

    def test_404_when_run_unknown(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._get_run", return_value=None),
        ):
            resp = client.get("/api/backtest/runs/run_does_not_exist")
        assert resp.status_code == 404


class TestBacktestCancel:
    def test_cancel_running_returns_200(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._cancel_run", return_value=True),
        ):
            resp = client.delete("/api/backtest/runs/run_a1b2c3")
        assert resp.status_code == 200
        assert resp.json() == {"canceled": True}

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "canceled"])
    def test_409_when_already_terminated(self, client: TestClient, terminal_status: str) -> None:
        terminated = _completed_run("run_xyz")
        terminated["status"] = terminal_status

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._get_run", return_value=terminated),
            patch("api.routes.backtest._cancel_run", return_value=False),
        ):
            resp = client.delete("/api/backtest/runs/run_xyz")
        assert resp.status_code == 409

    def test_404_when_cancel_unknown(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._get_run", return_value=None),
            patch("api.routes.backtest._cancel_run", return_value=False),
        ):
            resp = client.delete("/api/backtest/runs/run_unknown")
        assert resp.status_code == 404
