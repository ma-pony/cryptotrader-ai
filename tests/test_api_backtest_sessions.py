"""Tests for GET /api/backtest/sessions and /api/backtest/sessions/{name} — FR-806.

Surface saved backtest sessions for the historical-runs dropdown.
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


class TestSessionsList:
    def test_returns_session_names(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch(
                "cryptotrader.backtest.session.list_sessions",
                return_value=["q1-rules-baseline", "q2-llm-aggressive"],
            ),
        ):
            resp = client.get("/api/backtest/sessions")

        assert resp.status_code == 200
        body = resp.json()
        assert "sessions" in body
        assert body["sessions"] == ["q1-rules-baseline", "q2-llm-aggressive"]

    def test_empty_when_no_sessions(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.backtest.session.list_sessions", return_value=[]),
        ):
            resp = client.get("/api/backtest/sessions")

        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}


class TestSessionDetail:
    def test_returns_session_with_params_and_result(self, client: TestClient) -> None:
        loaded = {
            "name": "q1-rules-baseline",
            "params": {
                "start": "2026-01-01",
                "end": "2026-04-01",
                "pair": "BTC/USDT",
                "initial_capital": 10000,
                "mode": "rules",
                "session_name": "q1-rules-baseline",
            },
            "result": {
                "metrics": {
                    "total_return_pct": 0.085,
                    "sharpe": 1.42,
                    "max_drawdown_pct": 0.12,
                    "win_rate": 0.61,
                    "trades_count": 38,
                },
                "equity_curve": [],
                "decisions": [],
            },
            "saved_at": "2026-04-16T13:08:42Z",
        }
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._load_session", return_value=loaded),
        ):
            resp = client.get("/api/backtest/sessions/q1-rules-baseline")

        assert resp.status_code == 200
        body = resp.json()
        for key in ("name", "params", "result", "saved_at"):
            assert key in body
        assert body["name"] == "q1-rules-baseline"

    def test_404_when_session_unknown(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._load_session", return_value=None),
        ):
            resp = client.get("/api/backtest/sessions/never-existed")
        assert resp.status_code == 404
