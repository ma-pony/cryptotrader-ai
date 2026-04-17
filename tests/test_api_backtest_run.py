"""Tests for POST /api/backtest/run — FR-805.

Schedule a backtest as a background task; respond 202 with `run_id`.
Param validation rejects (date order, capital floor, mode literal) with 400/422.
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


def _valid_payload() -> dict:
    return {
        "start": "2026-01-01",
        "end": "2026-04-01",
        "pair": "BTC/USDT",
        "initial_capital": 10000,
        "mode": "rules",
        "session_name": "q1-rules-baseline",
    }


class TestBacktestRunHappyPath:
    def test_returns_202_with_run_id(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._spawn_run", return_value="run_a1b2c3"),
        ):
            resp = client.post("/api/backtest/run", json=_valid_payload())

        assert resp.status_code == 202
        body = resp.json()
        assert "run_id" in body
        assert body["run_id"].startswith("run_")

    def test_session_name_is_optional(self, client: TestClient) -> None:
        payload = _valid_payload()
        del payload["session_name"]
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._spawn_run", return_value="run_xyz"),
        ):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code == 202

    def test_llm_mode_accepted(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["mode"] = "llm"
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._spawn_run", return_value="run_llm1"),
        ):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code == 202


class TestBacktestRunValidation:
    def test_400_when_start_after_end(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["start"] = "2026-05-01"
        payload["end"] = "2026-04-01"
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (400, 422)

    def test_400_when_capital_below_minimum(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["initial_capital"] = 50  # < 100 floor
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (400, 422)

    def test_422_when_mode_not_literal(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["mode"] = "machine_learning"
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (400, 422)

    def test_422_when_required_field_missing(self, client: TestClient) -> None:
        payload = _valid_payload()
        del payload["pair"]
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code == 422

    def test_400_when_end_in_future(self, client: TestClient) -> None:
        """data-model §3 — end ≤ today."""
        payload = _valid_payload()
        payload["start"] = "2026-01-01"
        payload["end"] = "2199-12-31"
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (400, 422)

    @pytest.mark.parametrize("bad_date", ["not-a-date", "2026/01/01", "01-01-2026"])
    def test_422_on_malformed_date(self, client: TestClient, bad_date: str) -> None:
        payload = _valid_payload()
        payload["start"] = bad_date
        with patch("cryptotrader.config.load_config", return_value=_mock_config()):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code in (400, 422)
