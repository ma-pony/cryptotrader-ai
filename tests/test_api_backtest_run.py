"""Tests for POST /api/backtest/run — FR-805.

Schedule a backtest as a background task; respond 202 with `run_id`.
Param validation rejects invalid dates, capital, and retired strategy selectors.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    previous = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(
        repository=object(),
        snapshot=object(),
        signal_registry=object(),
    )
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.state.runtime = previous


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
        "session_name": "q1-profile-revision-3",
    }


class TestBacktestRunHappyPath:
    def test_returns_202_with_run_id(self, client: TestClient, monkeypatch) -> None:
        from api.main import app

        shared_repository = object()
        shared_snapshot = object()
        shared_signal_registry = object()
        monkeypatch.setattr(
            app.state,
            "runtime",
            SimpleNamespace(
                repository=shared_repository,
                snapshot=shared_snapshot,
                signal_registry=shared_signal_registry,
            ),
        )
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._spawn_run", return_value="run_a1b2c3") as spawn_run,
        ):
            resp = client.post("/api/backtest/run", json=_valid_payload())

        assert resp.status_code == 202
        assert spawn_run.call_args.args[1] is shared_repository
        assert spawn_run.call_args.args[2] is shared_snapshot
        assert spawn_run.call_args.args[3] is shared_signal_registry
        from cryptotrader.journal.store import MultiVenueCycleStore

        assert isinstance(spawn_run.call_args.args[4], MultiVenueCycleStore)
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

    def test_legacy_mode_selector_is_rejected(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["mode"] = "llm"
        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("api.routes.backtest._spawn_run", return_value="run_unused"),
        ):
            resp = client.post("/api/backtest/run", json=payload)
        assert resp.status_code == 422


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


@pytest.mark.asyncio
async def test_mounted_backtest_route_completes_with_runtime_dependencies(monkeypatch) -> None:
    import httpx

    from api.main import app
    from api.routes.backtest import _RUNS, _TASKS
    from cryptotrader.backtest.result import BacktestResult

    repository = object()
    snapshot = object()
    signal_registry = object()
    previous = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(
        repository=repository,
        snapshot=snapshot,
        signal_registry=signal_registry,
    )
    captured = {}

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            captured.update(kwargs)

        async def run(self):
            return BacktestResult(equity_curve=[10_000.0])

    monkeypatch.setattr("cryptotrader.backtest.engine.BacktestEngine", FakeEngine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/backtest/run", json=_valid_payload())
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            await _TASKS[run_id]
            status = await client.get(f"/api/backtest/runs/{run_id}")
    finally:
        app.state.runtime = previous

    assert status.json()["status"] == "completed"
    assert captured["repository"] is repository
    assert captured["snapshot"] is snapshot
    assert captured["signal_registry"] is signal_registry
    assert captured["journal_store"].database_url is None
    assert _RUNS[run_id]["error"] is None
