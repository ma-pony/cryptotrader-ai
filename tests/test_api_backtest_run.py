"""Tests for POST /api/backtest/run — FR-805.

Schedule a backtest as a background task; respond 202 with `run_id`.
Param validation rejects invalid dates, capital, and retired strategy selectors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType, SimpleNamespace
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
        assert spawn_run.call_args.args[1] is shared_snapshot
        assert spawn_run.call_args.args[2] is shared_signal_registry
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
    def test_422_when_session_name_is_not_a_safe_identifier(self, client: TestClient) -> None:
        payload = _valid_payload()
        payload["session_name"] = "../outside"
        with patch("api.routes.backtest._spawn_run", return_value="run_unused"):
            response = client.post("/api/backtest/run", json=payload)
        assert response.status_code == 422

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
    from cryptotrader.backtest.cache import _TF_MS
    from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink
    from cryptotrader.runtime import Runtime
    from cryptotrader.runtime_config.models import (
        MarketDataConfig,
        RuntimeConfigSnapshot,
        SignalComponentConfig,
        SignalConfig,
        SystemConfig,
    )
    from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.runtime_config import runtime_document

    class Component:
        id = "fixture"
        display_name = "Fixture"
        description = "deterministic API backtest signal"

        @staticmethod
        def requirements():
            return DataRequirements(candles=(CandleRequirement("1h", 20),))

        async def evaluate(self, context):
            return ComponentSignal(
                self.id,
                "long",
                1.0,
                "mounted route evidence",
                MappingProxyType({"evidence": MappingProxyType({"source": "mounted-real-engine"})}),
            )

    document = runtime_document(
        system=SystemConfig(active=True),
        market_data=MarketDataConfig(
            source_id="default",
            parameters={"timeframe": "1h", "limit": 20},
        ),
        signals=SignalConfig(
            components=(SignalComponentConfig(component_id="fixture", enabled=True, weight=1.0),),
            neutral_threshold=0.2,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
        ),
    )
    snapshot = RuntimeConfigSnapshot(4, document, datetime.now(UTC))

    class Repository:
        database_url = None

        async def get_or_create(self):
            return snapshot

        async def reveal_credentials(self, credential_ref):
            raise AssertionError("backtest must not reveal venue credentials")

    repository = Repository()
    signal_registry = SignalComponentRegistry((Component(),))
    runtime = Runtime(
        snapshot=snapshot,
        repository=repository,
        cycle=None,
        sessions={},
        signal_registry=signal_registry,
        market_registry=object(),
        venue_registry=object(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    previous = getattr(app.state, "runtime", None)
    app.state.runtime = runtime

    async def load_historical(self, requirements):
        for requirement in requirements.candles:
            interval_ms = _TF_MS[requirement.timeframe]
            start_ms = self.start_ms - requirement.limit * interval_ms
            self._candles_by_timeframe[requirement.timeframe] = [
                [
                    start_ms + index * interval_ms,
                    100.0 + index,
                    102.0 + index,
                    99.0 + index,
                    101.0 + index,
                    10.0,
                ]
                for index in range(requirement.limit + 3)
            ]
        self._candles = self._candles_by_timeframe[self.interval]

    monkeypatch.setattr(
        "cryptotrader.backtest.engine.BacktestEngine._fetch_historical_data",
        load_historical,
    )
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = _valid_payload()
            payload.pop("session_name")
            response = await client.post("/api/backtest/run", json=payload)
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            await _TASKS[run_id]
            status = await client.get(f"/api/backtest/runs/{run_id}")
    finally:
        app.state.runtime = previous

    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "completed"
    decision = body["result"]["decisions"][0]
    assert decision["config_revision"] == 4
    assert decision["components"][0]["details"]["evidence"] == {"source": "mounted-real-engine"}
    assert decision["books"][0]["book_id"] == "backtest"
    assert _RUNS[run_id]["error"] is None
