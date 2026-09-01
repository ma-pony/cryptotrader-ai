"""Persisted run progress, cancellation and startup recovery through HTTP."""

import asyncio

import pytest

from tests.factories.research import research as research
from tests.factories.research import research_payload
from tests.factories.research_offline import research_offline  # noqa: F401


@pytest.mark.asyncio
async def test_running_progress_and_cancel_are_readable_after_reentry(research):
    client, service = research
    reached = asyncio.Event()

    class WaitingEngine:
        def __init__(self, **kwargs):
            self.progress = kwargs["progress_callback"]

        async def run(self):
            await self.progress(0.42)
            reached.set()
            await asyncio.Event().wait()

    service.engine_factory = WaitingEngine
    run_id = (await client.post("/api/backtest/runs", json=research_payload())).json()["run_id"]
    await asyncio.wait_for(reached.wait(), 3)
    response = await client.get(f"/api/backtest/runs/{run_id}")
    assert response.json()["progress"] == 0.42
    assert (await client.delete(f"/api/backtest/runs/{run_id}")).json() == {"canceled": True}
    assert (await client.get(f"/api/backtest/runs/{run_id}")).json()["status"] == "canceled"
    assert (await client.delete(f"/api/backtest/runs/{run_id}")).status_code == 409


@pytest.mark.asyncio
async def test_failed_exception_body_is_never_exposed(research):
    client, service = research

    class FailedEngine:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            raise RuntimeError("super-secret-key?token=never-show")

    service.engine_factory = FailedEngine
    run_id = (await client.post("/api/backtest/runs", json=research_payload())).json()["run_id"]
    await service.task_manager.drain()
    response = await client.get(f"/api/backtest/runs/{run_id}")
    assert response.json()["status"] == "failed"
    assert "RuntimeError" in response.json()["error"]
    assert "super-secret" not in response.text


@pytest.mark.asyncio
async def test_startup_interrupted_is_distinct_and_terminal(research):
    client, service = research
    from cryptotrader.backtest.models import BacktestParams

    run_id = await service.store.create(BacktestParams(**research_payload()), None)
    await service.store.update(run_id, "running", 0.7)
    await service.store.recover_interrupted()
    body = (await client.get(f"/api/backtest/runs/{run_id}")).json()
    assert body["status"] == "interrupted"
    assert body["progress"] == 0.7
    assert (await client.delete(f"/api/backtest/runs/{run_id}")).status_code == 409


@pytest.mark.asyncio
async def test_missing_read_cancel_and_compare_are_404(research):
    client, _ = research
    assert (await client.get("/api/backtest/runs/missing")).status_code == 404
    assert (await client.delete("/api/backtest/runs/missing")).status_code == 404
    assert (await client.get("/api/backtest/runs/compare?left=missing&right=missing")).status_code == 404


@pytest.mark.asyncio
async def test_actual_local_paper_fill_does_not_lock_run_cancellation(research, monkeypatch):
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.engine import BacktestEngine
    from tests.factories.backtest import registries

    client, service = research
    markets, signals, _, _ = registries()
    reached = asyncio.Event()

    async def provider(_snapshot):
        return signals, markets

    class PausedAfterFill(BacktestEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, lookback=20)

        async def _run_bars(self, cycle, session):
            persist_progress = self.progress_callback

            async def pause(value):
                await persist_progress(value)
                if (await session.fetch_fills(None)).items:
                    reached.set()
                    await asyncio.Event().wait()

            self.progress_callback = pause
            return await super()._run_bars(cycle, session)

    service.engine_factory = PausedAfterFill
    service.registry_provider = provider
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    run_id = (await client.post("/api/backtest/runs", json=research_payload(pair="BTC/USDT:USDT"))).json()["run_id"]
    await asyncio.wait_for(reached.wait(), 5)
    assert service.task_manager.get(run_id).execution_started is False
    assert (await client.delete(f"/api/backtest/runs/{run_id}")).status_code == 200
    run = (await client.get(f"/api/backtest/runs/{run_id}")).json()
    assert run["status"] == "canceled"
    assert run["progress"] > 0
