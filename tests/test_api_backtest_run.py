"""Canonical admission and historical replay integration."""

import pytest

from tests.factories.research import research as research
from tests.factories.research import research_payload
from tests.factories.research_offline import research_offline  # noqa: F401


@pytest.mark.asyncio
async def test_unnamed_run_is_saved_and_only_explicit_post_runs(research):
    client, service = research
    response = await client.post("/api/backtest/runs", json=research_payload())
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    run_id = response.json()["run_id"]
    await service.task_manager.drain()
    body = (await client.get(f"/api/backtest/runs/{run_id}")).json()
    assert body["status"] == "completed"
    assert body["result"]["metrics"]["win_rate"] is None
    assert body["params"]["name"] is None
    assert body["config_snapshot"]["market_data"]["parameters"]["market_adapter_id"] == "bybit"
    assert "execution" not in body["config_snapshot"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"start": "not-a-date"},
        {"start": "2024-01-03"},
        {"end": "2199-01-01"},
        {"initial_equity": "50"},
        {"fee_rate": "-1"},
        {"slippage_bps": "10000"},
        {"interval": "weird"},
        {"session_name": "old"},
        {"initial_capital": 10000},
        {"save_dir": "retired-directory"},
        {"mode": "llm"},
    ],
)
async def test_invalid_or_retired_inputs_are_rejected_before_admission(research, change):
    client, service = research
    response = await client.post("/api/backtest/runs", json=research_payload(**change))
    assert response.status_code == 422
    assert await service.store.list() == []


@pytest.mark.asyncio
async def test_old_post_and_session_routes_have_no_alias(research):
    client, _ = research
    assert (await client.post("/api/backtest/run", json=research_payload())).status_code == 404
    assert (await client.get("/api/backtest/sessions")).status_code == 404


@pytest.mark.asyncio
async def test_real_replay_journal_and_evaluations_survive_fresh_services(research, monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.store import BacktestStore
    from cryptotrader.db import dispose_engine
    from cryptotrader.decision.read_service import DecisionReadService
    from cryptotrader.journal.store import MultiVenueCycleStore
    from cryptotrader.signals.evaluation import EvaluationService
    from cryptotrader.signals.evaluation_store import EvaluationStore
    from tests.factories.backtest import START, registries

    client, service = research
    markets, signals, source, component = registries()

    async def provider(_snapshot):
        return signals, markets

    from cryptotrader.backtest.engine import BacktestEngine

    errors = []

    class CapturingEngine(BacktestEngine):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, lookback=20)

        async def run(self):
            try:
                return await super().run()
            except Exception as error:
                errors.append(error)
                raise

    service.engine_factory = CapturingEngine
    service.registry_provider = provider
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    response = await client.post(
        "/api/backtest/runs", json=research_payload(initial_equity="1000", pair="BTC/USDT:USDT")
    )
    run_id = response.json()["run_id"]
    await service.task_manager.drain()
    original = await service.store.get(run_id)
    assert original.status == "completed", errors or original.error
    assert original.result.fills
    assert original.result.cycle_records
    url = service.store.database_url
    await dispose_engine(url)
    fresh = BacktestStore(url)
    restored = await fresh.get(run_id)
    assert restored.result.equity_curve == original.result.equity_curve
    assert restored.result.fills == original.result.fills
    assert restored.result.cycle_records == original.result.cycle_records
    journal = MultiVenueCycleStore(url)
    read = DecisionReadService(journal)
    decision = await read.get(restored.result.decision_ids[0])
    assert decision.mode == "backtest"
    assert decision.created_at == START + timedelta(hours=1)
    assert decision.components[0].component_id == "fixture"
    # Keep one contemporaneous live sample in a separate group.
    first = restored.result.cycle_records[0]
    await journal.save(
        replace(
            first,
            cycle_id="live-evaluation-fixture",
            book_results=(),
            execution_status="not_started",
            run=replace(first.run, mode="analysis", origin="manual"),
        )
    )
    evaluation_store = EvaluationStore(url)
    evaluations = EvaluationService(journal, evaluation_store, source_factory=lambda config: source)
    # Future evaluations are unnecessary for proving the original frozen samples exist.
    await evaluations.evaluate_due(START)
    groups = (await evaluation_store.summary()).groups
    assert {group.mode for group in groups} == {"backtest", "analysis"}
    assert sum(group.total for group in groups if group.mode == "backtest") == len(restored.result.cycle_records)
    count = len(component.contexts)
    for identity in restored.result.decision_ids:
        assert (await client.get(f"/api/decisions/{identity}")).status_code == 200
    await client.get(f"/api/backtest/runs/{run_id}")
    assert len(component.contexts) == count
