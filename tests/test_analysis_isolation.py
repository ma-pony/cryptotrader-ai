"""Independent analysis uses the real runtime/runner without any account capability."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from cryptotrader.configuration import registry as extensions
from cryptotrader.pair import Pair
from cryptotrader.runtime import build_runtime
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    RuntimeConfigSnapshot,
    SignalComponentConfig,
    SignalConfig,
)
from cryptotrader.runtime_config.repository import CredentialNotConfigured
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.presentation import TextBlock
from tests.factories.runtime_config import active_document, allocation, book, connection

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)
PAIR = Pair.parse("BTC/USDT:USDT")


class FakeComponent:
    id = "kronos"
    display_name = "Fixture"
    description = "Offline deterministic component"

    def __init__(self):
        self.contexts = []
        self.fail = False

    def requirements(self):
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, context):
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("private-provider-token-never-persist")
        return ComponentSignal(self.id, "long", 0.8, "saved opinion", blocks=(TextBlock(title="结果", body="已保存"),))


class FakeMarket:
    id = "default"

    def __init__(self):
        self.requirements_seen = []

    def requirements(self):
        return DataRequirements()

    async def read_candles(self, pair, timeframe, start, end, as_of):
        return ()

    async def collect(self, pair, as_of, requirements):
        self.requirements_seen.append(requirements)
        return SignalContext(pair, as_of, self.id, pair.market_type, 100.0, 5.0, {})


class Repository:
    def __init__(self, snapshot, database_url):
        self.snapshot = snapshot
        self.database_url = database_url
        self.reveal_credentials = AsyncMock(return_value=None)
        self.credential_state = AsyncMock(return_value=SimpleNamespace(updated_at=NOW))

    async def get_or_create(self):
        return self.snapshot

    async def reveal_token(self, _ref):
        raise CredentialNotConfigured("offline")


@pytest.fixture
async def analysis_fixture(tmp_path, monkeypatch):
    component, market = FakeComponent(), FakeMarket()
    session = SimpleNamespace(
        connection_id="live", close=AsyncMock(), fetch_portfolio=AsyncMock(), place_order=AsyncMock()
    )
    adapter = SimpleNamespace(adapter_id="okx", capabilities=Mock(), connect=AsyncMock(return_value=session))
    venue_factory = Mock(return_value=adapter)
    registered = extensions.get_extension_registry()
    registered.venues["okx"] = replace(registered.venues["okx"], factory=venue_factory)
    registered.components["kronos"] = replace(registered.components["kronos"], factory=lambda _ctx: component)
    registered.components["kronos"] = replace(
        registered.components["kronos"],
        configuration=replace(
            registered.components["kronos"].configuration, dependency_resolver=lambda _parameters: ()
        ),
    )
    registered.market_sources["default"] = replace(
        registered.market_sources["default"], factory=lambda *_args, **_kwargs: market
    )
    monkeypatch.setattr(extensions, "get_extension_registry", lambda: registered)
    live = connection("live", "live")
    execution = ExecutionConfig(
        connections=(live,),
        books=(book("real", "real", allocation("live"), hitl_required=True),),
        live_order_execution_enabled=True,
    )
    document = active_document(
        execution=execution,
        signals=SignalConfig(
            components=(SignalComponentConfig(component_id="kronos", enabled=True, weight=1.0),),
            neutral_threshold=0.2,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
        ),
    )
    snapshot = RuntimeConfigSnapshot(7, document, NOW)
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'analysis.db'}"
    await migrate_workbench_schema(database_url)
    repository = Repository(snapshot, database_url)
    approval_writer = AsyncMock()
    monkeypatch.setattr("cryptotrader.hitl.store.BookApprovalStore.create", approval_writer)
    return SimpleNamespace(
        repository=repository,
        component=component,
        market=market,
        venue_factory=venue_factory,
        adapter=adapter,
        session=session,
        approval_writer=approval_writer,
    )


@pytest.mark.asyncio
async def test_analysis_http_admission_uses_saved_revision_and_survives_response(analysis_fixture):
    import httpx
    from fastapi import Depends, FastAPI

    from api.dependencies import verify_api_key
    from api.routes.analyses import router as analysis_router
    from api.routes.decisions import router as decision_router

    fixture = analysis_fixture
    runtime = await build_runtime(repository=fixture.repository)
    app = FastAPI()
    app.state.runtime = runtime
    app.include_router(analysis_router, dependencies=[Depends(verify_api_key)])
    app.include_router(decision_router, dependencies=[Depends(verify_api_key)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            stale = await client.post("/api/analyses", json={"pair": PAIR.canonical(), "expected_revision": 6})
            assert stale.status_code == 409
            invalid = await client.post(
                "/api/analyses", json={"pair": PAIR.canonical(), "expected_revision": 7, "mode": "trading"}
            )
            assert invalid.status_code == 422
            response = await client.post("/api/analyses", json={"pair": PAIR.canonical(), "expected_revision": 7})
            assert response.status_code == 202
            decision_id = response.json()["decision_id"]
        await runtime.task_manager.drain()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            saved = await client.get(f"/api/decisions/{decision_id}")
        assert saved.status_code == 200
        assert saved.json()["status"] == "completed"
        assert saved.json()["books"] == []
        assert saved.json()["config_revision"] == 7
        fixture.venue_factory.assert_not_called()
        fixture.approval_writer.assert_not_called()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_shutdown_during_queued_save_rejects_late_http_admission(analysis_fixture, monkeypatch):
    import asyncio

    import httpx
    from fastapi import Depends, FastAPI

    from api.dependencies import verify_api_key
    from api.routes.analyses import router

    fixture = analysis_fixture
    fixture.repository.database_url = None
    runtime = await build_runtime(repository=fixture.repository)
    entered, release = asyncio.Event(), asyncio.Event()
    original_save = runtime.journal.save

    async def blocked_save(record):
        entered.set()
        await release.wait()
        await original_save(record)

    monkeypatch.setattr(runtime.journal, "save", blocked_save)
    app = FastAPI()
    app.state.runtime = runtime
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            admission = asyncio.create_task(
                client.post("/api/analyses", json={"pair": PAIR.canonical(), "expected_revision": 7})
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            await runtime.close()
            assert runtime._closed
            assert runtime.task_manager._tasks == {}
            release.set()
            response = await admission
        await runtime.task_manager.drain()
        assert response.status_code == 503
        assert fixture.component.contexts == []
        assert runtime.task_manager._tasks == {}
        records = await runtime.read_service.list()
        assert records.total == 1
        assert records.items[0].status == "cancelled"
        assert records.items[0].finished_at is not None
        assert records.items[0].failure.code == "admission_closed"
        assert records.items[0].books == []
        fixture.venue_factory.assert_not_called()
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_analysis_rejects_naive_clock_before_queueing(analysis_fixture):
    runtime = await build_runtime(repository=analysis_fixture.repository)
    runtime.run_service.clock = lambda: NOW.replace(tzinfo=None)
    try:
        with pytest.raises(ValueError, match="timezone-aware"):
            await runtime.run_service.start_analysis(PAIR, 7)
        assert (await runtime.read_service.list()).total == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_runtime_to_real_analysis_never_constructs_or_reads_accounts(analysis_fixture):
    fixture = analysis_fixture
    runtime = await build_runtime(repository=fixture.repository)
    try:
        fixture.venue_factory.assert_not_called()
        assert runtime.sessions == {}
        decision_id = await runtime.run_service.start_analysis(PAIR, 7)
        queued = await runtime.read_service.get(decision_id)
        assert queued.status == "queued"
        await runtime.task_manager.drain()
        record = await runtime.read_service.get(decision_id)
        assert record.mode == "analysis"
        assert record.status == "completed"
        assert record.books == []
        assert record.target.side == "long"
        assert record.components[0].blocks[0].body == "已保存"
        assert fixture.component.contexts[0].pair == PAIR
        assert fixture.market.requirements_seen[0].candles[0] == CandleRequirement("1h", 100)
        fixture.venue_factory.assert_not_called()
        fixture.adapter.connect.assert_not_called()
        fixture.repository.reveal_credentials.assert_not_called()
        fixture.session.fetch_portfolio.assert_not_called()
        fixture.session.place_order.assert_not_called()
        fixture.approval_writer.assert_not_called()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_failure_keeps_component_snapshot_and_safe_stage(analysis_fixture):
    fixture = analysis_fixture
    fixture.component.fail = True
    runtime = await build_runtime(repository=fixture.repository)
    try:
        decision_id = await runtime.run_service.start_analysis(PAIR, 7)
        await runtime.task_manager.drain()
        record = await runtime.read_service.get(decision_id)
        assert record.status == "failed"
        assert record.failure.stage == "components"
        assert record.failure.code == "component_failed"
        assert record.components[0].status == "failed"
        assert record.components[0].blocks
        assert record.target is None
        assert record.books == []
        assert "private-provider-token" not in record.model_dump_json()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_queued_configuration_and_actual_component_instances_are_frozen(analysis_fixture, monkeypatch):
    fixture = analysis_fixture
    constructed = []
    registered = extensions.get_extension_registry()

    def factory(context):
        component = FakeComponent()
        component.display_name = str(context.document.signals.neutral_threshold)
        constructed.append(component)
        return component

    registered.components["kronos"] = replace(registered.components["kronos"], factory=factory)
    runtime = await build_runtime(repository=fixture.repository)
    try:
        decision_id = await runtime.run_service.start_analysis(PAIR, 7)
        next_signals = fixture.repository.snapshot.document.signals.model_copy(update={"neutral_threshold": 0.9})
        fixture.repository.snapshot = replace(
            fixture.repository.snapshot,
            revision=8,
            document=fixture.repository.snapshot.document.model_copy(update={"signals": next_signals}),
        )
        second_id = await runtime.run_service.start_analysis(PAIR, 8)
        await runtime.task_manager.drain()
        first = await runtime.read_service.get(decision_id)
        second = await runtime.read_service.get(second_id)
        assert (await runtime.journal.get(decision_id)).run.config_snapshot["signals"]["neutral_threshold"] == 0.2
        assert first.target.side == "long"
        assert (await runtime.journal.get(second_id)).run.config_snapshot["signals"]["neutral_threshold"] == 0.9
        assert second.target.side == "flat"
        assert [(item.display_name, len(item.contexts)) for item in constructed] == [("0.2", 1), ("0.9", 1)]
        assert "credential" not in first.model_dump_json()
        assert "redis://" not in first.model_dump_json()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_shutdown_before_worker_first_tick_leaves_terminal_decision(analysis_fixture):
    runtime = await build_runtime(repository=analysis_fixture.repository)
    decision_id = await runtime.run_service.start_analysis(PAIR, 7)
    await runtime.close()
    record = await runtime.read_service.get(decision_id)
    assert record.status == "cancelled"
    assert record.finished_at is not None


@pytest.mark.asyncio
async def test_model_identity_and_validated_parameters_are_saved_without_provider_endpoint(analysis_fixture):
    fixture = analysis_fixture
    runtime = await build_runtime(repository=fixture.repository)
    try:
        decision_id = await runtime.run_service.start_analysis(PAIR, 7)
        await runtime.task_manager.drain()
        record = await runtime.read_service.get(decision_id)
        config = (await runtime.journal.get(decision_id)).run.config_snapshot
        parameters = config["signals"]["components"][0].get("parameters", {})
        assert parameters.get("model_name") == "NeoQuasar/Kronos-base"
        assert config["signals"]["components"][0]["model_identity"]["vocabulary"] == "NeoQuasar/Kronos-Tokenizer-base"
        assert config["llm"]["models"]["tech_agent"] == "gemini-3-flash"
        assert "NeoQuasar/Kronos-base" in record.model_dump_json()
        assert "gemini-3-flash" in record.model_dump_json()
        assert "base_url" not in record.model_dump_json()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_safe_component_stage_survives_runner_without_external_message(analysis_fixture):
    fixture = analysis_fixture

    async def fail(_context):
        from cryptotrader.signals.components.kronos import KronosComponent

        error = KronosComponent._error(fixture.component, "prediction", RuntimeError("private-predictor-body"))
        raise error

    fixture.component.evaluate = fail
    runtime = await build_runtime(repository=fixture.repository)
    try:
        decision_id = await runtime.run_service.start_analysis(PAIR, 7)
        await runtime.task_manager.drain()
        record = await runtime.read_service.get(decision_id)
        assert record.failure.stage == "components.prediction"
        assert "private-predictor-body" not in record.model_dump_json()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_restart_interrupts_unfinished_and_history_survives_trading_pause(analysis_fixture):
    fixture = analysis_fixture
    runtime = await build_runtime(repository=fixture.repository)
    decision_id = await runtime.run_service.start_analysis(PAIR, 7)
    await runtime.task_manager.drain()
    saved = await runtime.journal.get(decision_id)
    unfinished = replace(
        saved,
        cycle_id="orphaned",
        cycle_status="running",
        component_signals=(),
        fused_signal=None,
        target_position=None,
        run=replace(saved.run, finished_at=None),
    )
    await runtime.journal.save(unfinished)
    await runtime.close()
    fixture.repository.snapshot = replace(
        fixture.repository.snapshot,
        revision=8,
        document=fixture.repository.snapshot.document.model_copy(
            update={
                "scheduler": fixture.repository.snapshot.document.scheduler.model_copy(
                    update={"automation_enabled": False}
                )
            }
        ),
    )
    restarted = await build_runtime(repository=fixture.repository)
    try:
        assert restarted.cycle is None
        assert (await restarted.read_service.get(decision_id)).status == "completed"
        orphaned = await restarted.read_service.get("orphaned")
        assert orphaned.status == "interrupted"
        assert orphaned.finished_at is not None
        assert orphaned.books == []
        assert len(fixture.component.contexts) == 1
        fixture.venue_factory.assert_not_called()
    finally:
        await restarted.close()
