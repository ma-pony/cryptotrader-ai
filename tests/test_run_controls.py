"""Explicit admission and pause controls, using isolated records and offline owners."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SchedulerConfig, TriggerConfig
from tests.factories.runtime_config import active_document


@pytest.mark.asyncio
@pytest.mark.parametrize("application_in_progress", [False, True])
async def test_runtime_status_is_readable_without_global_activation(monkeypatch, application_in_progress):
    from api.main import app

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)

    previous = getattr(app.state, "runtime", None)
    document = active_document()
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(1, document, datetime.now(UTC)),
        application_in_progress=application_in_progress,
        run_service=SimpleNamespace(
            readiness=AsyncMock(
                return_value={
                    "analysis": {"ready": True, "reasons": []},
                    "trading": {"ready": False, "reasons": []},
                    "components": [],
                    "saved_revision": 1,
                    "applied_revision": 1,
                    "apply_error": None,
                    "automation_enabled": False,
                    "latest_run_at": None,
                    "execution_pairs": [],
                }
            )
        ),
    )
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/runtime/status")
        assert response.status_code == 200
        assert response.json()["analysis"]["ready"] is True
        assert response.json()["automation_enabled"] is False
    finally:
        app.state.runtime = previous


@pytest.mark.asyncio
async def test_trigger_uses_run_service_so_pause_cannot_be_bypassed():
    from api import main

    class Engine:
        instance = None

        def __init__(self, _store, _redis, callback, _config):
            self.callback = callback
            self.start = AsyncMock()
            Engine.instance = self

    document = active_document(triggers=TriggerConfig(enabled=True), scheduler=SchedulerConfig(enabled=True))
    service = SimpleNamespace(run_automatic=AsyncMock(return_value=None))
    from tests.runtime_lease import static_cycle_lease

    old_cycle = SimpleNamespace(run=AsyncMock())
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(1, document, datetime.now(UTC)),
        repository=SimpleNamespace(database_url="sqlite+aiosqlite://"),
        run_service=service,
        execution_lease=lambda _pair: static_cycle_lease(old_cycle)(),
    )
    application = SimpleNamespace(state=SimpleNamespace(runtime=runtime))
    with (
        patch("cryptotrader.triggers.engine.PriceTriggerEngine", Engine),
        patch("cryptotrader.triggers.store.TriggerRuleStore.ensure_tables", AsyncMock()),
    ):
        await main._init_trigger_engine(application)
    assert Engine.instance is not None
    await Engine.instance.callback("BTC/USDT", {"source": "price_change"})
    service.run_automatic.assert_awaited_once_with("BTC/USDT", "trigger")


@pytest.fixture
async def controls(tmp_path, monkeypatch):
    import base64
    from contextlib import asynccontextmanager
    from dataclasses import replace
    from unittest.mock import Mock

    from cryptotrader.configuration import registry as extensions
    from cryptotrader.runtime import build_runtime
    from cryptotrader.runtime_config.models import ExecutionConfig, SignalComponentConfig
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault
    from tests.factories.runtime_config import allocation, book, connection, signal_config
    from tests.factories.workbench_extensions import sample_registry
    from tests.test_analysis_isolation import FakeMarket

    registry, _ = sample_registry()
    market = FakeMarket()
    registry.market_sources["default"] = replace(registry.market_sources["default"], factory=lambda *_a, **_k: market)
    connected = []

    class Adapter:
        adapter_id = "paper"

        def capabilities(self, _environment):
            return registry.venues["paper"].configuration.capabilities

        async def connect(self, connection, credentials):
            connected.append(connection.id)
            return SimpleNamespace(
                connection_id=connection.id,
                close=AsyncMock(),
                fetch_portfolio=AsyncMock(side_effect=RuntimeError("offline portfolio unavailable")),
            )

    paper_factory = Mock(return_value=Adapter())
    forbidden_factory = Mock(side_effect=AssertionError("unconfirmed account factory"))
    registry.venues["paper"] = replace(registry.venues["paper"], factory=paper_factory)
    registry.venues["okx"] = replace(registry.venues["okx"], factory=forbidden_factory)
    monkeypatch.setattr(extensions, "get_extension_registry", lambda: registry)
    execution = ExecutionConfig(
        pairs=("BTC/USDT", "BTC/USDT:USDT"),
        connections=(
            connection("paper-a"),
            connection("paper-b"),
            connection("live", "live"),
            connection("unused", "live"),
        ),
        books=(
            book("a", "simulated", allocation("paper-a")),
            book("b", "simulated", allocation("paper-b")),
            book("real", "real", allocation("live")),
        ),
    )
    document = active_document(
        execution=execution,
        signals=signal_config(
            components=(SignalComponentConfig(component_id="sample_signal", enabled=True, weight=1),)
        ),
        scheduler=SchedulerConfig(automation_enabled=True, enabled=True),
        triggers=TriggerConfig(enabled=True),
    )
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'controls.db'}"
    await migrate_workbench_schema(database_url)
    repository = RuntimeConfigRepository(
        database_url,
        CredentialVault(base64.urlsafe_b64encode(b"c" * 32).decode()),
        default_factory=lambda: document,
    )
    reveal = AsyncMock(side_effect=AssertionError("unconfirmed credential read"))
    monkeypatch.setattr(repository, "reveal_credentials", reveal)

    @asynccontextmanager
    async def execution_book_lease(_url, _book):
        yield

    monkeypatch.setattr("cryptotrader.cycle_lock.execution_book_lease", execution_book_lease)
    runtime = await build_runtime(repository=repository)
    yield SimpleNamespace(
        runtime=runtime,
        repository=repository,
        connected=connected,
        forbidden_factory=forbidden_factory,
        reveal=reveal,
        market=market,
    )
    await runtime.close()


async def test_runtime_automation_uses_canonical_resource_without_legacy_aliases(controls, monkeypatch):
    from api.main import app

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    previous = getattr(app.state, "runtime", None)
    app.state.runtime = controls.runtime
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put("/api/runtime/automation", json={"enabled": False, "expected_revision": 1})
            assert response.status_code == 200
            assert response.json()["document"]["scheduler"]["automation_enabled"] is False
            assert response.json()["revision"] == 2
            assert (await client.get("/api/runtime-status")).status_code == 404
            assert (
                await client.put("/api/runtime-status/automation", json={"enabled": True, "expected_revision": 2})
            ).status_code == 404
        assert (await controls.repository.get_or_create()).document.scheduler.automation_enabled is False
        assert controls.connected == []
        controls.forbidden_factory.assert_not_called()
        controls.reveal.assert_not_called()
    finally:
        app.state.runtime = previous


@pytest.mark.parametrize("enabled", [False, True])
async def test_explicit_automation_survives_ordinary_save_of_older_rules(controls, monkeypatch, enabled):
    from api.main import app

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    monkeypatch.setattr(app.state, "runtime", controls.runtime)
    monkeypatch.setattr(app.state, "refresh_runtime_owners", None, raising=False)
    monkeypatch.setattr(app.state, "clear_runtime_owners", None, raising=False)
    if enabled:
        await controls.runtime.run_service.set_automation(False, 1)
    original = await controls.repository.get_or_create()
    old_draft = original.document.model_dump(mode="json")
    for connection in old_draft["execution"]["connections"]:
        connection.pop("credential_ref")
    old_draft["scheduler"]["interval_minutes"] = 60
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        explicit = await client.put(
            "/api/runtime/automation", json={"enabled": enabled, "expected_revision": original.revision}
        )
        assert explicit.status_code == 200, explicit.text
        assert explicit.json()["document"]["scheduler"]["automation_enabled"] is enabled
        response = await client.put(
            "/api/config", json={"expected_revision": explicit.json()["revision"], "document": old_draft}
        )
        assert response.status_code == 200, response.text
        assert response.json()["document"]["scheduler"]["automation_enabled"] is enabled
    saved = await controls.repository.get_or_create()
    assert controls.runtime.snapshot.document == saved.document
    assert saved.document.scheduler.automation_enabled is enabled
    assert saved.document.scheduler.interval_minutes == 60
    assert saved.document.execution == original.document.execution
    assert controls.connected == []
    controls.forbidden_factory.assert_not_called()
    controls.reveal.assert_not_called()


async def test_scope_requires_every_eligible_pool_and_never_opens_skipped_accounts(controls):
    service = controls.runtime.run_service
    scope = await service.trading_scope("BTC/USDT")
    assert [(book.book_id, book.eligible) for book in scope.books] == [("a", True), ("b", True), ("real", False)]
    assert "real_authorization_missing" in [reason.code for reason in scope.books[2].reasons]
    assert controls.connected == []
    for ids, revision, pair in [(("a",), 1, "BTC/USDT"), (("a", "b"), 0, "BTC/USDT"), (("a", "b"), 1, "ETH/USDT")]:
        with pytest.raises(ValueError):
            await service.start_trading(pair, revision, ids)
    decision_id = await service.start_trading("BTC/USDT", 1, ("b", "a"))
    await controls.runtime.task_manager.drain()
    record = await controls.runtime.journal.get(decision_id)
    assert record.run.mode == "trading"
    assert record.run.origin == "manual"
    assert {book.book_id for book in record.book_results} == {"a", "b"}
    assert controls.connected == ["paper-a", "paper-b"]
    controls.forbidden_factory.assert_not_called()
    controls.reveal.assert_not_called()


async def test_global_manual_run_reads_exit_stop_and_only_opens_remaining_eligible_pool(controls, monkeypatch):
    from contextlib import asynccontextmanager
    from decimal import Decimal

    from api.main import app
    from cryptotrader.pair import Pair
    from cryptotrader.venues.models import OrderIntent
    from cryptotrader.venues.paper import PaperVenueAdapter
    from tests.test_account_operations import wait_operation

    runtime = controls.runtime
    paper = PaperVenueAdapter()
    paper.database_url = controls.repository.database_url
    pair = Pair.parse("BTC/USDT")

    async def connect(connection, credentials):
        controls.connected.append(connection.id)
        session = await paper.connect(connection, credentials)
        await session.set_quote(pair, Decimal("100"))
        return session

    monkeypatch.setattr(runtime.venue_registry.require("paper"), "connect", connect)

    @asynccontextmanager
    async def offline_ownership(_self, _book):
        yield

    monkeypatch.setattr("cryptotrader.execution_ownership.ExecutionOwnership.book", offline_ownership)
    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    monkeypatch.setattr(app.state, "runtime", runtime)
    async with runtime.account_session("paper-a") as session:
        await session.place_order(OrderIntent(pair, "buy", Decimal("2"), "market", None, False))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/accounts/paper-a/operations/prepare",
            json={"pair": "BTC/USDT", "kind": "flatten", "expected_revision": 1, "confirm_stop": True},
        )
        assert response.status_code == 202, response.text
        operation = await wait_operation(client, response.json()["operation_id"])
        assert operation["status"] == "awaiting_confirmation"
        response = await client.post(
            f"/api/account-operations/{operation['operation_id']}/execute", json={"plan_version": 1}
        )
        assert response.status_code == 202, response.text
        assert (await wait_operation(client, operation["operation_id"]))["status"] == "completed"

        controls.connected.clear()
        scope = await runtime.run_service.trading_scope("BTC/USDT")
        assert scope.saved_revision == 2
        assert [(book.book_id, book.enabled, book.eligible) for book in scope.books] == [
            ("a", False, False),
            ("b", True, True),
            ("real", True, False),
        ]
        response = await client.post(
            "/api/trading-runs", json={"pair": "BTC/USDT", "expected_revision": 2, "confirmed_book_ids": ["b"]}
        )
        assert response.status_code == 202, response.text
        await runtime.task_manager.drain()
        record = await runtime.journal.get(response.json()["decision_id"])
    assert {book.book_id for book in record.book_results} == {"b"}
    assert controls.connected == ["paper-b"]
    assert not (await controls.repository.get_existing()).document.execution.books[0].enabled
    controls.forbidden_factory.assert_not_called()
    controls.reveal.assert_not_called()


@pytest.mark.parametrize("active", [False, True])
async def test_explicit_config_migration_backs_up_original_and_never_reactivates_inactive_books(
    controls, tmp_path, active
):
    import json

    from sqlalchemy import update

    from cryptotrader.db import get_async_session
    from cryptotrader.migrations import workbench
    from cryptotrader.runtime_config.repository import _RuntimeConfigRow

    legacy = controls.runtime.snapshot.document.model_dump(mode="json")
    legacy["system"] = {"active": active}
    legacy["scheduler"]["pairs"] = legacy["execution"].pop("pairs")
    legacy["scheduler"].pop("automation_enabled")
    legacy["signals"]["hitl_required"] = True
    session = await get_async_session(controls.repository.database_url)
    async with session, session.begin():
        await session.execute(update(_RuntimeConfigRow).values(document=legacy))
    backup = tmp_path / "config-backup.json"
    assert await workbench.migrate_run_controls(controls.repository.database_url, backup) == 1
    saved = await controls.repository.get_or_create()
    assert saved.document.execution.pairs == ("BTC/USDT", "BTC/USDT:USDT")
    assert saved.document.scheduler.enabled is True
    assert saved.document.triggers.enabled is True
    assert saved.document.signals.components == controls.runtime.snapshot.document.signals.components
    assert saved.document.execution.connections == controls.runtime.snapshot.document.execution.connections
    assert (
        saved.document.execution.books[0].allocations
        == controls.runtime.snapshot.document.execution.books[0].allocations
    )
    assert saved.document.scheduler.automation_enabled is False
    assert all(book.enabled is active for book in saved.document.execution.books)
    raw_backup = json.loads(backup.read_text())
    assert raw_backup["row"]["document"] == legacy
    assert await workbench.migrate_run_controls(controls.repository.database_url, backup) == 0


def test_run_control_migration_uses_orm_jsonb_binding_for_postgres():
    from sqlalchemy import update
    from sqlalchemy.dialects.postgresql import dialect
    from sqlalchemy.schema import CreateTable

    from cryptotrader.runtime_config.repository import _RuntimeConfigRow

    compiled = str(
        update(_RuntimeConfigRow).values(document={"execution": {"pairs": ["BTC/USDT"]}}).compile(dialect=dialect())
    )
    assert "::JSONB" in compiled
    assert "document JSONB" in str(CreateTable(_RuntimeConfigRow.__table__).compile(dialect=dialect()))


async def test_pause_stops_both_automatic_sources_but_manual_and_history_remain(controls):
    service = controls.runtime.run_service
    saved_id = await service.start_analysis(
        __import__("cryptotrader.pair", fromlist=["Pair"]).Pair.parse("BTC/USDT"), 1
    )
    await controls.runtime.task_manager.drain()
    paused = await service.set_automation(False, 1)
    assert paused.document.scheduler.enabled is True
    assert paused.document.triggers.enabled is True
    assert paused.document.scheduler.automation_enabled is False
    assert await service.run_automatic("BTC/USDT", "scheduled") is None
    assert await service.run_automatic("BTC/USDT", "trigger") is None
    assert controls.connected == []
    assert (await controls.runtime.read_service.get(saved_id)).decision_id == saved_id
    manual = await service.start_trading("BTC/USDT", paused.revision, ("a", "b"))
    await controls.runtime.task_manager.drain()
    assert (await controls.runtime.journal.get(manual)).book_results


async def test_revision_change_while_queued_is_rejected_before_account_access(controls, monkeypatch):
    service = controls.runtime.run_service
    original = controls.runtime.task_manager.create

    def change_after_enqueue(*args, **kwargs):
        task = original(*args, **kwargs)
        snapshot = controls.runtime.snapshot
        monkeypatch.setattr(
            controls.repository,
            "get_or_create",
            AsyncMock(return_value=RuntimeConfigSnapshot(2, snapshot.document, snapshot.updated_at)),
        )
        return task

    monkeypatch.setattr(controls.runtime.task_manager, "create", change_after_enqueue)
    decision_id = await service.start_trading("BTC/USDT", 1, ("a", "b"))
    await controls.runtime.task_manager.drain()
    assert controls.runtime.task_manager.get(decision_id).task.exception() is None
    assert (await controls.runtime.journal.get(decision_id)).cycle_status == "cycle_failed"
    assert controls.connected == []
    controls.forbidden_factory.assert_not_called()
    controls.reveal.assert_not_called()


@pytest.mark.parametrize("cached", [True, False, "sentinel"])
def test_kronos_readiness_inspects_offline_cache_without_model_load(tmp_path, monkeypatch, cached):
    from unittest.mock import Mock

    from cryptotrader.configuration.catalog import ComponentDependency
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.decision.readiness import local_artifact_available

    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"offline weight fixture")
    cache = Mock(return_value=str(weights) if cached is True else (object() if cached == "sentinel" else None))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cache)
    dependency = ComponentDependency(
        "local_artifact",
        "fixture/model",
        LocalizedText(en_US="Model", zh_CN="模型"),
        "signals.components.kronos.parameters.model_name",
    )
    assert local_artifact_available(dependency) is (cached is True)
    cache.assert_called()
    local = tmp_path / "model"
    local.mkdir()
    (local / "model.safetensors").write_bytes(b"local")
    from dataclasses import replace

    cache.reset_mock()
    assert local_artifact_available(replace(dependency, key=str(local)))
    cache.assert_not_called()
    assert not local_artifact_available(replace(dependency, key=str(tmp_path / "missing")))
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    assert local_artifact_available(
        replace(dependency, key=str(gate), configuration_path="signals.components.kronos.parameters.gate_path")
    )


async def test_dynamic_dependency_and_account_draft_readiness_are_independent(controls, monkeypatch):
    from dataclasses import replace

    from cryptotrader.configuration.catalog import ComponentDependency
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.registry import get_extension_registry
    from cryptotrader.decision.service import CapabilityUnavailableError
    from cryptotrader.pair import Pair

    registry = get_extension_registry()
    component = registry.components["sample_signal"]
    registry.components["sample_signal"] = replace(
        component,
        configuration=replace(
            component.configuration,
            dependency_resolver=lambda _parameters: (
                ComponentDependency(
                    "local_artifact", "missing-fixture", LocalizedText(en_US="Fixture", zh_CN="离线夹具"), "sample.path"
                ),
            ),
        ),
    )
    monkeypatch.setattr("cryptotrader.decision.readiness.local_artifact_available", lambda _dep: False)
    status = await controls.runtime.run_service.readiness()
    assert not status.analysis.ready
    assert status.components[0].dependencies[0].key == "missing-fixture"
    assert status.analysis.reasons[0].path == "sample.path"
    assert status.automation_enabled is True
    with pytest.raises(CapabilityUnavailableError):
        await controls.runtime.run_service.start_analysis(Pair.parse("BTC/USDT"), 1)
    assert controls.connected == []
    controls.reveal.assert_not_called()

    from cryptotrader.runtime_config.models import SignalComponentConfig

    registry.components["ready_signal"] = replace(
        component,
        configuration=replace(component.configuration, id="ready_signal", dependency_resolver=lambda _parameters: ()),
    )
    document = controls.runtime.snapshot.document
    signals = document.signals.model_copy(
        update={
            "components": (
                SignalComponentConfig(component_id="sample_signal", enabled=False, weight=0),
                SignalComponentConfig(component_id="ready_signal", enabled=True, weight=1),
            )
        }
    )
    await controls.repository.replace(1, document.model_copy(update={"signals": signals}))
    status = await controls.runtime.run_service.readiness()
    assert status.analysis.ready
    assert not status.components[0].enabled
    assert not status.components[0].ready
    assert status.components[1].ready


async def test_one_unavailable_allocation_excludes_whole_book_without_reweighting(controls):
    from dataclasses import replace

    snapshot = controls.runtime.snapshot
    connections = tuple(
        replace(connection, enabled=False) if connection.id == "paper-b" else connection
        for connection in snapshot.document.execution.connections
    )
    from tests.factories.runtime_config import allocation

    books = (
        replace(
            snapshot.document.execution.books[0],
            allocations=(allocation("paper-a", weight=0.5), allocation("paper-b", weight=0.5)),
        ),
        replace(snapshot.document.execution.books[1], enabled=False),
        *snapshot.document.execution.books[2:],
    )
    desired = snapshot.document.model_copy(
        update={
            "execution": snapshot.document.execution.model_copy(update={"connections": connections, "books": books})
        }
    )
    await controls.repository.replace(1, desired)
    scope = await controls.runtime.run_service.trading_scope("BTC/USDT")
    assert not any(book.eligible for book in scope.books)
    assert not scope.books[1].enabled
    with pytest.raises(ValueError):
        await controls.runtime.run_service.start_trading("BTC/USDT", 2, ())
    assert controls.connected == []


@pytest.mark.parametrize("failure", ["existing_backup", "fsync"])
async def test_migration_backup_failure_leaves_original_row_unchanged(controls, tmp_path, monkeypatch, failure):
    from sqlalchemy import select, update

    from cryptotrader.db import get_async_session
    from cryptotrader.migrations import workbench
    from cryptotrader.runtime_config.repository import _RuntimeConfigRow

    legacy = controls.runtime.snapshot.document.model_dump(mode="json")
    legacy["system"] = {"active": False}
    legacy["scheduler"]["pairs"] = legacy["execution"].pop("pairs")
    legacy["scheduler"].pop("automation_enabled")

    async def row():
        session = await get_async_session(controls.repository.database_url)
        async with session:
            saved = (await session.execute(select(_RuntimeConfigRow))).scalar_one()
            return {column.name: getattr(saved, column.name) for column in _RuntimeConfigRow.__table__.columns}

    session = await get_async_session(controls.repository.database_url)
    async with session, session.begin():
        await session.execute(update(_RuntimeConfigRow).values(document=legacy))
    original = await row()
    backup = tmp_path / "backup.json"
    if failure == "existing_backup":
        backup.write_text("original backup")
    else:
        monkeypatch.setattr(workbench.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("offline disk failure")))
    with pytest.raises(OSError):
        await workbench.migrate_run_controls(controls.repository.database_url, backup)
    assert await row() == original
    if failure == "existing_backup":
        assert backup.read_text() == "original backup"


async def test_automation_owner_failure_is_fail_closed_and_cleans_both_owners(controls):
    runtime = controls.runtime
    runtime.refresh_owners = AsyncMock(side_effect=RuntimeError("owner failed"))
    runtime.clear_owners = AsyncMock(side_effect=RuntimeError("clear failed"))
    with pytest.raises(RuntimeError, match="owner failed"):
        await runtime.run_service.set_automation(False, 1)
    snapshot = await controls.repository.get_or_create()
    assert snapshot.apply_status == "failed"
    assert snapshot.applied_revision == 1
    assert snapshot.document.scheduler.automation_enabled is False
    assert not runtime.application_in_progress
    assert runtime.sessions == {}
    assert runtime.cycle is None
    runtime.clear_owners.assert_awaited_once()


async def test_trading_after_shutdown_terminalizes_queued_record_and_http_is_503(controls, monkeypatch):
    from api.main import app

    runtime = controls.runtime
    monkeypatch.setattr(app.state, "runtime", runtime)
    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    await runtime.task_manager.shutdown()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/trading-runs", json={"pair": "BTC/USDT", "expected_revision": 1, "confirmed_book_ids": ["a", "b"]}
        )
    assert response.status_code == 503
    records = await runtime.journal.list(limit=10)
    assert len(records) == 1
    assert records[0].cycle_status == "cancelled"
    assert records[0].run.failure.code == "admission_closed"
    assert controls.connected == []


async def test_run_service_order_marker_makes_shutdown_wait_without_external_observer(controls, monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager

    from cryptotrader.cycle_events import CycleEvent
    from cryptotrader.decision.models import CycleOutcome
    from cryptotrader.tasks import ExecutionInProgressError, TaskManagerClosedError

    runtime = controls.runtime
    started, release = asyncio.Event(), asyncio.Event()

    async def run(request):
        await runtime.events.publish(CycleEvent("book_execution_started", {"book_id": "a"}))
        started.set()
        await release.wait()
        return CycleOutcome(request.decision_id, 1, None, (), "no_change", "not_started", False)

    @asynccontextmanager
    async def lease(*_args, **_kwargs):
        yield SimpleNamespace(run=run)

    monkeypatch.setattr(runtime, "execution_lease", lease)
    decision_id = await runtime.run_service.start_trading("BTC/USDT", 1, ("a", "b"))
    await asyncio.wait_for(started.wait(), 1)
    managed = runtime.task_manager.get(decision_id)
    assert managed.orders_started
    with pytest.raises(ExecutionInProgressError):
        runtime.task_manager.create(decision_id, "BTC/USDT", AsyncMock(), "manual")
    shutdown = asyncio.create_task(runtime.task_manager.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert not managed.task.cancelled()
    with pytest.raises(TaskManagerClosedError):
        runtime.task_manager.create("late", "BTC/USDT", AsyncMock(), "manual")
    release.set()
    await shutdown
    assert managed.outcome.cycle_id == decision_id
