"""Runtime graph 的热同步、session 复用与关闭边界。"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.runtime import RuntimeLeaseUnavailableError, build_runtime
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    InfrastructureConfig,
    RuntimeConfigSnapshot,
    SchedulerConfig,
    SystemConfig,
    TriggerConfig,
)
from cryptotrader.runtime_config.repository import CredentialState
from cryptotrader.venues.models import VenueConnection
from tests.factories.runtime_config import runtime_document

NOW = datetime(2026, 8, 29, tzinfo=UTC)


class _Repository:
    database_url = None

    def __init__(self, snapshot: RuntimeConfigSnapshot) -> None:
        self.snapshot = snapshot
        self.credentials: dict[str, tuple[object, datetime]] = {}

    async def get_or_create(self):
        return self.snapshot

    async def credential_state(self, credential_ref):
        value = self.credentials.get(credential_ref)
        return CredentialState(credential_ref, value is not None, value[1] if value else None)

    async def reveal_credentials(self, credential_ref):
        return self.credentials[credential_ref][0]

    async def reveal_token(self, _credential_ref):
        from cryptotrader.runtime_config.repository import CredentialNotConfigured

        raise CredentialNotConfigured("test")

    def publish(self, document) -> RuntimeConfigSnapshot:
        self.snapshot = RuntimeConfigSnapshot(
            self.snapshot.revision + 1,
            document,
            self.snapshot.updated_at + timedelta(seconds=1),
        )
        return self.snapshot

    async def mark_applied(self, revision):
        assert self.snapshot.revision == revision
        self.snapshot = replace(
            self.snapshot,
            apply_status="applied",
            applied_revision=revision,
            apply_error=None,
        )
        return self.snapshot

    async def mark_failed(self, revision, error):
        assert self.snapshot.revision == revision
        self.snapshot = replace(
            self.snapshot,
            apply_status="failed",
            apply_error=error,
        )
        return self.snapshot


class _Session:
    def __init__(self, connection_id: str, *, close_error: BaseException | None = None) -> None:
        self.connection_id = connection_id
        self.close_calls = 0
        self.close_error = close_error

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _Adapter:
    adapter_id = "recording"

    def __init__(self) -> None:
        self.connect_calls: list[tuple[VenueConnection, object | None]] = []
        self.sessions: list[_Session] = []
        self.fail_connection_id: str | None = None
        self.fail_exception: BaseException | None = None
        self.shared_session: _Session | None = None

    async def connect(self, connection, credentials):
        self.connect_calls.append((connection, credentials))
        if connection.id == self.fail_connection_id:
            raise self.fail_exception or RuntimeError("sensitive-adapter-failure")
        session = self.shared_session or _Session(connection.id)
        self.sessions.append(session)
        return session


class _VenueRegistry:
    def __init__(self, adapter: _Adapter) -> None:
        self.adapter = adapter

    def installed_ids(self):
        return frozenset({"recording"})

    def require(self, adapter_id):
        assert adapter_id == "recording"
        return self.adapter


class _InstalledRegistry:
    def __init__(self, installed: set[str]) -> None:
        self._installed = frozenset(installed)

    def installed_ids(self):
        return self._installed


class _MarketRegistry(_InstalledRegistry):
    def __init__(self) -> None:
        super().__init__({"default", "candidate"})
        self.fail_source: str | None = None

    def require(self, source_id):
        if source_id == self.fail_source:
            raise RuntimeError("candidate graph failed")
        return SimpleNamespace(id=source_id)


def _connection(
    connection_id: str,
    *,
    label: str | None = None,
    parameters: dict | None = None,
    enabled: bool = True,
    credential_ref: str | None = None,
) -> VenueConnection:
    return VenueConnection(
        id=connection_id,
        label=label or connection_id,
        adapter_id="recording",
        environment="demo" if credential_ref else "paper",
        enabled=enabled,
        credential_ref=credential_ref,
        leverage=1,
        margin_mode="isolated",
        parameters=parameters or {"initial_equity": "10000"},
    )


def _document(
    *connections: VenueConnection,
    active: bool = True,
    weights: tuple[float, ...] | None = None,
    market_source: str = "default",
):
    from cryptotrader.runtime_config.models import MarketDataConfig

    weights = weights or tuple(1 / len(connections) for _ in connections)
    allocations = tuple(
        ConnectionAllocation(connection.id, True, weight)
        for connection, weight in zip(connections, weights, strict=True)
        if connection.enabled
    )
    books = (
        (
            ExecutionBook(
                id="simulation",
                label="Simulation",
                capital_scope="simulated",
                enabled=True,
                hitl_required=False,
                allocations=allocations,
            ),
        )
        if active
        else ()
    )
    return runtime_document(
        connections=connections,
        books=books,
        system=SystemConfig(active=active),
        execution=ExecutionConfig(connections=connections, books=books),
        infrastructure=InfrastructureConfig(redis_url="redis://runtime-test"),
        market_data=MarketDataConfig(source_id=market_source),
    )


async def _build(document, *, adapter=None, repository=None, markets=None, snapshot=None):
    snapshot = snapshot or RuntimeConfigSnapshot(7, document, NOW)
    repository = repository or _Repository(snapshot)
    adapter = adapter or _Adapter()
    markets = markets or _MarketRegistry()
    runtime = await build_runtime(
        repository=repository,
        snapshot=snapshot,
        signal_registry=_InstalledRegistry({"kronos", "llm_committee"}),
        venue_registry=_VenueRegistry(adapter),
        market_registry=markets,
    )
    return runtime, repository, adapter, markets


@pytest.mark.asyncio
async def test_active_runtime_opens_each_enabled_connection_once_and_closes_all():
    runtime, _, adapter, _ = await _build(_document(_connection("paper-a"), _connection("paper-b")))
    sessions = tuple(runtime.sessions.values())

    assert set(runtime.sessions) == {"paper-a", "paper-b"}
    assert [call[0].id for call in adapter.connect_calls] == ["paper-a", "paper-b"]

    await runtime.close()
    await runtime.close()

    assert all(session.close_calls == 1 for session in sessions)


@pytest.mark.asyncio
async def test_cycle_lease_defers_retired_session_close_until_all_execution_owners_exit():
    first = _connection("paper-a", parameters={"initial_equity": "10000"})
    runtime, repository, adapter, _ = await _build(_document(first))
    old_session = runtime.sessions["paper-a"]

    async with runtime.cycle_lease() as leased_cycle:
        assert leased_cycle.snapshot.revision == 7
        changed = _connection("paper-a", parameters={"initial_equity": "20000"})
        repository.publish(_document(changed))
        replacement = await runtime.reload_for_cycle()
        assert replacement.snapshot.revision == 8
        assert old_session.close_calls == 0

    assert old_session.close_calls == 1
    assert len(adapter.sessions) == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_close_rejects_new_lease_and_waits_for_inflight_owner():
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    lease = runtime.cycle_lease()
    await lease.__aenter__()

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not closing.done()
    with pytest.raises(RuntimeLeaseUnavailableError, match="unavailable"):
        async with runtime.cycle_lease():
            pass

    await lease.__aexit__(None, None, None)
    await closing


@pytest.mark.asyncio
async def test_inactive_runtime_lease_uses_dedicated_unavailable_error():
    runtime, _, _, _ = await _build(_document(active=False))

    with pytest.raises(RuntimeLeaseUnavailableError, match="unavailable"):
        async with runtime.cycle_lease():
            pass


@pytest.mark.asyncio
async def test_setup_runtime_activates_the_published_candidate_revision_in_process():
    runtime, repository, adapter, _ = await _build(_document(active=False))

    activated = repository.publish(_document(_connection("paper-a")))
    async with runtime.cycle_lease() as cycle:
        assert cycle.snapshot == activated
        assert cycle is runtime.cycle

    assert runtime.snapshot == activated
    assert [connection.id for connection, _credentials in adapter.connect_calls] == ["paper-a"]
    await runtime.close()

    await runtime.close()


@pytest.mark.asyncio
async def test_changed_revision_rebuilds_registry_graph_before_publishing_cycle():
    runtime, repository, _adapter, _markets = await _build(_document(_connection("paper-a")))
    discovered_markets = _MarketRegistry()
    discovered: list[object] = []

    async def discover(_document):
        discovered.append(_document)
        return _InstalledRegistry({"kronos", "llm_committee"}), _VenueRegistry(_Adapter()), discovered_markets

    runtime._registry_discoverer = discover
    next_document = _document(_connection("paper-a"), market_source="candidate")
    repository.publish(next_document)

    cycle = await runtime.reload_for_cycle()

    assert discovered == [next_document]
    assert runtime.market_registry is discovered_markets
    assert cycle.market_source.id == "candidate"
    await runtime.close()


@pytest.mark.asyncio
async def test_multiple_reloads_accumulate_unique_retired_sessions_until_lease_drain():
    initial = _connection("paper-a", parameters={"initial_equity": "10000"})
    runtime, repository, adapter, _ = await _build(_document(initial))
    first_session = runtime.sessions["paper-a"]

    async with runtime.cycle_lease():
        repository.publish(_document(_connection("paper-a", parameters={"initial_equity": "20000"})))
        await runtime.reload_for_cycle()
        second_session = runtime.sessions["paper-a"]
        repository.publish(_document(_connection("paper-a", parameters={"initial_equity": "30000"})))
        await runtime.reload_for_cycle()
        assert first_session.close_calls == 0
        assert second_session.close_calls == 0

    assert first_session.close_calls == 1
    assert second_session.close_calls == 1
    assert runtime.sessions["paper-a"].close_calls == 0
    assert len(adapter.sessions) == 3
    await runtime.close()
    assert first_session.close_calls == 1
    assert second_session.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_runtime_close_waits_active_lease_and_unique_session_terminal():
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    lease = runtime.cycle_lease()
    await lease.__aenter__()
    blocking = _BlockingCloseSession("paper-a")
    runtime.sessions = {"paper-a": blocking}

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    assert closing.done() is False
    assert blocking.started.is_set() is False

    await lease.__aexit__(None, None, None)
    await blocking.started.wait()
    assert closing.done() is False
    blocking.release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert blocking.completed is True
    assert blocking.close_calls == 1
    assert runtime._active_leases == 0


@pytest.mark.asyncio
async def test_release_cleanup_cannot_signal_drain_for_a_new_active_generation():
    connection = _connection("paper-a", parameters={"initial_equity": "10000"})
    runtime, repository, _, _ = await _build(_document(connection))
    first_lease = runtime.cycle_lease()
    await first_lease.__aenter__()
    retired = _BlockingCloseSession("paper-a")
    runtime.sessions = {"paper-a": retired}

    repository.publish(_document(_connection("paper-a", parameters={"initial_equity": "20000"})))
    await runtime.reload_for_cycle()
    current = runtime.sessions["paper-a"]

    first_exit = asyncio.create_task(first_lease.__aexit__(None, None, None))
    await retired.started.wait()
    assert runtime._active_leases == 1

    repository.publish(_document(_connection("paper-a", parameters={"initial_equity": "30000"})))
    second_lease = runtime.cycle_lease()
    await second_lease.__aenter__()
    latest = runtime.sessions["paper-a"]
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert closing.done() is False

    retired.release.set()
    await first_exit
    await asyncio.sleep(0)

    assert runtime._active_leases == 1
    assert runtime._leases_drained.is_set() is False
    assert current.close_calls == 0
    assert latest.close_calls == 0
    assert closing.done() is False

    await second_lease.__aexit__(None, None, None)
    await closing
    assert retired.close_calls == 1
    assert current.close_calls == 1
    assert latest.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_close_publishes_owned_completion_before_waiting_for_lifecycle_lock():
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    session = runtime.sessions["paper-a"]
    await runtime._lifecycle_lock.acquire()

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    completion = runtime._close_completion
    assert completion is not None
    assert completion.done() is False

    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    assert closing.done() is False

    runtime._lifecycle_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert completion.done()
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_lease_exit_waits_for_lifecycle_lock_then_signals_drained():
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    lease = runtime.cycle_lease()
    await lease.__aenter__()
    await runtime._lifecycle_lock.acquire()

    exiting = asyncio.create_task(lease.__aexit__(None, None, None))
    await asyncio.sleep(0)
    exiting.cancel()
    await asyncio.sleep(0)
    exiting.cancel()
    assert runtime._active_leases == 1
    assert exiting.done() is False

    runtime._lifecycle_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await exiting
    assert runtime._active_leases == 0
    assert runtime._leases_drained.is_set()
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_close_callers_share_one_owned_completion():
    runtime, _, _, _ = await _build(_document(active=False))
    blocking = _BlockingCloseSession("shared")
    runtime.sessions = {"first": blocking, "alias": blocking}

    first = asyncio.create_task(runtime.close())
    second = asyncio.create_task(runtime.close())
    await blocking.started.wait()
    first.cancel()
    await asyncio.sleep(0)
    assert first.done() is False
    assert second.done() is False

    blocking.release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert blocking.close_calls == 1


@pytest.mark.asyncio
async def test_failed_close_concurrent_retry_callers_share_one_new_completion():
    class FailThenBlockSession:
        def __init__(self) -> None:
            self.close_calls = 0
            self.retry_started = asyncio.Event()
            self.retry_release = asyncio.Event()

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("first close failed")
            self.retry_started.set()
            await self.retry_release.wait()

    runtime, _, _, _ = await _build(_document(active=False))
    session = FailThenBlockSession()
    runtime.sessions = {"session": session}

    with pytest.raises(RuntimeError, match="failed to close venue session"):
        await runtime.close()
    failed_completion = runtime._close_completion

    first_retry = asyncio.create_task(runtime.close())
    await session.retry_started.wait()
    retry_completion = runtime._close_completion
    second_retry = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)

    assert retry_completion is not failed_completion
    assert runtime._close_completion is retry_completion
    session.retry_release.set()
    await asyncio.gather(first_retry, second_retry)
    assert session.close_calls == 2


@pytest.mark.asyncio
async def test_reload_publishes_latest_snapshot_and_new_cycle_without_reopening_unchanged_connections():
    first = _connection("paper-a")
    second = _connection("paper-b")
    runtime, repository, adapter, _ = await _build(_document(first, second, weights=(0.5, 0.5)))
    first_sessions = dict(runtime.sessions)
    first_cycle = runtime.cycle
    changed = _document(first, second, weights=(0.7, 0.3))
    changed_position = changed.risk.position.model_copy(update={"max_total_exposure_pct": 0.65})
    changed = changed.model_copy(
        update={
            "risk": changed.risk.model_copy(update={"position": changed_position}),
            "signals": changed.signals.model_copy(update={"neutral_threshold": 0.3}),
        }
    )
    saved = repository.publish(changed)

    cycle = await runtime.reload_for_cycle()

    assert runtime.snapshot.revision == saved.revision
    assert cycle is runtime.cycle
    assert cycle is not first_cycle
    assert all(runtime.sessions[key] is session for key, session in first_sessions.items())
    assert cycle.book_risk._limits.max_net_exposure == Decimal("0.65")
    assert len(adapter.connect_calls) == 2


@pytest.mark.asyncio
async def test_reload_ignores_label_global_revision_and_parameter_key_order_for_session_reuse():
    connection = _connection("paper-a", parameters={"b": [2, 3], "a": {"x": 1}})
    runtime, repository, adapter, _ = await _build(_document(connection))
    first_session = runtime.sessions[connection.id]
    changed = replace(connection, label="Renamed", parameters={"a": {"x": 1}, "b": [2, 3]})
    repository.publish(_document(changed))

    await runtime.reload_for_cycle()

    assert runtime.sessions[connection.id] is first_session
    assert len(adapter.connect_calls) == 1


@pytest.mark.asyncio
async def test_credential_timestamp_change_reopens_connection_and_closes_replaced_session():
    connection = _connection("demo-a", credential_ref="credential-a")
    snapshot = RuntimeConfigSnapshot(7, _document(connection), NOW)
    repository = _Repository(snapshot)
    repository.credentials["credential-a"] = (object(), NOW)
    runtime, _, adapter, _ = await _build(snapshot.document, repository=repository)
    first_session = runtime.sessions[connection.id]
    repository.credentials["credential-a"] = (object(), NOW + timedelta(seconds=1))
    repository.publish(_document(connection))

    await runtime.reload_for_cycle()

    assert runtime.sessions[connection.id] is not first_session
    assert first_session.close_calls == 1
    assert len(adapter.connect_calls) == 2


@pytest.mark.asyncio
async def test_credential_values_never_enter_session_key_and_same_timestamp_reuses_session():
    connection = _connection("demo-a", credential_ref="credential-a")
    snapshot = RuntimeConfigSnapshot(7, _document(connection), NOW)
    repository = _Repository(snapshot)
    repository.credentials["credential-a"] = (_Credential("first-private-value"), NOW)
    runtime, _, adapter, _ = await _build(snapshot.document, repository=repository)
    first_session = runtime.sessions[connection.id]
    repository.credentials["credential-a"] = (_Credential("second-private-value"), NOW)
    repository.publish(_document(connection))

    await runtime.reload_for_cycle()

    assert runtime.sessions[connection.id] is first_session
    assert len(adapter.connect_calls) == 1
    assert "first-private-value" not in repr(runtime._session_keys)
    assert "second-private-value" not in repr(runtime._session_keys)


@pytest.mark.asyncio
async def test_inactive_reload_closes_sessions_and_publishes_no_cycle():
    connection = _connection("paper-a")
    runtime, repository, _, _ = await _build(_document(connection))
    first_session = runtime.sessions[connection.id]
    saved = repository.publish(_document(connection, active=False))

    cycle = await runtime.reload_for_cycle()

    assert runtime.snapshot is saved
    assert runtime.sessions == {}
    assert cycle is None
    assert runtime.cycle is None
    assert first_session.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("apply_status", ["pending", "failed"])
async def test_non_applied_startup_and_reload_never_admit_a_cycle_or_session(apply_status):
    connection = _connection("paper-a")
    desired = RuntimeConfigSnapshot(7, _document(connection), NOW, apply_status, 6, "application failed")
    repository = _Repository(desired)

    startup, _, startup_adapter, _ = await _build(desired.document, repository=repository, snapshot=desired)
    assert startup.cycle is None
    assert startup.sessions == {}
    assert startup_adapter.connect_calls == []

    runtime, _, adapter, _ = await _build(_document(connection))
    runtime.repository.snapshot = desired
    assert await runtime.reload_for_cycle() is None
    assert runtime.cycle is None
    assert runtime.sessions == {}
    assert adapter.sessions[0].close_calls == 1


@pytest.mark.asyncio
async def test_application_barrier_rejects_new_cycle_work_but_never_blocks_an_existing_lease_release():
    """A desired revision cannot be observed by cycle work until its application commits."""
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    lease = runtime.cycle_lease()
    await lease.__aenter__()
    try:
        async with runtime.application_barrier():
            with pytest.raises(RuntimeLeaseUnavailableError, match="application"):
                await runtime.cycle_lease().__aenter__()
            with pytest.raises(RuntimeError, match="application"):
                await runtime.reload_for_cycle()

            # Releasing an already admitted lease must not wait for the application
            # barrier.  Owner shutdown relies on that property to avoid deadlock.
            await asyncio.wait_for(lease.__aexit__(None, None, None), timeout=0.2)
    finally:
        if runtime._active_leases:
            await lease.__aexit__(None, None, None)
    await runtime.close()


@pytest.mark.asyncio
async def test_application_protocol_starts_real_owner_factories_for_pending_graph_then_exposes_applied_revision(
    monkeypatch,
):
    """Owners initialize from the pending graph; HTTP/cycle admission opens only after commit."""
    from api import main as api_main
    from api.routes.config import publish_pending_snapshot

    old = _connection("paper-a", parameters={"initial_equity": "10000"})
    runtime, repository, _, _ = await _build(_document(old))
    changed = _connection("paper-a", parameters={"initial_equity": "20000"})
    pending = RuntimeConfigSnapshot(8, _document(changed), NOW + timedelta(seconds=1), "pending", 7)
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)
    started: list[tuple[str, int]] = []
    stopped: list[str] = []
    rejected: list[str] = []
    transitions: list[str] = []

    class _Trigger:
        async def stop(self):
            stopped.append("old-trigger")

    app = SimpleNamespace(
        state=SimpleNamespace(runtime=runtime, trigger_engine=_Trigger(), scheduler=object(), scheduler_task=object())
    )

    async def shutdown_scheduler(_app):
        stopped.append("old-scheduler")

    async def init_trigger(_app, *, snapshot=None):
        started.append(("trigger", snapshot.revision))
        _app.state.trigger_engine = SimpleNamespace(stop=lambda: None)

    async def init_scheduler(_app, *, snapshot=None):
        with pytest.raises(RuntimeLeaseUnavailableError):
            await runtime.cycle_lease().__aenter__()
        with pytest.raises(RuntimeError):
            await runtime.reload_for_cycle()
        rejected.append("cycle-and-reload")
        started.append(("scheduler", snapshot.revision))
        _app.state.scheduler = object()
        _app.state.scheduler_task = object()

    monkeypatch.setattr(api_main, "_shutdown_scheduler", shutdown_scheduler)
    monkeypatch.setattr(api_main, "_init_trigger_engine", init_trigger)
    monkeypatch.setattr(api_main, "_init_scheduler", init_scheduler)
    original_activate = runtime.activate_applied
    original_mark_applied = repository.mark_applied

    async def activate_then_record(snapshot):
        transitions.append("activate")
        await original_activate(snapshot)

    async def mark_then_record(revision):
        transitions.append("mark_applied")
        return await original_mark_applied(revision)

    monkeypatch.setattr(runtime, "activate_applied", activate_then_record)
    monkeypatch.setattr(repository, "mark_applied", mark_then_record)

    async with runtime.application_barrier():
        applied = await publish_pending_snapshot(
            runtime,
            pending,
            lambda snapshot: api_main._refresh_runtime_owners(app, snapshot=snapshot),
            candidate,
        )

    assert started == [("trigger", 8), ("scheduler", 8)]
    assert rejected == ["cycle-and-reload"]
    assert stopped == ["old-scheduler", "old-trigger"]
    assert transitions == ["activate", "mark_applied"]
    assert applied.apply_status == "applied"
    assert runtime.snapshot == applied
    assert runtime.cycle is not None
    assert runtime.cycle.snapshot.revision == 8
    await runtime.close()


@pytest.mark.asyncio
async def test_published_revision_replaces_api_scheduler_with_its_new_runtime_schedule(monkeypatch):
    """The API-owned scheduler must be reconstructed from the published revision."""
    from api import main as api_main
    from api.routes.config import publish_pending_snapshot
    from cryptotrader.scheduler import Scheduler

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    revised_document = runtime.snapshot.document.model_copy(
        update={
            "scheduler": SchedulerConfig(
                enabled=True,
                pairs=("ETH/USDT:USDT", "SOL/USDT:USDT"),
                interval_minutes=90,
                daily_summary_hour=13,
            ),
            "triggers": TriggerConfig(
                enabled=True,
                max_rules=9,
                ws_reconnect_max_s=17,
                funding_rate_poll_interval_minutes=11,
            ),
        }
    )
    pending = RuntimeConfigSnapshot(8, revised_document, NOW + timedelta(seconds=1), "pending", 7)
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)
    stopped: list[str] = []

    class OldScheduler:
        def stop(self):
            stopped.append("scheduler")

    class OldTrigger:
        async def stop(self):
            stopped.append("trigger")

    async def wait_for_stop(self):
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def install_trigger(_app, *, snapshot=None):
        _app.state.trigger_engine = SimpleNamespace(config=snapshot.document.triggers, stop=OldTrigger().stop)
        _app.state.trigger_store = object()

    monkeypatch.setattr(Scheduler, "start", wait_for_stop)
    monkeypatch.setattr(api_main, "_init_trigger_engine", install_trigger)
    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=runtime,
            scheduler=OldScheduler(),
            scheduler_task=asyncio.create_task(asyncio.sleep(0)),
            trigger_engine=OldTrigger(),
            trigger_store=object(),
        )
    )
    await app.state.scheduler_task

    async with runtime.application_barrier():
        applied = await publish_pending_snapshot(
            runtime,
            pending,
            lambda snapshot: api_main._refresh_runtime_owners(app, snapshot=snapshot),
            candidate,
        )

    scheduler = app.state.scheduler
    assert stopped == ["scheduler", "trigger"]
    assert scheduler.config_revision == 8
    assert tuple(pair.canonical() for pair in scheduler.pairs) == ("ETH/USDT:USDT", "SOL/USDT:USDT")
    assert scheduler.interval_minutes == 90
    assert scheduler.daily_summary_hour == 13
    assert scheduler.config == applied.document.scheduler
    assert app.state.trigger_engine.config == applied.document.triggers
    assert app.state.trigger_store is not None

    await api_main._clear_runtime_owners(app)
    await runtime.close()


@pytest.mark.asyncio
async def test_application_protocol_owner_start_failure_stops_partial_owner_and_fails_closed(monkeypatch):
    """A partially initialized owner can never survive a failed desired revision."""
    from api import main as api_main
    from api.routes.config import publish_pending_snapshot

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    pending = RuntimeConfigSnapshot(
        8,
        _document(_connection("paper-a", parameters={"initial_equity": "20000"})),
        NOW + timedelta(seconds=1),
        "pending",
        7,
    )
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)
    stopped: list[str] = []

    class _PartialTrigger:
        async def stop(self):
            stopped.append("partial-trigger")

    app = SimpleNamespace(
        state=SimpleNamespace(runtime=runtime, trigger_engine=None, scheduler=None, scheduler_task=None)
    )

    async def init_trigger(_app, *, snapshot=None):
        _app.state.trigger_engine = _PartialTrigger()

    async def init_scheduler(_app, *, snapshot=None):
        raise RuntimeError("scheduler factory failed")

    monkeypatch.setattr(api_main, "_init_trigger_engine", init_trigger)
    monkeypatch.setattr(api_main, "_init_scheduler", init_scheduler)

    async with runtime.application_barrier():
        with pytest.raises(Exception, match="Runtime configuration cannot be applied"):
            await publish_pending_snapshot(
                runtime,
                pending,
                lambda snapshot: api_main._refresh_runtime_owners(app, snapshot=snapshot),
                candidate,
                clear_owners=lambda: api_main._clear_runtime_owners(app),
            )

    assert stopped == ["partial-trigger"]
    assert runtime.snapshot.apply_status == "failed"
    assert runtime.sessions == {}
    assert runtime.cycle is None
    await runtime.close()


@pytest.mark.asyncio
async def test_mark_applied_failure_after_runtime_activation_returns_to_failed_old_applied_revision_and_empty_graph(
    monkeypatch,
):
    """The final repository commit may fail after local activation, but must leave no executable desired graph."""
    from api.routes.config import publish_pending_snapshot

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    pending = RuntimeConfigSnapshot(
        8,
        _document(_connection("paper-a", parameters={"initial_equity": "20000"})),
        NOW + timedelta(seconds=1),
        "pending",
        7,
    )
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)

    async def reject_final_commit(revision):
        assert revision == pending.revision
        assert runtime.snapshot.apply_status == "applied"
        raise RuntimeError("final repository commit unavailable")

    monkeypatch.setattr(repository, "mark_applied", reject_final_commit)
    async with runtime.application_barrier():
        with pytest.raises(Exception, match="Runtime configuration cannot be applied"):
            await publish_pending_snapshot(runtime, pending, None, candidate)

    assert repository.snapshot.apply_status == "failed"
    assert repository.snapshot.applied_revision == 7
    assert runtime.snapshot.apply_status == "failed"
    assert runtime.snapshot.applied_revision == 7
    assert runtime.sessions == {}
    assert runtime.cycle is None
    await runtime.close()


@pytest.mark.asyncio
async def test_real_sqlite_runtime_recovers_in_process_after_final_apply_commit_failure(tmp_path, monkeypatch):
    """A real persisted failed revision is fail-closed and the next desired revision recovers without restart."""
    from api.routes.config import publish_pending_snapshot
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    initial = _document(_connection("paper-a", parameters={"initial_equity": "10000"}))
    repository = RuntimeConfigRepository(
        f"sqlite+aiosqlite:///{tmp_path / 'runtime-lifecycle.db'}",
        CredentialVault(base64.urlsafe_b64encode(b"r" * 32).decode()),
        default_factory=lambda: initial,
    )
    first = await repository.get_or_create()
    adapter = _Adapter()
    runtime, _, _, _ = await _build(initial, repository=repository, adapter=adapter, snapshot=first)
    initial_session = runtime.sessions["paper-a"]
    failed_document = _document(_connection("paper-a", parameters={"initial_equity": "20000"}))
    original_mark_applied = repository.mark_applied
    mark_attempts = 0

    async def fail_final_commit_once(revision):
        nonlocal mark_attempts
        mark_attempts += 1
        if mark_attempts == 1:
            assert revision == 2
            assert (runtime.snapshot.revision, runtime.snapshot.apply_status, runtime.snapshot.applied_revision) == (
                2,
                "applied",
                2,
            )
            assert runtime.cycle is candidate_cycle
            assert runtime.sessions == {"paper-a": candidate_session}
            assert initial_session.close_calls == 1
            raise RuntimeError("private final write failure")
        return await original_mark_applied(revision)

    monkeypatch.setattr(repository, "mark_applied", fail_final_commit_once)
    async with runtime.application_barrier():
        pending = await repository.replace(first.revision, failed_document)
        candidate = await runtime.prepare_candidate(pending)
        candidate_session = candidate.sessions["paper-a"]
        candidate_cycle = candidate.cycle
        with pytest.raises(Exception, match="Runtime configuration cannot be applied"):
            await publish_pending_snapshot(runtime, pending, None, candidate)

    failed = await repository.get_or_create()
    assert (failed.revision, failed.apply_status, failed.applied_revision, failed.apply_error) == (
        2,
        "failed",
        1,
        "runtime application failed",
    )
    assert runtime.sessions == {}
    assert runtime.cycle is None
    assert initial_session.close_calls == 1
    assert candidate_session.close_calls == 1
    async with runtime.application_barrier():
        assert runtime.application_in_progress is True

    recovered_document = _document(_connection("paper-a", parameters={"initial_equity": "30000"}))
    async with runtime.application_barrier():
        pending = await repository.replace(failed.revision, recovered_document)
        candidate = await runtime.prepare_candidate(pending)
        applied = await publish_pending_snapshot(runtime, pending, None, candidate)

    stored = await repository.get_or_create()
    assert (applied.revision, stored.apply_status, stored.applied_revision) == (3, "applied", 3)
    assert runtime.cycle is not None
    assert set(runtime.sessions) == {"paper-a"}
    recovered_session = runtime.sessions["paper-a"]
    assert recovered_session.close_calls == 0
    async with runtime.cycle_lease() as cycle:
        assert cycle.snapshot.revision == 3
    await runtime.close()
    assert recovered_session.close_calls == 1


@pytest.mark.asyncio
async def test_application_barrier_releases_every_owner_marker_after_repeated_cancellation():
    """A second cancellation while cleanup waits for the lifecycle lock cannot strand mutation admission."""
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    entered = asyncio.Event()

    async def hold_barrier():
        async with runtime.application_barrier():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_barrier())
    await entered.wait()
    await runtime._lifecycle_lock.acquire()
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
    finally:
        runtime._lifecycle_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime.application_in_progress is False
    async with runtime.application_barrier():
        assert runtime.application_in_progress is True
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_after", ["publish", "owners", "mark_applied"])
async def test_application_cancellation_after_cas_fails_closed_and_rethrows_original_cancellation(
    monkeypatch, cancel_after
):
    """Cancelling any post-CAS application step must persist failed state and retire executable resources."""
    from api.routes.config import publish_pending_snapshot

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    pending = RuntimeConfigSnapshot(
        8,
        _document(_connection("paper-a", parameters={"initial_equity": "20000"})),
        NOW + timedelta(seconds=1),
        "pending",
        7,
    )
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)

    async def cancel_here(*args, **kwargs):
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    if cancel_after == "publish":
        original = runtime.publish_candidate

        async def publish_then_cancel(*args, **kwargs):
            await original(*args, **kwargs)
            await cancel_here()

        monkeypatch.setattr(runtime, "publish_candidate", publish_then_cancel)
        refresh = None
    elif cancel_after == "owners":

        async def refresh(_snapshot):
            await cancel_here()

    else:
        original_mark_applied = repository.mark_applied

        async def mark_then_cancel(*args, **kwargs):
            await cancel_here()
            return await original_mark_applied(*args, **kwargs)

        monkeypatch.setattr(repository, "mark_applied", mark_then_cancel)
        refresh = None

    async with runtime.application_barrier():
        with pytest.raises(asyncio.CancelledError):
            await publish_pending_snapshot(runtime, pending, refresh, candidate)

    assert repository.snapshot.apply_status == "failed"
    assert runtime.snapshot.apply_status == "failed"
    assert runtime.cycle is None
    assert runtime.sessions == {}
    assert runtime.application_in_progress is False
    await runtime.close()


@pytest.mark.asyncio
async def test_clear_runtime_owners_stops_trigger_and_clears_references_after_scheduler_shutdown_failure(monkeypatch):
    """A scheduler teardown error cannot leave a live trigger behind on a failed revision."""
    from api import main as api_main

    stopped: list[str] = []

    class _Trigger:
        async def stop(self):
            stopped.append("trigger")

    app = SimpleNamespace(
        state=SimpleNamespace(
            scheduler=object(), scheduler_task=object(), trigger_engine=_Trigger(), trigger_store=object()
        )
    )

    async def fail_scheduler(_app):
        raise RuntimeError("scheduler stop failed")

    monkeypatch.setattr(api_main, "_shutdown_scheduler", fail_scheduler)

    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await api_main._clear_runtime_owners(app)

    assert stopped == ["trigger"]
    assert app.state.scheduler is None
    assert app.state.scheduler_task is None
    assert app.state.trigger_engine is None
    assert app.state.trigger_store is None


@pytest.mark.asyncio
async def test_failed_application_records_sanitized_cleanup_state_after_scheduler_shutdown_failure(monkeypatch):
    """A partial owner cleanup makes the desired revision failed, never silently applied."""
    from api import main as api_main
    from api.routes.config import publish_pending_snapshot

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    pending = RuntimeConfigSnapshot(
        8,
        _document(_connection("paper-a", parameters={"initial_equity": "20000"})),
        NOW + timedelta(seconds=1),
        "pending",
        7,
    )
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)
    stopped: list[str] = []

    class _Trigger:
        async def stop(self):
            stopped.append("trigger")

    app = SimpleNamespace(
        state=SimpleNamespace(
            runtime=runtime,
            scheduler=object(),
            scheduler_task=object(),
            trigger_engine=_Trigger(),
            trigger_store=object(),
        )
    )

    async def fail_scheduler(_app):
        raise RuntimeError("private scheduler failure")

    async def fail_owner_start(_snapshot):
        raise RuntimeError("private owner failure")

    monkeypatch.setattr(api_main, "_shutdown_scheduler", fail_scheduler)
    async with runtime.application_barrier():
        with pytest.raises(Exception, match="Runtime configuration cannot be applied"):
            await publish_pending_snapshot(
                runtime,
                pending,
                fail_owner_start,
                candidate,
                clear_owners=lambda: api_main._clear_runtime_owners(app),
            )

    assert stopped == ["trigger"]
    assert repository.snapshot.apply_status == "failed"
    assert repository.snapshot.apply_error == "runtime application failed: cleanup incomplete"
    assert runtime.sessions == {}
    assert runtime.cycle is None
    await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_application_rethrows_original_cancellation_when_failed_state_write_also_fails(monkeypatch):
    """A failed-state persistence error cannot replace the request's original cancellation or retain the barrier."""
    from api.routes.config import publish_pending_snapshot

    runtime, repository, _, _ = await _build(_document(_connection("paper-a")))
    pending = RuntimeConfigSnapshot(
        8,
        _document(_connection("paper-a", parameters={"initial_equity": "20000"})),
        NOW + timedelta(seconds=1),
        "pending",
        7,
    )
    repository.snapshot = pending
    candidate = await runtime.prepare_candidate(pending)

    async def cancel_owner_start(_snapshot):
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    async def reject_failed_state(*_args):
        raise RuntimeError("private persistence failure")

    monkeypatch.setattr(repository, "mark_failed", reject_failed_state)
    async with runtime.application_barrier():
        with pytest.raises(asyncio.CancelledError):
            await publish_pending_snapshot(runtime, pending, cancel_owner_start, candidate)

    assert runtime.snapshot.apply_status == "failed"
    assert runtime.cycle is None
    assert runtime.sessions == {}
    async with runtime.application_barrier():
        assert runtime.application_in_progress is True
    await runtime.close()


@pytest.mark.asyncio
async def test_active_replacement_close_failure_keeps_published_cycle_and_retries_retired_session(caplog):
    connection = _connection("paper-a")
    runtime, repository, adapter, _ = await _build(_document(connection))
    retired = runtime.sessions[connection.id]
    retired.close_error = RuntimeError("private-retired-close-error")
    repository.publish(_document(replace(connection, parameters={"initial_equity": "20000"})))

    cycle = await runtime.reload_for_cycle()

    assert cycle is runtime.cycle
    assert runtime.snapshot.revision == 8
    assert runtime.sessions[connection.id] is not retired
    assert retired.close_calls == 1
    assert "retired venue session cleanup pending" in caplog.text
    assert "private-retired-close-error" not in caplog.text
    retired.close_error = None

    next_cycle = await runtime.reload_for_cycle()

    assert next_cycle is runtime.cycle
    assert retired.close_calls == 2
    assert len(adapter.connect_calls) == 2


@pytest.mark.asyncio
async def test_inactive_reload_close_failure_returns_none_and_retries_retired_session():
    connection = _connection("paper-a")
    runtime, repository, _, _ = await _build(_document(connection))
    retired = runtime.sessions[connection.id]
    retired.close_error = RuntimeError("private-retired-close-error")
    saved = repository.publish(_document(connection, active=False))

    cycle = await runtime.reload_for_cycle()

    assert cycle is None
    assert runtime.snapshot is saved
    assert runtime.sessions == {}
    assert runtime.cycle is None
    assert retired.close_calls == 1
    retired.close_error = None

    await runtime.close()

    assert retired.close_calls == 2
    with pytest.raises(RuntimeError, match=r"^runtime is closed$"):
        await runtime.reload_for_cycle()


@pytest.mark.asyncio
async def test_setup_runtime_activates_after_configuration_write():
    setup = _document(active=False)
    runtime, repository, adapter, _ = await _build(setup)
    repository.publish(_document(_connection("paper-a")))

    cycle = await runtime.reload_for_cycle()

    assert cycle is runtime.cycle
    assert runtime.snapshot.setup_required is False
    assert set(runtime.sessions) == {"paper-a"}
    assert [connection.id for connection, _credentials in adapter.connect_calls] == ["paper-a"]
    await runtime.close()


@pytest.mark.asyncio
async def test_candidate_open_failure_closes_new_sessions_and_preserves_old_graph():
    first = _connection("paper-a")
    second = _connection("paper-b")
    runtime, repository, adapter, _ = await _build(_document(first, second))
    old_snapshot = runtime.snapshot
    old_cycle = runtime.cycle
    old_sessions = dict(runtime.sessions)
    repository.publish(
        _document(
            replace(first, parameters={"initial_equity": "20000"}),
            replace(second, parameters={"initial_equity": "20000"}),
        )
    )
    adapter.fail_connection_id = second.id

    with pytest.raises(RuntimeError, match=r"^failed to open venue session$") as error:
        await runtime.reload_for_cycle()

    assert "sensitive-adapter-failure" not in str(error.value)
    assert runtime.snapshot is old_snapshot
    assert runtime.cycle is old_cycle
    assert runtime.sessions == old_sessions
    assert all(session.close_calls == 0 for session in old_sessions.values())
    assert adapter.sessions[-1].connection_id == first.id
    assert adapter.sessions[-1].close_calls == 1


@pytest.mark.asyncio
async def test_candidate_graph_failure_closes_new_session_and_preserves_old_graph():
    first = _connection("paper-a")
    markets = _MarketRegistry()
    runtime, repository, adapter, _ = await _build(_document(first), markets=markets)
    old_snapshot = runtime.snapshot
    old_cycle = runtime.cycle
    old_session = runtime.sessions[first.id]
    changed = replace(first, parameters={"initial_equity": "20000"})
    repository.publish(_document(changed, market_source="candidate"))
    markets.fail_source = "candidate"

    with pytest.raises(RuntimeError, match="candidate graph failed"):
        await runtime.reload_for_cycle()

    assert runtime.snapshot is old_snapshot
    assert runtime.cycle is old_cycle
    assert runtime.sessions[first.id] is old_session
    assert old_session.close_calls == 0
    assert adapter.sessions[-1].close_calls == 1


@pytest.mark.asyncio
async def test_close_attempts_each_unique_session_once_and_uses_safe_error():
    shared = _Session("shared", close_error=RuntimeError("private-close-detail"))
    other = _Session("other")
    runtime, _, _, _ = await _build(_document(active=False))
    runtime.sessions = {"first": shared, "second": shared, "other": other}

    with pytest.raises(RuntimeError, match=r"^failed to close venue session$") as error:
        await runtime.close()

    assert "private-close-detail" not in str(error.value)
    assert shared.close_calls == 1
    assert other.close_calls == 1
    shared.close_error = None
    await runtime.close()
    assert shared.close_calls == 2
    assert other.close_calls == 1


@pytest.mark.asyncio
async def test_close_propagates_cancellation_and_still_closes_other_sessions():
    adapter = _Adapter()
    runtime, _, _, _ = await _build(_document(_connection("paper-a"), _connection("paper-b")), adapter=adapter)
    sessions = tuple(runtime.sessions.values())
    sessions[0].close_error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await runtime.close()

    assert all(session.close_calls == 1 for session in sessions)


@pytest.mark.asyncio
async def test_external_close_cancellation_waits_for_all_unique_sessions_before_propagating():
    runtime, _, _, _ = await _build(_document(active=False))
    first = _BlockingCloseSession("first")
    second = _BlockingCloseSession("second")
    runtime.sessions = {"first": first, "first-alias": first, "second": second}

    close_task = asyncio.create_task(runtime.close())
    await asyncio.gather(first.started.wait(), second.started.wait())
    close_task.cancel()
    await asyncio.sleep(0)

    assert close_task.done() is False

    first.release.set()
    second.release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert first.close_calls == 1
    assert second.close_calls == 1
    assert first.completed is True
    assert second.completed is True
    with pytest.raises(RuntimeError, match=r"^runtime is closed$"):
        await runtime.reload_for_cycle()


@pytest.mark.asyncio
async def test_reload_after_close_is_rejected():
    runtime, _, _, _ = await _build(_document(_connection("paper-a")))
    await runtime.close()

    with pytest.raises(RuntimeError, match=r"^runtime is closed$"):
        await runtime.reload_for_cycle()


@pytest.mark.asyncio
async def test_candidate_cancellation_closes_new_sessions_and_preserves_old_graph():
    first = _connection("paper-a")
    second = _connection("paper-b")
    runtime, repository, adapter, _ = await _build(_document(first, second))
    old_snapshot = runtime.snapshot
    old_cycle = runtime.cycle
    old_sessions = dict(runtime.sessions)
    repository.publish(
        _document(
            replace(first, parameters={"initial_equity": "20000"}),
            replace(second, parameters={"initial_equity": "20000"}),
        )
    )
    adapter.fail_connection_id = second.id
    adapter.fail_exception = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await runtime.reload_for_cycle()

    assert runtime.snapshot is old_snapshot
    assert runtime.cycle is old_cycle
    assert runtime.sessions == old_sessions
    assert all(session.close_calls == 0 for session in old_sessions.values())
    assert adapter.sessions[-1].connection_id == first.id
    assert adapter.sessions[-1].close_calls == 1


@pytest.mark.asyncio
async def test_concurrent_reloads_are_serialized_and_publish_complete_graphs():
    connection = _connection("paper-a")
    adapter = _BlockingAdapter()
    runtime, repository, _, _ = await _build(_document(connection), adapter=adapter)
    repository.publish(_document(replace(connection, parameters={"initial_equity": "20000"})))
    adapter.block = True

    first_reload = asyncio.create_task(runtime.reload_for_cycle())
    await adapter.entered.wait()
    repository.publish(_document(replace(connection, parameters={"initial_equity": "30000"})))
    second_reload = asyncio.create_task(runtime.reload_for_cycle())
    await asyncio.sleep(0)

    assert adapter.max_active_connects == 1
    assert adapter.active_connects == 1
    assert len(adapter.connect_calls) == 1

    adapter.release.set()
    first_cycle, second_cycle = await asyncio.gather(first_reload, second_reload)

    assert first_cycle is not second_cycle
    assert second_cycle is runtime.cycle
    assert runtime.snapshot.revision == 9
    assert set(runtime.sessions) == {"paper-a"}
    assert adapter.max_active_connects == 1


class _BlockingAdapter(_Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active_connects = 0
        self.max_active_connects = 0

    async def connect(self, connection, credentials):
        if self.block:
            self.active_connects += 1
            self.max_active_connects = max(self.max_active_connects, self.active_connects)
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.active_connects -= 1
        return await super().connect(connection, credentials)


class _Credential:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __repr__(self) -> str:
        return self.marker


class _BlockingCloseSession:
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_calls = 0
        self.completed = False

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        self.completed = True
