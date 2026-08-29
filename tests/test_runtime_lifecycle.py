"""Runtime graph 的热同步、session 复用与关闭边界。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.runtime import RuntimeLeaseUnavailableError, build_runtime
from cryptotrader.runtime_config.models import ExecutionConfig, RuntimeConfigSnapshot, SystemConfig
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

    def publish(self, document) -> RuntimeConfigSnapshot:
        self.snapshot = RuntimeConfigSnapshot(
            self.snapshot.revision + 1,
            document,
            self.snapshot.updated_at + timedelta(seconds=1),
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
        market_data=MarketDataConfig(source_id=market_source),
    )


async def _build(document, *, adapter=None, repository=None, markets=None):
    snapshot = RuntimeConfigSnapshot(7, document, NOW)
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
