"""Database-runtime staging gate contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from staging_validate import (
    _check_enabled_connections,
    _check_runtime_health,
    _run_gate,
    main,
    run_step,
)


class _Runtime:
    def __init__(self, document=None, repository=None, venue_registry=None) -> None:
        self.snapshot = SimpleNamespace(document=document)
        self.repository = repository
        self.venue_registry = venue_registry
        self.closed = 0

    async def close(self):
        self.closed += 1


class _RecordingSession:
    def __init__(self, connection_id: str, *, close_error: Exception | None = None) -> None:
        self.connection_id = connection_id
        self._close_error = close_error
        self.capability_reads = 0
        self.close_calls = 0
        self.mutation_calls = 0

    @property
    def capabilities(self):
        self.capability_reads += 1
        return object()

    async def close(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error

    async def place_order(self, *_args, **_kwargs):
        self.mutation_calls += 1
        raise AssertionError("staging validation must not place orders")

    async def replace_protection(self, *_args, **_kwargs):
        self.mutation_calls += 1
        raise AssertionError("staging validation must not create protection")

    async def cancel_protection(self, *_args, **_kwargs):
        self.mutation_calls += 1
        raise AssertionError("staging validation must not cancel protection")


class _RecordingAdapter:
    def __init__(
        self,
        adapter_id: str,
        *,
        events: list[str] | None = None,
        failure: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.adapter_id = adapter_id
        self.events = events if events is not None else []
        self.failure = failure
        self.close_error = close_error
        self.connect_calls: list[str] = []
        self.sessions: list[_RecordingSession] = []

    async def connect(self, connection, _credentials):
        self.connect_calls.append(connection.id)
        self.events.append(connection.id)
        if self.failure is not None:
            raise self.failure
        session = _RecordingSession(connection.id, close_error=self.close_error)
        self.sessions.append(session)
        return session


class _RecordingRegistry:
    def __init__(self, adapters: dict[str, _RecordingAdapter]) -> None:
        self.adapters = adapters

    def require(self, adapter_id: str):
        return self.adapters[adapter_id]

    def installed_ids(self):
        return frozenset(self.adapters)


class _NoCredentialRepository:
    async def credential_state(self, _credential_ref):
        return SimpleNamespace(configured=True)

    async def reveal_credentials(self, _credential_ref):
        return object()


def _connection(connection_id: str, adapter_id: str, *, enabled: bool = True):
    from tests.factories.runtime_config import connection

    return connection(connection_id, adapter_id=adapter_id, enabled=enabled)


def _staging_runtime(*connections, adapters: dict[str, _RecordingAdapter]) -> _Runtime:
    from tests.factories.runtime_config import runtime_document

    return _Runtime(
        runtime_document(connections=connections),
        _NoCredentialRepository(),
        _RecordingRegistry(adapters),
    )


def test_run_step_reports_failure_without_stopping_the_gate():
    result = run_step(2, "runtime health", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert result.status == "FAIL"
    assert result.error == "down"
    assert result.fmt().startswith("[step 2] runtime health: FAIL")


def test_main_runs_schema_runtime_then_each_enabled_connection(monkeypatch):
    calls: list[str] = []

    async def schema():
        calls.append("schema")
        return _Runtime()

    async def runtime(_runtime):
        calls.append("runtime")

    async def connections(_runtime):
        calls.append("connections")

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", runtime)
    monkeypatch.setattr("staging_validate._check_enabled_connections", connections)

    assert main() == 0
    assert calls == ["schema", "runtime", "connections"]


def test_main_fails_when_no_active_runtime_configuration(monkeypatch):
    async def missing_config():
        raise RuntimeError("runtime configuration is not active")

    monkeypatch.setattr("staging_validate._load_runtime_config", missing_config)

    assert main() == 1


def test_main_fails_when_an_enabled_connection_is_unhealthy(monkeypatch):
    async def schema():
        return _Runtime()

    async def runtime(_runtime):
        return None

    async def unhealthy(_runtime):
        raise RuntimeError("connection demo-okx is unhealthy")

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", runtime)
    monkeypatch.setattr("staging_validate._check_enabled_connections", unhealthy)

    assert main() == 1


def test_main_prints_a_three_step_summary(monkeypatch, capsys):
    async def schema():
        return _Runtime()

    async def nothing(_runtime):
        return None

    monkeypatch.setattr("staging_validate._load_runtime_config", schema)
    monkeypatch.setattr("staging_validate._check_runtime_health", nothing)
    monkeypatch.setattr("staging_validate._check_enabled_connections", nothing)

    assert main() == 0
    output = capsys.readouterr().out
    assert "[step 1] database schema and config revision: PASS" in output
    assert "[step 2] runtime health: PASS" in output
    assert "[step 3] enabled connection health: PASS" in output


async def test_runtime_health_discovers_registries_without_connecting(monkeypatch):
    from tests.factories.runtime_config import active_document

    calls = 0
    adapter = _RecordingAdapter("paper")

    async def discover(_document, _events, _repository):
        nonlocal calls
        calls += 1
        return (
            SimpleNamespace(installed_ids=lambda: frozenset({"kronos", "llm_committee"})),
            _RecordingRegistry({"paper": adapter}),
            SimpleNamespace(installed_ids=lambda: frozenset({"default"})),
        )

    runtime = _Runtime(active_document(), _NoCredentialRepository())
    monkeypatch.setattr("cryptotrader.runtime._discover_registry_graph", discover)

    await _check_runtime_health(runtime)

    assert calls == 1
    assert runtime.venue_registry.installed_ids() == frozenset({"paper"})
    assert adapter.connect_calls == []


async def test_enabled_connections_are_opened_once_in_config_order_and_never_mutated():
    events: list[str] = []
    first = _RecordingAdapter("first", events=events)
    second = _RecordingAdapter("second", events=events)
    disabled = _RecordingAdapter("disabled", events=events)
    runtime = _staging_runtime(
        _connection("first", "first"),
        _connection("disabled", "disabled", enabled=False),
        _connection("second", "second"),
        adapters={"first": first, "disabled": disabled, "second": second},
    )

    await _check_enabled_connections(runtime)

    assert events == ["first", "second"]
    assert first.connect_calls == ["first"]
    assert second.connect_calls == ["second"]
    assert disabled.connect_calls == []
    for adapter in (first, second):
        assert [session.capability_reads for session in adapter.sessions] == [1]
        assert [session.close_calls for session in adapter.sessions] == [1]
        assert [session.mutation_calls for session in adapter.sessions] == [0]


async def test_connection_failure_stops_later_connects_and_closes_prior_session(monkeypatch):
    first = _RecordingAdapter("first")
    second = _RecordingAdapter("second", failure=ValueError("credential=secret-value"))
    third = _RecordingAdapter("third")
    runtime = _staging_runtime(
        _connection("first", "first"),
        _connection("second", "second"),
        _connection("third", "third"),
        adapters={"first": first, "second": second, "third": third},
    )

    async def load_runtime():
        return runtime

    async def healthy_runtime(_runtime):
        return None

    monkeypatch.setattr("staging_validate._load_runtime_config", load_runtime)
    monkeypatch.setattr("staging_validate._check_runtime_health", healthy_runtime)

    results = await _run_gate()

    assert [(result.idx, result.status) for result in results] == [(1, "PASS"), (2, "PASS"), (3, "FAIL")]
    assert first.connect_calls == ["first"]
    assert [session.close_calls for session in first.sessions] == [1]
    assert second.connect_calls == ["second"]
    assert second.sessions == []
    assert third.connect_calls == []
    assert runtime.closed == 1
    assert "secret-value" not in results[-1].error


async def test_close_failure_fails_connection_step_and_preserves_outer_cleanup(monkeypatch):
    adapter = _RecordingAdapter("one", close_error=ValueError("credential=secret-value"))
    runtime = _staging_runtime(_connection("one", "one"), adapters={"one": adapter})

    async def load_runtime():
        return runtime

    async def healthy_runtime(_runtime):
        return None

    monkeypatch.setattr("staging_validate._load_runtime_config", load_runtime)
    monkeypatch.setattr("staging_validate._check_runtime_health", healthy_runtime)

    results = await _run_gate()

    assert [(result.idx, result.status) for result in results] == [(1, "PASS"), (2, "PASS"), (3, "FAIL")]
    assert adapter.connect_calls == ["one"]
    assert [session.close_calls for session in adapter.sessions] == [1]
    assert runtime.closed == 1
    assert results[-1].error == "connection one did not close cleanly"
    assert "secret-value" not in results[-1].error


async def test_loaded_staging_runtime_has_total_cleanup_after_all_steps(monkeypatch, tmp_path):
    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault
    from tests.factories.runtime_config import active_document

    url = f"sqlite+aiosqlite:///{tmp_path / 'staging.db'}"
    key = "c3Rha2luZy1nYXRlLW1hc3Rlci1rZXktMzItYnl0ZXM="
    repository = RuntimeConfigRepository(url, CredentialVault(key))
    initial = await repository.get_or_create()
    await repository.replace(initial.revision, active_document())
    monkeypatch.setattr(
        BootstrapSettings,
        "from_environment",
        classmethod(lambda _cls: SimpleNamespace(database_url=url, config_master_key=key)),
    )

    async def healthy_runtime(_runtime):
        return None

    async def healthy_connections(_runtime):
        return None

    monkeypatch.setattr("staging_validate._check_runtime_health", healthy_runtime)
    monkeypatch.setattr("staging_validate._check_enabled_connections", healthy_connections)

    results = await _run_gate()

    assert [result.status for result in results] == ["PASS", "PASS", "PASS"]


@pytest.mark.parametrize("name", ["DATABASE_URL", "CONFIG_MASTER_KEY"])
def test_staging_gate_names_only_bootstrap_environment(name):
    source = (Path(__file__).parent.parent / "scripts" / "staging_validate.py").read_text()
    assert name in source
    assert "load_dotenv" not in source
