"""数据库 RuntimeConfig API 的 CAS、setup 与严格脱敏契约。"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.hitl.store import BookApprovalStore
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.pair import Pair
from cryptotrader.portfolio.aggregator import PortfolioAggregator
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    InfrastructureConfig,
    RuntimeConfigDocument,
    SignalComponentConfig,
)
from cryptotrader.runtime_config.repository import RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.venues.models import ConnectionPosition, VenueCapabilities, VenueConnection
from cryptotrader.venues.registry import VenueAdapterRegistry
from tests.factories.runtime_config import market_config, signal_config

PAIR = Pair.parse("BTC/USDT:USDT")
MASTER_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _connection(
    connection_id: str,
    environment: str,
    adapter_id: str,
    *,
    credential_ref: str | None,
) -> VenueConnection:
    return VenueConnection(
        id=connection_id,
        label=connection_id,
        adapter_id=adapter_id,
        environment=environment,
        enabled=True,
        credential_ref=credential_ref,
        leverage=1,
        margin_mode="cross" if adapter_id == "paper" else "isolated",
        canary_only=False,
        parameters={},
    )


def active_document() -> RuntimeConfigDocument:
    connections = (
        _connection("okx-demo", "demo", "okx", credential_ref="okx-demo-credentials"),
        _connection("bybit-testnet", "testnet", "bybit", credential_ref="bybit-testnet-credentials"),
        _connection("okx-live", "live", "okx", credential_ref="okx-live-credentials"),
        _connection("paper-spare", "paper", "paper", credential_ref=None),
    )
    books = (
        ExecutionBook(
            "simulation",
            "Simulation",
            "simulated",
            True,
            False,
            (
                ConnectionAllocation("okx-demo", True, 0.5),
                ConnectionAllocation("bybit-testnet", True, 0.5),
            ),
        ),
        ExecutionBook(
            "production",
            "Production",
            "real",
            True,
            True,
            (ConnectionAllocation("okx-live", True, 1.0),),
        ),
    )
    return RuntimeConfigDocument(
        market_data=market_config(),
        signals=signal_config(components=(SignalComponentConfig(component_id="kronos", enabled=True, weight=1.0),)),
        execution=ExecutionConfig(connections=connections, books=books),
        infrastructure=InfrastructureConfig(redis_url="redis://runtime-test:6379/0"),
    )


def active_payload() -> dict:
    payload = active_document().model_dump(mode="json")
    for connection in payload["execution"]["connections"]:
        connection.pop("credential_ref")
    return payload


def _portfolio(connection_id: str, equity: str, notional: str) -> ConnectionPortfolioSnapshot:
    return ConnectionPortfolioSnapshot(
        connection_id,
        Decimal(equity),
        {"USDT": Decimal(equity)},
        ConnectionPosition(PAIR, Decimal("0.01"), Decimal(notional), Decimal("50000")),
    )


class FakeSession:
    def __init__(self, snapshot: ConnectionPortfolioSnapshot) -> None:
        self.connection_id = snapshot.connection_id
        self.snapshot = snapshot
        self._capabilities = VenueCapabilities(
            frozenset({"swap"}),
            True,
            False,
            True,
            frozenset({"market"}),
        )
        self.capabilities_error: Exception | None = None
        self.check_error: Exception | None = None
        self.check_calls = 0
        self.closed = 0
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.close_release: asyncio.Event | None = None
        self.order_calls = 0

    @property
    def capabilities(self) -> VenueCapabilities:
        if self.capabilities_error is not None:
            raise self.capabilities_error
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value: VenueCapabilities) -> None:
        self._capabilities = value

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        return self.snapshot

    async def fetch_account(self):
        from tests.fakes.account_session import account_from_portfolio

        return account_from_portfolio(self.snapshot)

    async def check_connection(self) -> None:
        self.check_calls += 1
        if self.check_error is not None:
            raise self.check_error

    async def close(self) -> None:
        self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        self.closed += 1
        self.close_finished.set()


class FakeAdapter:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.connect_calls: list[tuple[VenueConnection, object]] = []
        self.opened_sessions: list[FakeSession] = []
        self.error: Exception | None = None
        self.session_capabilities_error: Exception | None = None
        self.session_check_error: Exception | None = None
        self.block_close = False
        self.session_opened = asyncio.Event()

    def capabilities(self, environment):
        return VenueCapabilities(
            frozenset({"swap"}),
            True,
            False,
            True,
            frozenset({"limit", "market"}),
        )

    async def connect(self, connection, credentials):
        self.connect_calls.append((connection, credentials))
        if self.error is not None:
            raise self.error
        session = FakeSession(_portfolio(connection.id, "1", "0"))
        session.capabilities = self.capabilities(connection.environment)
        session.capabilities_error = self.session_capabilities_error
        session.check_error = self.session_check_error
        session.close_release = asyncio.Event() if self.block_close else None
        self.opened_sessions.append(session)
        self.session_opened.set()
        return session


@dataclass
class ApiHarness:
    client: httpx.AsyncClient
    runtime: SimpleNamespace
    adapters: dict[str, FakeAdapter]


@pytest.fixture
async def api_harness(tmp_path, monkeypatch):
    from api.main import app
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    # A prior lifespan may leave callbacks bound to unrelated scheduler/trigger owners.
    monkeypatch.setattr(app.state, "refresh_runtime_owners", None, raising=False)
    monkeypatch.setattr(app.state, "clear_runtime_owners", None, raising=False)
    monkeypatch.setattr(app.state, "account_sync_owner", None, raising=False)

    document = active_document()
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'runtime-api.db'}"
    await migrate_workbench_schema(database_url)
    repository = RuntimeConfigRepository(
        database_url,
        CredentialVault(MASTER_KEY),
        default_factory=lambda: document,
    )
    snapshot = await repository.get_or_create()
    adapters = {adapter_id: FakeAdapter(adapter_id) for adapter_id in ("paper", "okx", "bybit")}
    sessions = {
        "okx-demo": FakeSession(_portfolio("okx-demo", "100", "20")),
        "bybit-testnet": FakeSession(_portfolio("bybit-testnet", "300", "-10")),
        "okx-live": FakeSession(_portfolio("okx-live", "1000", "250")),
    }
    cycle = SimpleNamespace(
        portfolios=PortfolioAggregator(),
        approvals=BookApprovalStore(),
        journal=MultiVenueCycleStore(),
    )

    application_lock = asyncio.Lock()

    @asynccontextmanager
    async def application_barrier():
        async with application_lock:
            runtime.application_in_progress = True
            try:
                yield
            finally:
                runtime.application_in_progress = False

    async def activate_applied(applied):
        runtime.snapshot = applied

    runtime = SimpleNamespace(
        snapshot=snapshot,
        repository=repository,
        cycle=cycle,
        sessions=sessions,
        signal_registry=SimpleNamespace(registered_ids=lambda: frozenset({"kronos", "llm_committee"})),
        market_registry=SimpleNamespace(registered_ids=lambda: frozenset({"default"})),
        venue_registry=VenueAdapterRegistry(tuple(adapters.values())),
        reload_for_cycle=AsyncMock(),
        application_barrier=application_barrier,
        prepare_candidate=AsyncMock(return_value=SimpleNamespace(close=AsyncMock())),
        publish_candidate=AsyncMock(),
        activate_applied=AsyncMock(side_effect=activate_applied),
        fail_closed=AsyncMock(),
        application_in_progress=False,
    )
    from cryptotrader.decision.read_service import DecisionReadService

    runtime.read_service = DecisionReadService(cycle.journal)
    previous = getattr(app.state, "runtime", None)
    app.state.runtime = runtime
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield ApiHarness(client, runtime, adapters)
    app.state.runtime = previous


@pytest.mark.parametrize("publication_fails", [False, True])
async def test_api_harness_isolates_and_restores_previous_runtime_owners(tmp_path, monkeypatch, publication_fails):
    from api.main import app

    refresh = AsyncMock(side_effect=AssertionError("previous runtime refresh must not run"))
    clear = AsyncMock(side_effect=AssertionError("previous runtime clear must not run"))
    monkeypatch.setattr(app.state, "refresh_runtime_owners", refresh, raising=False)
    monkeypatch.setattr(app.state, "clear_runtime_owners", clear, raising=False)

    # Enter the real fixture after installing offline sentinels, with its own teardown scope.
    with pytest.MonkeyPatch.context() as fixture_patch:
        async with asynccontextmanager(api_harness.__wrapped__)(tmp_path, fixture_patch) as harness:
            if publication_fails:
                harness.runtime.publish_candidate.side_effect = RuntimeError("offline publication failure")
            document = active_payload()
            document["scheduler"]["enabled"] = True
            document["triggers"]["enabled"] = True
            response = await harness.client.put("/api/config", json={"expected_revision": 1, "document": document})
            assert response.status_code == (503 if publication_fails else 200), response.text
            saved = await harness.runtime.repository.get_or_create()
            assert saved.revision == 2
            assert saved.apply_status == ("failed" if publication_fails else "applied")

    assert app.state.refresh_runtime_owners is refresh
    assert app.state.clear_runtime_owners is clear
    refresh.assert_not_awaited()
    clear.assert_not_awaited()


async def put_fixture_credentials(
    harness: ApiHarness,
    connection_id: str,
    *,
    marker: str = "credential-value",
) -> httpx.Response:
    current = await harness.client.get("/api/config")
    return await harness.client.put(
        f"/api/venue-connections/{connection_id}/credentials",
        json={
            "expected_revision": current.json()["revision"],
            "values": {
                "api_key": marker,
                "secret": marker,
                **({"passphrase": marker} if connection_id.startswith("okx") else {}),
            },
        },
    )


async def test_get_config_exposes_latest_snapshot_and_credential_state(api_harness):
    response = await api_harness.client.get("/api/config")

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 1
    assert body["apply_status"] == "applied"
    assert body["document"]["execution"]["connections"][0]["credential_configured"] is False
    assert body["document"]["execution"]["connections"][0]["credential_updated_at"] is None
    assert "credential_ref" not in json.dumps(body)


async def test_runtime_secret_endpoints_are_cas_protected_and_never_echo_tokens(api_harness):
    current = await api_harness.client.get("/api/config")
    revision = current.json()["revision"]

    saved = await api_harness.client.put(
        "/api/config/credentials/llm-gateway",
        json={"expected_revision": revision, "token": "gateway-secret-value"},
    )

    assert saved.status_code == 200
    assert saved.json()["revision"] == revision + 1
    assert saved.json()["configured"] is True
    assert "gateway-secret-value" not in saved.text
    stale = await api_harness.client.put(
        "/api/config/credentials/llm-gateway",
        json={"expected_revision": revision, "token": "another-gateway-secret"},
    )
    assert stale.status_code == 409
    visible = await api_harness.client.get("/api/config")
    assert visible.json()["document"]["llm"]["gateway_credential_configured"] is True
    assert "gateway-secret-value" not in visible.text


async def test_put_config_requires_expected_revision(api_harness):
    current = await api_harness.client.get("/api/config")
    saved = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": active_payload()},
    )
    assert saved.status_code == 200

    stale = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": active_payload()},
    )
    assert stale.status_code == 409
    assert stale.json() == {"detail": "Runtime configuration changed; reload and retry"}


@pytest.mark.parametrize("automation_enabled", [False, True])
async def test_ordinary_put_preserves_automation_in_candidate_and_saved_document(api_harness, automation_enabled):
    repository = api_harness.runtime.repository
    initial = await repository.get_or_create()
    document = initial.document.model_copy(
        update={"scheduler": initial.document.scheduler.model_copy(update={"automation_enabled": automation_enabled})}
    )
    await repository.replace(initial.revision, document)
    current = await repository.mark_applied(2)
    api_harness.runtime.snapshot = current
    payload = active_payload()
    payload["scheduler"].update(automation_enabled=not automation_enabled, interval_minutes=60, enabled=True)
    payload["triggers"]["enabled"] = True

    response = await api_harness.client.put("/api/config", json={"expected_revision": 2, "document": payload})

    assert response.status_code == 200, response.text
    assert response.json()["document"]["scheduler"]["automation_enabled"] is automation_enabled
    saved = await repository.get_or_create()
    prepared = api_harness.runtime.prepare_candidate.call_args.args[0]
    assert prepared.document == saved.document == api_harness.runtime.snapshot.document
    assert saved.document.scheduler.automation_enabled is automation_enabled
    assert saved.document.scheduler.interval_minutes == 60
    assert saved.document.scheduler.enabled is True
    assert saved.document.triggers.enabled is True
    assert saved.document.execution.live_order_execution_enabled is False
    assert [book.hitl_required for book in saved.document.execution.books] == [False, True]
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())

    stale = await api_harness.client.put("/api/config", json={"expected_revision": 2, "document": payload})
    assert stale.status_code == 409
    assert api_harness.runtime.prepare_candidate.await_count == 1
    assert (await repository.get_or_create()).revision == 3


async def test_put_config_publishes_the_saved_revision_to_the_running_runtime(api_harness):
    current = await api_harness.client.get("/api/config")

    saved = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": active_payload()},
    )

    assert saved.status_code == 200
    api_harness.runtime.prepare_candidate.assert_awaited_once()
    api_harness.runtime.publish_candidate.assert_awaited_once()


async def test_put_config_persists_the_explicit_live_order_execution_gate(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["live_order_execution_enabled"] = True

    saved = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert saved.status_code == 200
    assert saved.json()["document"]["execution"]["live_order_execution_enabled"] is True
    assert (await api_harness.client.get("/api/config")).json()["document"]["execution"][
        "live_order_execution_enabled"
    ] is True
    assert api_harness.runtime.snapshot.document.execution.live_order_execution_enabled is True


async def test_put_config_refreshes_application_runtime_owners_after_publication(api_harness):
    from api.main import app

    refresh = AsyncMock()
    previous = getattr(app.state, "refresh_runtime_owners", None)
    app.state.refresh_runtime_owners = refresh
    try:
        current = await api_harness.client.get("/api/config")
        saved = await api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
    finally:
        app.state.refresh_runtime_owners = previous

    assert saved.status_code == 200
    refresh.assert_awaited_once()


async def test_candidate_failure_leaves_the_desired_revision_unwritten(api_harness):
    api_harness.runtime.prepare_candidate = AsyncMock(side_effect=RuntimeError("adapter failure"))
    current = await api_harness.client.get("/api/config")

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": active_payload()},
    )

    assert response.status_code == 503
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"]


async def test_owner_failure_marks_desired_revision_failed_then_a_later_revision_recovers(api_harness):
    from api.main import app

    old_refresh = getattr(app.state, "refresh_runtime_owners", None)
    app.state.refresh_runtime_owners = AsyncMock(side_effect=RuntimeError("owner failure"))
    try:
        current = await api_harness.client.get("/api/config")
        failed = await api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
        assert failed.status_code == 503
        status = (await api_harness.client.get("/api/config")).json()
        assert (status["revision"], status["apply_status"], status["applied_revision"]) == (2, "failed", 1)
        api_harness.runtime.fail_closed.assert_awaited_once()
        app.state.refresh_runtime_owners = AsyncMock()
        recovered = await api_harness.client.put(
            "/api/config",
            json={"expected_revision": status["revision"], "document": active_payload()},
        )
    finally:
        app.state.refresh_runtime_owners = old_refresh

    assert recovered.status_code == 200
    assert (recovered.json()["apply_status"], recovered.json()["applied_revision"]) == ("applied", 3)


async def test_cas_conflict_closes_the_candidate_and_keeps_the_old_runtime_operational(api_harness):
    from cryptotrader.runtime_config.repository import RevisionConflict

    current = await api_harness.client.get("/api/config")
    candidate = SimpleNamespace(close=AsyncMock())
    api_harness.runtime.prepare_candidate = AsyncMock(return_value=candidate)
    original_replace = api_harness.runtime.repository.replace
    api_harness.runtime.repository.replace = AsyncMock(side_effect=RevisionConflict(current.json()["revision"], 2))
    try:
        conflict = await api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
    finally:
        api_harness.runtime.repository.replace = original_replace

    assert conflict.status_code == 409
    candidate.close.assert_awaited_once()
    api_harness.runtime.publish_candidate.assert_not_awaited()
    assert api_harness.runtime.snapshot.revision == current.json()["revision"]


async def test_mark_applied_failure_fails_closed_and_exposes_failed_desired_revision(api_harness):
    current = await api_harness.client.get("/api/config")
    original_mark_applied = api_harness.runtime.repository.mark_applied
    api_harness.runtime.repository.mark_applied = AsyncMock(side_effect=RuntimeError("write failure"))
    try:
        failed = await api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
    finally:
        api_harness.runtime.repository.mark_applied = original_mark_applied

    assert failed.status_code == 503
    status = (await api_harness.client.get("/api/config")).json()
    assert (status["apply_status"], status["applied_revision"]) == ("failed", 1)
    api_harness.runtime.fail_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_config_read_is_rejected_while_a_real_asgi_application_request_has_marked_db_applied(
    api_harness,
):
    """The API never leaks repository-applied state before the runtime graph activates it."""
    current = await api_harness.client.get("/api/config")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block_activation(applied):
        entered.set()
        await release.wait()
        api_harness.runtime.snapshot = applied

    api_harness.runtime.activate_applied = AsyncMock(side_effect=block_activation)
    applying = asyncio.create_task(
        api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
    )
    await entered.wait()

    blocked = await api_harness.client.get("/api/config")
    assert blocked.status_code == 503
    assert blocked.json() == {"detail": "Runtime configuration is being applied"}
    business = await api_harness.client.post("/api/backtest/runs", json={})
    assert business.status_code == 503
    assert business.json() == {"detail": "Runtime configuration is being applied"}

    release.set()
    assert (await applying).status_code == 200


@pytest.mark.asyncio
async def test_venue_credential_mutation_waits_for_application_barrier_while_reads_and_connection_test_are_rejected(
    api_harness,
):
    """Only commissioning mutations queue behind an application; reads and test-connect never enter it."""
    current = await api_harness.client.get("/api/config")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def block_activation(applied):
        entered.set()
        await release.wait()
        api_harness.runtime.snapshot = applied

    api_harness.runtime.activate_applied = AsyncMock(side_effect=block_activation)
    applying = asyncio.create_task(
        api_harness.client.put(
            "/api/config",
            json={"expected_revision": current.json()["revision"], "document": active_payload()},
        )
    )
    await entered.wait()

    queued = asyncio.create_task(
        api_harness.client.put(
            "/api/venue-connections/okx-demo/credentials",
            json={
                "expected_revision": current.json()["revision"] + 1,
                "values": {
                    "api_key": "queued-key",  # pragma: allowlist secret
                    "secret": "queued-secret",  # pragma: allowlist secret
                    "passphrase": "queued-pass",
                },  # pragma: allowlist secret
            },
        )
    )
    await asyncio.sleep(0)
    assert queued.done() is False
    assert (await api_harness.client.get("/api/config")).status_code == 503
    assert (await api_harness.client.post("/api/venue-connections/okx-demo/test")).status_code == 503

    release.set()
    assert (await applying).status_code == 200
    assert (await queued).status_code == 200


async def test_stale_put_config_conflicts_before_connection_domain_construction(api_harness):
    current = await api_harness.client.get("/api/config")
    revision = current.json()["revision"]
    assert (
        await api_harness.client.put(
            "/api/config",
            json={"expected_revision": revision, "document": active_payload()},
        )
    ).status_code == 200
    invalid = active_payload()
    connection = next(item for item in invalid["execution"]["connections"] if item["id"] == "paper-spare")
    connection.update(adapter_id="okx", environment="live")

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": revision, "document": invalid},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Runtime configuration changed; reload and retry"}


async def test_put_config_validates_the_whole_document(api_harness):
    current = await api_harness.client.get("/api/config")
    invalid = active_payload()
    invalid["execution"]["books"][0]["allocations"][0]["weight"] = 0.4

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": invalid},
    )

    assert response.status_code == 422


async def test_activation_requires_nonempty_api_and_llm_credentials_before_persisting(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["security"] = {"enabled": True}

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422
    after = await api_harness.client.get("/api/config")
    assert after.json()["revision"] == current.json()["revision"]


async def test_saving_an_analysis_draft_does_not_require_gateway_credentials(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["signals"] = signal_config().model_dump(mode="json")

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 200
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"] + 1
    assert response.json()["document"]["llm"]["gateway_credential_configured"] is False


async def test_put_config_cannot_change_existing_connection_environment(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    connection = next(item for item in document["execution"]["connections"] if item["id"] == "paper-spare")
    connection.update(
        adapter_id="okx",
        environment="demo",
    )

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422


async def test_put_config_cannot_delete_enabled_unverified_connection(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["connections"] = [
        item for item in document["execution"]["connections"] if item["id"] != "paper-spare"
    ]

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 409


async def test_put_config_cannot_disable_connection_referenced_by_enabled_book(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    connection = next(item for item in document["execution"]["connections"] if item["id"] == "okx-demo")
    connection["enabled"] = False

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422


async def test_config_and_connection_responses_never_return_credentials(api_harness):
    marker = "credential-redaction-sentinel"
    saved = await put_fixture_credentials(api_harness, "okx-demo", marker=marker)
    assert saved.status_code == 200

    config = await api_harness.client.get("/api/config")
    portfolio = await api_harness.client.get("/api/portfolio/books", params={"pair": PAIR.canonical()})
    body = json.dumps(config.json()) + json.dumps(portfolio.json()) + saved.text
    assert marker not in body
    assert "credential_ref" not in body
    assert '"configured":true' in saved.text
    assert config.json()["document"]["execution"]["connections"][0]["credential_configured"] is True


async def test_signal_profile_route_remains_unmounted(api_harness):
    assert (await api_harness.client.get("/api/signal-profile")).status_code == 404


async def test_all_config_request_models_forbid_unknown_fields(api_harness):
    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": 1, "document": active_payload(), "unexpected": True},
    )

    assert response.status_code == 422


async def test_credential_validation_error_never_echoes_secret_input(api_harness):
    marker = "credential-validation-redaction-sentinel"
    response = await api_harness.client.put(
        "/api/venue-connections/okx-demo/credentials",
        json={
            "expected_revision": 1,
            "values": {"api_key": marker, "secret": marker, "unexpected": marker},
        },
    )

    assert response.status_code == 422
    assert marker not in response.text
