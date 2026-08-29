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
    RuntimeConfigDocument,
    SignalComponentConfig,
    SystemConfig,
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
        margin_mode="isolated",
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
        system=SystemConfig(active=True),
        market_data=market_config(),
        signals=signal_config(components=(SignalComponentConfig(component_id="kronos", enabled=True, weight=1.0),)),
        execution=ExecutionConfig(connections=connections, books=books),
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
async def api_harness(tmp_path):
    from api.main import app

    document = active_document()
    repository = RuntimeConfigRepository(
        f"sqlite+aiosqlite:///{tmp_path / 'runtime-api.db'}",
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
        signal_registry=SimpleNamespace(installed_ids=lambda: frozenset({"kronos", "llm_committee"})),
        market_registry=SimpleNamespace(installed_ids=lambda: frozenset({"default"})),
        venue_registry=VenueAdapterRegistry(tuple(adapters.values())),
        reload_for_cycle=AsyncMock(),
        application_barrier=application_barrier,
        prepare_candidate=AsyncMock(return_value=SimpleNamespace(close=AsyncMock())),
        publish_candidate=AsyncMock(),
        activate_applied=AsyncMock(side_effect=activate_applied),
        fail_closed=AsyncMock(),
        application_in_progress=False,
    )
    previous = getattr(app.state, "runtime", None)
    app.state.runtime = runtime
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield ApiHarness(client, runtime, adapters)
    app.state.runtime = previous


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
            "credentials": {"api_key": marker, "secret": marker, "passphrase": marker},
        },
    )


async def test_get_config_exposes_latest_snapshot_and_credential_state(api_harness):
    response = await api_harness.client.get("/api/config")

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 1
    assert body["setup_required"] is False
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


async def test_put_config_publishes_the_saved_revision_to_the_running_runtime(api_harness):
    current = await api_harness.client.get("/api/config")

    saved = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": active_payload()},
    )

    assert saved.status_code == 200
    api_harness.runtime.prepare_candidate.assert_awaited_once()
    api_harness.runtime.publish_candidate.assert_awaited_once()


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
    business = await api_harness.client.get("/api/backtest/sessions")
    assert business.status_code == 503
    assert business.json() == {"detail": "Runtime configuration is being applied"}

    release.set()
    assert (await applying).status_code == 200


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


async def test_activation_with_llm_committee_requires_gateway_credential_before_persisting(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["signals"] = signal_config().model_dump(mode="json")

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"]


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


async def test_put_config_cannot_delete_existing_connection(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["connections"] = [
        item for item in document["execution"]["connections"] if item["id"] != "paper-spare"
    ]

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422


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
            "credentials": {"api_key": marker, "secret": marker, "unexpected": marker},
        },
    )

    assert response.status_code == 422
    assert marker not in response.text
