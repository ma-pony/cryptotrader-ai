"""数据库 RuntimeConfig API 的 CAS、setup 与严格脱敏契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace

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
        signals=signal_config(),
        execution=ExecutionConfig(connections=connections, books=books),
    )


def active_payload() -> dict:
    return active_document().model_dump(mode="json")


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
        self.capabilities = VenueCapabilities(
            frozenset({"swap"}),
            True,
            False,
            True,
            frozenset({"market"}),
        )
        self.closed = 0

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        return self.snapshot

    async def close(self) -> None:
        self.closed += 1


class FakeAdapter:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.connect_calls: list[tuple[VenueConnection, object]] = []
        self.opened_sessions: list[FakeSession] = []
        self.error: Exception | None = None

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
        self.opened_sessions.append(session)
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
    runtime = SimpleNamespace(
        snapshot=snapshot,
        repository=repository,
        cycle=cycle,
        sessions=sessions,
        signal_registry=SimpleNamespace(installed_ids=lambda: frozenset({"kronos", "llm_committee"})),
        market_registry=SimpleNamespace(installed_ids=lambda: frozenset({"default"})),
        venue_registry=VenueAdapterRegistry(tuple(adapters.values())),
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


async def test_put_config_validates_the_whole_document(api_harness):
    current = await api_harness.client.get("/api/config")
    invalid = active_payload()
    invalid["execution"]["books"][0]["allocations"][0]["weight"] = 0.4

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": invalid},
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
