"""Fully isolated workbench app used by Playwright and browser acceptance.

It owns a temporary SQLite database, test vault, deterministic clock, fixed
market source, model-like signal component, Paper-backed venue and fake
Webhook.  It never loads BootstrapSettings or starts production owners.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from api.dependencies import verify_api_key
from api.routes import (
    account_operations,
    accounts,
    alerts,
    analyses,
    backtest,
    components,
    config,
    decisions,
    hitl,
    portfolio_books,
    runtime_status,
    scheduler,
    trading_runs,
    venues,
)
from cryptotrader import cycle_lock as cycle_lock_module
from cryptotrader.configuration import registry as registry_module
from cryptotrader.configuration.catalog import EnvironmentDefinition, PluginConfiguration
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import PluginParameters
from cryptotrader.configuration.registry import ExtensionRegistration, ExtensionRegistry
from cryptotrader.market_sources.protocol import HistoricalCandle
from cryptotrader.migrations.workbench import migrate_workbench_schema
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.runtime import build_runtime
from cryptotrader.runtime_config.defaults import minimal_runtime_document
from cryptotrader.runtime_config.models import HitlConfig, InfrastructureConfig, SecurityConfig
from cryptotrader.runtime_config.repository import API_ACCESS_CREDENTIAL_REF, RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault, TokenPayload
from cryptotrader.signals.models import ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.presentation import (
    Metric,
    MetricsBlock,
    TableBlock,
    TableCell,
    TableColumn,
    TableRow,
    interval_delta,
)
from cryptotrader.venues.models import ACCOUNT_READS, EXIT_OPERATIONS, VenueCapabilities
from cryptotrader.venues.paper import PaperVenueAdapter

MASTER_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
ACCESS_KEY = "workbench-test-access"
PAIR = Pair.parse("BTC/USDT")
INITIAL_TIME = datetime(2026, 8, 31, 10, tzinfo=UTC)
_BUILTIN_REGISTRY = registry_module.get_extension_registry


class WorkbenchClock:
    def __init__(self) -> None:
        self.current = INITIAL_TIME

    def now(self) -> datetime:
        return self.current

    def advance(self, *, hours: int) -> datetime:
        if not 1 <= hours <= 168:
            raise ValueError("test clock advance must be between 1 and 168 hours")
        self.current += timedelta(hours=hours)
        return self.current


clock = WorkbenchClock()


class WorkbenchLockState:
    """Process-local strict-lock transport for the isolated acceptance app."""

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    async def try_acquire_strict_lock(self, key: str, owner_id: str, _ttl: int) -> bool:
        if key in self._owners:
            return False
        self._owners[key] = owner_id
        return True

    async def release_strict_lock(self, key: str, owner_id: str) -> bool:
        if self._owners.get(key) != owner_id:
            return False
        del self._owners[key]
        return True

    async def aclose(self) -> None:
        return None


class SampleParameters(PluginParameters):
    account_code: str = Field(
        min_length=1,
        title="Account code",
        json_schema_extra={"label": {"zh_CN": "账户编码", "en_US": "Account code"}},
    )


class SampleCredentials(PluginParameters):
    access_token: SecretStr = Field(
        min_length=1,
        title="Access token",
        json_schema_extra={"label": {"zh_CN": "访问令牌", "en_US": "Access token"}},
    )
    tenant_pin: SecretStr = Field(
        min_length=1,
        title="Tenant PIN",
        json_schema_extra={"label": {"zh_CN": "租户 PIN", "en_US": "Tenant PIN"}},
    )


class SampleSignalParameters(PluginParameters):
    window: int = Field(
        default=12,
        ge=1,
        le=120,
        title="Observation window",
        json_schema_extra={"label": {"zh_CN": "观察窗口", "en_US": "Observation window"}},
    )


class WorkbenchSignal:
    id = "sample_signal"
    display_name = "测试趋势信号"
    description = "固定本地信号且不调用外部模型。"

    def __init__(self, context) -> None:
        configured = next(item for item in context.document.signals.components if item.component_id == self.id)
        self.parameters = SampleSignalParameters.model_validate(dict(configured.parameters))

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, _context) -> ComponentSignal:
        window = self.parameters.window
        return ComponentSignal(
            self.id,
            "long",
            0.8,
            "固定离线样本判断为做多。",
            details={"window": window},
            blocks=(
                MetricsBlock(title="测试指标", metrics=(Metric(key="观察窗口", value=window),)),
                TableBlock(
                    title="测试结果",
                    columns=(
                        TableColumn(key="window", label="观察窗口"),
                        TableColumn(key="direction", label="方向"),
                    ),
                    rows=(
                        TableRow(
                            cells=(
                                TableCell(column_key="window", value=window),
                                TableCell(column_key="direction", value="做多"),
                            )
                        ),
                    ),
                ),
            ),
        )


class WorkbenchMarket:
    id = "default"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        snapshots = {}
        for requirement in requirements.candles:
            opened_at = as_of - interval_delta(requirement.timeframe)
            frame = pd.DataFrame(
                {
                    "timestamp": [int(opened_at.timestamp() * 1000)],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                    "volume": [10.0],
                }
            )
            snapshots[requirement.timeframe] = DataSnapshot(
                as_of,
                str(pair),
                MarketData(str(pair), frame, {"last": 100.0}, 0.0, 0.0, 0.0),
                OnchainData(),
                NewsSentiment(),
                MacroData(),
            )
        return SignalContext(pair, as_of, self.id, pair.market_type, 100.0, 5.0, snapshots)

    async def read_candles(self, _pair, _timeframe, start, end, as_of):
        if start >= end or end > as_of:
            return ()
        return (
            HistoricalCandle(
                open_time=start,
                open=Decimal("100"),
                high=Decimal("112"),
                low=Decimal("99"),
                close=Decimal("110"),
                volume=Decimal("10"),
            ),
        )


class WorkbenchVenue(PaperVenueAdapter):
    """Credentialed code-registered venue backed by one persistent Paper ledger."""

    adapter_id = "sample_venue"

    def __init__(self) -> None:
        super().__init__(clock=clock.now)
        self.connect_calls = 0
        self.account_reads = 0
        self.order_writes = 0

    def capabilities(self, environment: str) -> VenueCapabilities:
        if environment != "sandbox":
            raise ValueError("sample venue only supports sandbox")
        return VenueCapabilities(
            frozenset({"spot"}),
            True,
            False,
            True,
            frozenset({"market", "limit"}),
            ACCOUNT_READS,
            EXIT_OPERATIONS,
        )

    async def connect(self, connection, credentials):
        if connection.adapter_id != self.adapter_id or connection.environment != "sandbox":
            raise ValueError("invalid sample venue connection")
        if credentials is None or set(credentials.values) != {"access_token", "tenant_pin"}:
            raise ValueError("sample credentials are incomplete")
        self.connect_calls += 1
        paper_connection = replace(
            connection,
            adapter_id="paper",
            environment="paper",
            credential_ref=None,
            parameters={"initial_equity": 100},
        )
        paper = PaperVenueAdapter(clock=clock.now)
        paper.account_store = self.account_store
        session = await paper.connect(paper_connection, None)
        session.connection = connection
        session._capabilities = self.capabilities("sandbox")
        await session.set_quote(PAIR, Decimal("100"))
        original_fetch_account = session.fetch_account
        original_place_order = session.place_order

        async def fetch_account():
            self.account_reads += 1
            return await original_fetch_account()

        async def place_order(intent):
            self.order_writes += 1
            return await original_place_order(intent)

        session.fetch_account = fetch_account
        session.place_order = place_order
        return session


_venue_instances: list[WorkbenchVenue] = []


def _venue_factory() -> WorkbenchVenue:
    venue = WorkbenchVenue()
    _venue_instances.append(venue)
    return venue


def _market_factory(*_args, **_kwargs) -> WorkbenchMarket:
    return WorkbenchMarket()


def workbench_base_registry() -> ExtensionRegistry:
    """Return built-ins with the deterministic local market used by both fixtures."""

    builtins = _BUILTIN_REGISTRY()
    market = builtins.market_sources["default"]
    return ExtensionRegistry(
        components=builtins.components,
        venues=builtins.venues,
        market_sources={
            **builtins.market_sources,
            "default": replace(market, factory=_market_factory),
        },
    )


def workbench_registry() -> ExtensionRegistry:
    """Return built-ins plus the two code-only acceptance extensions."""

    builtins = workbench_base_registry()
    label = LocalizedText("测试扩展", "Sample extension")
    description = LocalizedText("仅用于隔离验收。不连接外部服务。", "Local acceptance fixture only.")
    venue = PluginConfiguration(
        "sample_venue",
        label,
        description,
        SampleParameters,
        environments=(EnvironmentDefinition("sandbox", LocalizedText("本地沙盒", "Local sandbox"), "simulated"),),
        credential_model=SampleCredentials,
        margin_modes=("cross",),
        capabilities=WorkbenchVenue().capabilities("sandbox"),
    )
    signal = PluginConfiguration(
        "sample_signal",
        LocalizedText("测试趋势信号", "Sample trend signal"),
        description,
        SampleSignalParameters,
    )
    return ExtensionRegistry(
        components={
            **builtins.components,
            "sample_signal": ExtensionRegistration(signal, lambda context: WorkbenchSignal(context)),
        },
        venues={**builtins.venues, "sample_venue": ExtensionRegistration(venue, _venue_factory)},
        market_sources=builtins.market_sources,
    )


class FakeWebhook:
    def __init__(self) -> None:
        self.deliveries: list[tuple[str, dict[str, Any]]] = []

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        self.deliveries.append((event, payload))


fake_webhook = FakeWebhook()


class AdvanceClockIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: int = Field(default=2, ge=1, le=168)


test_router = APIRouter(prefix="/__workbench__", tags=["workbench-test-control"])


@test_router.get("/state")
async def test_state(request: Request):
    runtime = request.app.state.runtime
    decisions_page = await runtime.read_service.list(limit=100)
    return {
        "now": clock.now().isoformat(),
        "database_kind": "temporary_sqlite",
        "revision": runtime.snapshot.revision,
        "decisions": [item.model_dump(mode="json") for item in decisions_page.items],
        "venue": {
            "connect_calls": sum(item.connect_calls for item in _venue_instances),
            "account_reads": sum(item.account_reads for item in _venue_instances),
            "order_writes": sum(item.order_writes for item in _venue_instances),
        },
        "webhook_deliveries": len(fake_webhook.deliveries),
    }


@test_router.post("/clock/advance")
async def advance_clock(body: AdvanceClockIn, request: Request):
    now = clock.advance(hours=body.hours)
    completed = await request.app.state.runtime.evaluation_service.evaluate_due(now)
    return {"now": now.isoformat(), "completed": completed}


@test_router.post("/tasks/drain")
async def drain_tasks(request: Request):
    await request.app.state.runtime.task_manager.drain()
    return {"drained": True}


@test_router.get("/paper/{connection_id}")
async def paper_ledger(connection_id: str, request: Request):
    account_store = request.app.state.runtime.account_store
    saved = await account_store.load_paper(connection_id)
    if saved is None:
        raise HTTPException(404, "paper ledger not found")
    latest = await account_store.latest(connection_id)
    return {
        "positions": {
            str(position.instrument.pair): {"signed_amount": str(position.signed_amount)}
            for position in (latest.positions if latest is not None else ())
        },
        "fills": saved["fills"],
        "orders": saved["orders"],
        "protections": saved["protections"],
    }


async def _no_runtime_owners(*_args, **_kwargs) -> None:
    return None


def _initial_document():
    return minimal_runtime_document().model_copy(
        update={
            "security": SecurityConfig(enabled=True),
            "infrastructure": InfrastructureConfig(redis_url="redis://workbench.invalid:6379/0"),
            "hitl": HitlConfig(approval_ttl_minutes=10_080),
        }
    )


def _lifespan_for(registry_factory):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        previous_registry = registry_module.get_extension_registry
        previous_redis_state = cycle_lock_module.RedisStateManager
        local_lock_state = WorkbenchLockState()
        registry_module.get_extension_registry = registry_factory
        cycle_lock_module.RedisStateManager = lambda _url: local_lock_state
        clock.current = INITIAL_TIME
        _venue_instances.clear()
        fake_webhook.deliveries.clear()
        scratch = Path(tempfile.mkdtemp(prefix="cryptotrader-workbench-"))
        database_url = f"sqlite+aiosqlite:///{scratch / 'workbench.db'}"
        repository = RuntimeConfigRepository(
            database_url,
            CredentialVault(MASTER_KEY),
            default_factory=_initial_document,
        )
        runtime = None
        try:
            await migrate_workbench_schema(database_url)
            initial = await repository.get_or_create()
            pending = await repository.put_token(
                initial.revision,
                API_ACCESS_CREDENTIAL_REF,
                TokenPayload(token=ACCESS_KEY),
            )
            applied = await repository.mark_applied(pending.revision)
            runtime = await build_runtime(repository=repository, snapshot=applied)
            runtime.run_service.clock = clock.now

            @asynccontextmanager
            async def local_execution_lease(pair, *, expected_revision=None, confirmed_book_ids=None, origin="manual"):
                async with cycle_lock_module.cycle_lock(local_lock_state, pair) as acquired:
                    if not acquired:
                        raise RuntimeError("workbench execution lease is already owned")
                    async with runtime.cycle_lease(
                        expected_revision=expected_revision,
                        confirmed_book_ids=confirmed_book_ids,
                        origin=origin,
                        pair=pair,
                    ) as cycle:
                        yield cycle

            runtime.execution_lease = local_execution_lease
            runtime.refresh_owners = _no_runtime_owners
            runtime.clear_owners = _no_runtime_owners
            runtime.deliveries.backend_factory = lambda _config: fake_webhook
            app.state.runtime = runtime
            app.state.refresh_runtime_owners = _no_runtime_owners
            app.state.clear_runtime_owners = _no_runtime_owners
            app.state.workbench_scratch = scratch
            yield
        finally:
            if runtime is not None:
                await runtime.close()
            cycle_lock_module.RedisStateManager = previous_redis_state
            registry_module.get_extension_registry = previous_registry
            shutil.rmtree(scratch, ignore_errors=True)

    return lifespan


def build_workbench_app(*, registry_factory=workbench_registry) -> FastAPI:
    fixture = FastAPI(title="CryptoTrader isolated workbench", lifespan=_lifespan_for(registry_factory))
    fixture.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4174"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Trace-ID"],
    )

    @fixture.get("/health")
    async def health():
        return {"status": "ok", "environment": "isolated-workbench"}

    protected = [Depends(verify_api_key)]
    for router in (
        config.router,
        venues.router,
        portfolio_books.router,
        accounts.router,
        account_operations.router,
        alerts.router,
        decisions.router,
        components.router,
        analyses.router,
        runtime_status.router,
        trading_runs.router,
        backtest.router,
        scheduler.api_router,
        hitl.router,
    ):
        fixture.include_router(router, dependencies=protected)
    fixture.include_router(test_router)
    return fixture


app = build_workbench_app()
