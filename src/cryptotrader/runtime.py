"""数据库配置驱动的唯一运行时装配边界。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.cycle_events import CycleEventSink, MultiplexedCycleEventSink, NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.execution.allocation import WeightedAllocationPolicy
from cryptotrader.execution.coordinator import ExecutionCoordinator
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.execution.service import VenueExecutionService
from cryptotrader.hitl.store import BookApprovalStore
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.portfolio.aggregator import PortfolioAggregator
from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
from cryptotrader.risk.models import BookRiskLimits, ConnectionRiskLimits
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, validate_runtime_document
from cryptotrader.runtime_config.repository import RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunner
from cryptotrader.trading_cycle import TradingCycle
from cryptotrader.venues.registry import VenueAdapterRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cryptotrader.venues.protocol import VenueSession


@dataclass
class Runtime:
    """One frozen startup snapshot and the resources assembled from it."""

    snapshot: RuntimeConfigSnapshot
    repository: RuntimeConfigRepository
    cycle: TradingCycle | None
    sessions: Mapping[str, VenueSession]
    signal_registry: SignalComponentRegistry
    market_registry: MarketSourceRegistry
    venue_registry: VenueAdapterRegistry
    events: MultiplexedCycleEventSink
    _closed: bool = field(default=False, init=False, repr=False)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sessions = tuple(self.sessions.values())
        if not sessions:
            return
        outcomes = await asyncio.gather(*(session.close() for session in sessions), return_exceptions=True)
        failure = next((item for item in outcomes if isinstance(item, BaseException)), None)
        if failure is not None:
            raise RuntimeError("failed to close venue session") from None


async def build_runtime(
    *,
    repository: RuntimeConfigRepository | None = None,
    snapshot: RuntimeConfigSnapshot | None = None,
    signal_registry: SignalComponentRegistry | None = None,
    venue_registry: VenueAdapterRegistry | None = None,
    market_registry: MarketSourceRegistry | None = None,
    events: CycleEventSink | None = None,
) -> Runtime:
    """Build setup or active runtime from exactly one database snapshot."""

    runtime_repository = repository or _repository_from_bootstrap_environment()
    frozen = snapshot or await runtime_repository.get_or_create()
    event_sink = (
        events
        if isinstance(events, MultiplexedCycleEventSink)
        else MultiplexedCycleEventSink(events or NullCycleEventSink())
    )
    signals = signal_registry or SignalComponentRegistry.discover(frozen.document, event_sink)
    venues = venue_registry or VenueAdapterRegistry.discover(
        connection.adapter_id for connection in frozen.document.execution.connections
    )
    markets = market_registry or MarketSourceRegistry.discover(frozen.document.market_data)
    validate_runtime_document(
        frozen.document,
        set(signals.installed_ids()),
        set(venues.installed_ids()),
        set(markets.installed_ids()),
    )
    if frozen.setup_required:
        return Runtime(frozen, runtime_repository, None, {}, signals, markets, venues, event_sink)

    sessions = await _open_sessions(frozen, runtime_repository, venues)
    services = {connection_id: VenueExecutionService(session) for connection_id, session in sessions.items()}
    allocation = WeightedAllocationPolicy()
    risk = frozen.document.risk
    book_risk = BookRiskGate(
        BookRiskLimits(
            Decimal(str(risk.position.max_total_exposure_pct)),
            Decimal(str(risk.position.max_total_exposure_pct)),
            Decimal(str(risk.loss.max_drawdown_pct)),
            Decimal(str(risk.position.max_single_pct)),
        ),
        allocation,
    )
    connection_risk = ConnectionRiskGate(ConnectionRiskLimits(Decimal(str(risk.position.max_margin_used_pct))))
    planner = ExecutionPlanner(book_risk_gate=book_risk, connection_risk_gate=connection_risk)
    database_url = getattr(runtime_repository, "database_url", None)
    market_source = markets.require(frozen.document.market_data.source_id)
    timeframe = str(frozen.document.market_data.parameters.get("timeframe", "1h"))
    limit = int(frozen.document.market_data.parameters.get("limit", 100))
    cycle = TradingCycle(
        repository=runtime_repository,
        market_source=market_source,
        registry=signals,
        runner=ComponentRunner(event_sink),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        sessions=sessions,
        portfolios=PortfolioAggregator(),
        allocation_policy=allocation,
        book_risk=book_risk,
        connection_risk=connection_risk,
        planner=planner,
        approvals=BookApprovalStore(database_url),
        coordinator=ExecutionCoordinator(services),
        journal=MultiVenueCycleStore(database_url),
        events=event_sink,
        exit_requirement=DataRequirements(candles=(CandleRequirement(timeframe, max(20, limit)),)),
    )
    return Runtime(frozen, runtime_repository, cycle, sessions, signals, markets, venues, event_sink)


def _repository_from_bootstrap_environment() -> RuntimeConfigRepository:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    master_key = os.environ.get("CONFIG_MASTER_KEY", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    if not master_key:
        raise RuntimeError("CONFIG_MASTER_KEY is required")
    return RuntimeConfigRepository(database_url, CredentialVault(master_key))


async def _open_sessions(snapshot, repository, venue_registry) -> dict[str, VenueSession]:
    sessions: dict[str, VenueSession] = {}
    try:
        for connection in snapshot.document.execution.connections:
            if not connection.enabled:
                continue
            credentials = (
                await repository.reveal_credentials(connection.credential_ref)
                if connection.credential_ref is not None
                else None
            )
            adapter = venue_registry.require(connection.adapter_id)
            sessions[connection.id] = await adapter.connect(connection, credentials)
    except BaseException:
        await asyncio.gather(*(session.close() for session in sessions.values()), return_exceptions=True)
        raise
    return sessions
