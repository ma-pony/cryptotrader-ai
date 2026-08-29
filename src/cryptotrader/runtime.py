"""数据库配置驱动的唯一运行时装配边界。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping as MappingABC
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

    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.venues.protocol import VenueSession


@dataclass
class Runtime:
    """One atomically published runtime graph and its owned venue sessions."""

    snapshot: RuntimeConfigSnapshot
    repository: RuntimeConfigRepository
    cycle: TradingCycle | None
    sessions: Mapping[str, VenueSession]
    signal_registry: SignalComponentRegistry
    market_registry: MarketSourceRegistry
    venue_registry: VenueAdapterRegistry
    events: MultiplexedCycleEventSink
    _session_keys: dict[str, tuple[object, ...]] = field(default_factory=dict, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _started_setup_required: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_setup_required = self.snapshot.setup_required

    async def reload_for_cycle(self) -> TradingCycle | None:
        """Synchronize one active runtime to the latest validated database graph."""

        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("runtime is closed")
            if self._started_setup_required:
                return None

            candidate_snapshot = await self.repository.get_or_create()
            _validate_snapshot(
                candidate_snapshot,
                self.signal_registry,
                self.venue_registry,
                self.market_registry,
            )
            if candidate_snapshot.setup_required:
                retired = tuple(self.sessions.values())
                self.snapshot, self.sessions, self.cycle, self._session_keys = (
                    candidate_snapshot,
                    {},
                    None,
                    {},
                )
                await _close_sessions(retired)
                return None

            candidate_sessions, candidate_keys, opened = await _candidate_sessions(
                candidate_snapshot,
                self.repository,
                self.venue_registry,
                self.sessions,
                self._session_keys,
            )
            try:
                candidate_cycle = _assemble_cycle(
                    candidate_snapshot,
                    self.repository,
                    candidate_sessions,
                    self.signal_registry,
                    self.market_registry,
                    self.events,
                )
            except BaseException:
                await _close_candidate_sessions(opened)
                raise

            retired = tuple(
                session
                for session in self.sessions.values()
                if all(session is not candidate for candidate in candidate_sessions.values())
            )
            self.snapshot, self.sessions, self.cycle, self._session_keys = (
                candidate_snapshot,
                candidate_sessions,
                candidate_cycle,
                candidate_keys,
            )
            await _close_sessions(retired)
            return candidate_cycle

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            await _close_sessions(tuple(self.sessions.values()))


async def build_runtime(
    settings: BootstrapSettings | None = None,
    event_sink: CycleEventSink | None = None,
    *,
    repository: RuntimeConfigRepository | None = None,
    snapshot: RuntimeConfigSnapshot | None = None,
    signal_registry: SignalComponentRegistry | None = None,
    venue_registry: VenueAdapterRegistry | None = None,
    market_registry: MarketSourceRegistry | None = None,
) -> Runtime:
    """Build setup or active runtime from exactly one database snapshot."""

    if repository is None:
        if settings is None:
            from cryptotrader.bootstrap import BootstrapSettings

            settings = BootstrapSettings.from_environment()
        runtime_repository = RuntimeConfigRepository(
            settings.database_url,
            CredentialVault(settings.config_master_key),
        )
    else:
        runtime_repository = repository
    frozen = snapshot or await runtime_repository.get_or_create()
    routed_events = (
        event_sink
        if isinstance(event_sink, MultiplexedCycleEventSink)
        else MultiplexedCycleEventSink(event_sink or NullCycleEventSink())
    )
    signals = signal_registry or SignalComponentRegistry.discover(frozen.document, routed_events)
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
        return Runtime(frozen, runtime_repository, None, {}, signals, markets, venues, routed_events)

    sessions, session_keys, opened = await _candidate_sessions(
        frozen,
        runtime_repository,
        venues,
        {},
        {},
    )
    try:
        cycle = _assemble_cycle(frozen, runtime_repository, sessions, signals, markets, routed_events)
    except BaseException:
        await _close_candidate_sessions(opened)
        raise
    return Runtime(
        frozen,
        runtime_repository,
        cycle,
        sessions,
        signals,
        markets,
        venues,
        routed_events,
        session_keys,
    )


def _validate_snapshot(snapshot, signals, venues, markets) -> None:
    validate_runtime_document(
        snapshot.document,
        set(signals.installed_ids()),
        set(venues.installed_ids()),
        set(markets.installed_ids()),
    )


def _assemble_cycle(snapshot, repository, sessions, signals, markets, event_sink) -> TradingCycle:
    frozen = snapshot
    runtime_repository = repository
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
    return TradingCycle(
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


async def _candidate_sessions(
    snapshot,
    repository,
    venue_registry,
    current_sessions,
    current_keys,
) -> tuple[dict[str, VenueSession], dict[str, tuple[object, ...]], tuple[VenueSession, ...]]:
    sessions: dict[str, VenueSession] = {}
    keys: dict[str, tuple[object, ...]] = {}
    opened: list[VenueSession] = []
    try:
        for connection in snapshot.document.execution.connections:
            if not connection.enabled:
                continue
            key = await _session_key(connection, repository)
            keys[connection.id] = key
            existing = current_sessions.get(connection.id)
            if existing is not None and current_keys.get(connection.id) == key:
                sessions[connection.id] = existing
                continue
            credentials = (
                await repository.reveal_credentials(connection.credential_ref)
                if connection.credential_ref is not None
                else None
            )
            adapter = venue_registry.require(connection.adapter_id)
            session = await adapter.connect(connection, credentials)
            sessions[connection.id] = session
            opened.append(session)
    except BaseException as error:
        await _close_candidate_sessions(tuple(opened))
        if isinstance(error, Exception):
            raise RuntimeError("failed to open venue session") from None
        raise
    return sessions, keys, tuple(opened)


async def _session_key(connection, repository) -> tuple[object, ...]:
    credential_updated_at = None
    if connection.credential_ref is not None:
        state = await repository.credential_state(connection.credential_ref)
        credential_updated_at = state.updated_at
    return (
        connection.id,
        connection.adapter_id,
        connection.environment,
        connection.leverage,
        connection.margin_mode,
        _canonical_value(connection.parameters),
        connection.credential_ref,
        credential_updated_at,
    )


def _canonical_value(value):
    if isinstance(value, MappingABC):
        return (
            "mapping",
            tuple((key, _canonical_value(item)) for key, item in sorted(value.items())),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_canonical_value(item) for item in value))
    return (type(value).__name__, value)


async def _close_candidate_sessions(sessions) -> None:
    try:
        await _close_sessions(sessions)
    except Exception:
        return


async def _close_sessions(sessions) -> None:
    unique = tuple({id(session): session for session in sessions}.values())
    if not unique:
        return
    outcomes = await asyncio.gather(*(session.close() for session in unique), return_exceptions=True)
    control_flow = next(
        (item for item in outcomes if isinstance(item, BaseException) and not isinstance(item, Exception)),
        None,
    )
    if control_flow is not None:
        raise control_flow
    if any(isinstance(item, Exception) for item in outcomes):
        raise RuntimeError("failed to close venue session") from None
