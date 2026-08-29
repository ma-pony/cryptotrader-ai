"""数据库配置驱动的唯一运行时装配边界。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from collections.abc import Mapping as MappingABC
from contextlib import asynccontextmanager
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
from cryptotrader.execution_ownership import wait_for_owned
from cryptotrader.hitl.store import BookApprovalStore
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.portfolio.aggregator import PortfolioAggregator
from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
from cryptotrader.risk.models import BookRiskLimits, ConnectionRiskLimits
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, validate_runtime_document
from cryptotrader.runtime_config.repository import (
    LLM_GATEWAY_CREDENTIAL_REF,
    CredentialNotConfigured,
    RuntimeConfigRepository,
)
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

logger = logging.getLogger(__name__)


class RuntimeLeaseUnavailableError(RuntimeError):
    """The Runtime cannot admit a new execution owner."""


@dataclass(frozen=True)
class _CloseBatchResult:
    ordinary_failures: tuple[object, ...]
    control_failures: tuple[tuple[object, BaseException], ...]
    external_cancellation: asyncio.CancelledError | None

    @property
    def failed_sessions(self) -> tuple[object, ...]:
        return self.ordinary_failures + tuple(session for session, _error in self.control_failures)


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
    _registry_discoverer: (
        Callable[[object], Awaitable[tuple[SignalComponentRegistry, VenueAdapterRegistry, MarketSourceRegistry]]] | None
    ) = field(default=None, repr=False)
    _pending_retired: dict[int, VenueSession] = field(default_factory=dict, init=False, repr=False)
    _deferred_retired: dict[int, VenueSession] = field(default_factory=dict, init=False, repr=False)
    _active_leases: int = field(default=0, init=False, repr=False)
    _leases_drained: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_completion: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _application_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _application_owner: asyncio.Task[object] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._leases_drained.set()

    @property
    def application_in_progress(self) -> bool:
        """Whether a desired revision is being committed into the live graph."""
        return self._application_owner is not None

    @asynccontextmanager
    async def application_barrier(self):
        """Serialize configuration application while leaving existing leases releasable."""
        await self._application_lock.acquire()
        owner = asyncio.current_task()
        try:
            async with self._lifecycle_lock:
                if self._closing or self._closed:
                    raise RuntimeError("runtime is unavailable")
                self._application_owner = owner
            yield
        finally:
            # A caller may be cancelled again while its context-manager cleanup
            # is waiting on the lifecycle lock.  The admission lock is ours, so
            # its release must complete before that control flow is propagated.
            await wait_for_owned(asyncio.create_task(self._release_application_barrier(owner)))

    async def _release_application_barrier(self, owner: asyncio.Task[object] | None) -> None:
        async with self._lifecycle_lock:
            if self._application_owner is owner:
                self._application_owner = None
            self._application_lock.release()

    @asynccontextmanager
    async def cycle_lease(self):
        """Publish and pin one frozen runtime graph for an execution owner."""

        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeLeaseUnavailableError("runtime is unavailable")
            if self._application_owner is not None:
                raise RuntimeLeaseUnavailableError("runtime application is in progress")
            cycle = await self._reload_for_cycle_locked()
            if cycle is None:
                raise RuntimeLeaseUnavailableError("runtime is unavailable")
            self._active_leases += 1
            self._leases_drained.clear()
        try:
            yield cycle
        finally:
            await wait_for_owned(asyncio.create_task(self._release_cycle_lease()))

    @asynccontextmanager
    async def execution_lease(self, pair: str):
        """Admit exactly one production cycle for a canonical pair.

        Unlike the graph lease this is a strict distributed ownership lease:
        an absent or unhealthy Redis is an execution refusal.
        """
        from cryptotrader.cycle_lock import ExecutionLeaseUnavailableError, execution_pair_lease
        from cryptotrader.pair import Pair

        canonical_pair = Pair.parse(pair).canonical()
        async with self.cycle_lease() as cycle:
            redis_url = cycle.snapshot.document.infrastructure.redis_url.strip()
            if not redis_url:
                raise RuntimeLeaseUnavailableError("Redis is required for production execution")
            try:
                async with execution_pair_lease(redis_url, canonical_pair):
                    yield cycle
            except ExecutionLeaseUnavailableError as error:
                raise RuntimeLeaseUnavailableError(str(error)) from error

    async def _release_cycle_lease(self) -> None:
        deferred_control: BaseException | None = None
        while True:
            retired = ()
            async with self._lifecycle_lock:
                if self._active_leases > 1:
                    self._active_leases -= 1
                    release_complete = True
                elif self._deferred_retired:
                    retired = tuple(self._deferred_retired.values())
                    self._deferred_retired.clear()
                    release_complete = False
                else:
                    self._active_leases = 0
                    self._leases_drained.set()
                    release_complete = True

            if release_complete:
                if deferred_control is not None:
                    raise deferred_control
                return

            result = await _close_session_batch(retired)
            async with self._lifecycle_lock:
                self._pending_retired.update((id(session), session) for session in result.failed_sessions)
            if deferred_control is None:
                deferred_control = _close_control(result)

    async def reload_for_cycle(self) -> TradingCycle | None:
        """Synchronize one active runtime to the latest validated database graph."""

        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is closed")
            if self._application_owner is not None:
                raise RuntimeError("runtime application is in progress")
            return await self._reload_for_cycle_locked()

    async def prepare_candidate(self, snapshot: RuntimeConfigSnapshot) -> Runtime:
        """Build a complete, unpublished graph for one desired revision."""
        prepared = RuntimeConfigSnapshot(
            snapshot.revision,
            snapshot.document,
            snapshot.updated_at,
            apply_status="applied",
            applied_revision=snapshot.revision,
        )
        if self._registry_discoverer is None:
            return await build_runtime(repository=self.repository, snapshot=prepared, event_sink=self.events)
        signals, venues, markets = await self._registry_discoverer(prepared.document)
        return await build_runtime(
            repository=self.repository,
            snapshot=prepared,
            event_sink=self.events,
            signal_registry=signals,
            venue_registry=venues,
            market_registry=markets,
        )

    async def publish_candidate(self, candidate: Runtime, snapshot: RuntimeConfigSnapshot) -> None:
        """Atomically make an already prepared graph the only executable graph."""
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is unavailable")
            if self._application_owner is None:
                raise RuntimeError("runtime application barrier is required")
            retired = tuple(self.sessions.values())
            self.snapshot = snapshot
            self.sessions = candidate.sessions
            self.cycle = candidate.cycle
            self.signal_registry = candidate.signal_registry
            self.market_registry = candidate.market_registry
            self.venue_registry = candidate.venue_registry
            self._session_keys = candidate._session_keys
            candidate.sessions = {}
            candidate._session_keys = {}
            candidate.cycle = None
            await self._retire_sessions(retired)

    async def activate_applied(self, snapshot: RuntimeConfigSnapshot) -> None:
        """Expose a committed revision only after its owners have started."""
        async with self._lifecycle_lock:
            if self._application_owner is None:
                raise RuntimeError("runtime application barrier is required")
            if self.snapshot.revision != snapshot.revision:
                raise RuntimeError("runtime revision changed during application")
            self.snapshot = snapshot

    async def fail_closed(self, snapshot: RuntimeConfigSnapshot) -> None:
        """Retain the failed desired document while ensuring no executable graph remains."""
        async with self._lifecycle_lock:
            if self._application_owner is None:
                raise RuntimeError("runtime application barrier is required")
            retired = tuple(self.sessions.values())
            self.snapshot = snapshot
            self.sessions = {}
            self._session_keys = {}
            self.cycle = None
            await self._retire_sessions(retired)

    async def _reload_for_cycle_locked(self) -> TradingCycle | None:
        if self._active_leases == 0:
            await self._cleanup_retired_sessions()
        candidate_snapshot = await self.repository.get_or_create()
        if candidate_snapshot.revision == self.snapshot.revision:
            candidate_signals = self.signal_registry
            candidate_venues = self.venue_registry
            candidate_markets = self.market_registry
        elif self._registry_discoverer is None:
            candidate_signals, candidate_venues, candidate_markets = await _discover_registry_graph(
                candidate_snapshot.document,
                self.events,
                self.repository,
            )
        else:
            candidate_signals, candidate_venues, candidate_markets = await self._registry_discoverer(
                candidate_snapshot.document
            )
        _validate_snapshot(candidate_snapshot, candidate_signals, candidate_venues, candidate_markets)
        if candidate_snapshot.setup_required:
            retired = tuple(self.sessions.values())
            self.snapshot, self.sessions, self.cycle, self._session_keys = candidate_snapshot, {}, None, {}
            self.signal_registry, self.venue_registry, self.market_registry = (
                candidate_signals,
                candidate_venues,
                candidate_markets,
            )
            await self._retire_sessions(retired)
            return None

        candidate_sessions, candidate_keys, opened = await _candidate_sessions(
            candidate_snapshot, self.repository, candidate_venues, self.sessions, self._session_keys
        )
        try:
            candidate_cycle = _assemble_cycle(
                candidate_snapshot,
                self.repository,
                candidate_sessions,
                candidate_signals,
                candidate_markets,
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
        self.signal_registry, self.venue_registry, self.market_registry = (
            candidate_signals,
            candidate_venues,
            candidate_markets,
        )
        await self._retire_sessions(retired)
        return candidate_cycle

    async def _retire_sessions(self, sessions) -> None:
        if self._active_leases:
            self._deferred_retired.update((id(session), session) for session in sessions)
            return
        await self._cleanup_retired_sessions(sessions, retry_pending=False)

    async def close(self) -> None:
        if not self._closed:
            self._closing = True
        completion = self._close_completion
        if completion is None or (completion.done() and self._pending_retired):
            completion = asyncio.create_task(self._close_owned())
            self._close_completion = completion
        await wait_for_owned(completion)

    async def _close_owned(self) -> None:
        async with self._lifecycle_lock:
            if self._closed and not self._pending_retired:
                return
            self._closing = True
            wait_for_leases = self._active_leases > 0
        if wait_for_leases:
            await self._leases_drained.wait()
        async with self._lifecycle_lock:
            first_close = not self._closed
            if not first_close and not self._pending_retired:
                return
            self._closed = True
            targets = tuple(self._pending_retired.values())
            if first_close:
                targets += tuple(self._deferred_retired.values()) + tuple(self.sessions.values())
            self._pending_retired.clear()
            self._deferred_retired.clear()
        result = await _close_session_batch(targets)
        async with self._lifecycle_lock:
            self._pending_retired = {id(session): session for session in result.failed_sessions}
        _raise_close_control(result)
        if result.ordinary_failures:
            raise RuntimeError("failed to close venue session") from None

    async def _cleanup_retired_sessions(self, sessions=(), *, retry_pending: bool = True) -> None:
        carried = {} if retry_pending else dict(self._pending_retired)
        targets = tuple(self._pending_retired.values()) if retry_pending else ()
        targets += tuple(sessions)
        result = await _close_session_batch(targets)
        carried.update((id(session), session) for session in result.failed_sessions)
        self._pending_retired = carried
        if result.ordinary_failures:
            logger.warning(
                "retired venue session cleanup pending",
                extra={"pending_count": len(self._pending_retired)},
            )
        _raise_close_control(result)


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
    if signal_registry is None and venue_registry is None and market_registry is None:

        async def registry_discoverer(document):
            return await _discover_registry_graph(document, routed_events, runtime_repository)

        signals, venues, markets = await registry_discoverer(frozen.document)
    else:
        signals = signal_registry or SignalComponentRegistry.discover(frozen.document, routed_events)
        venues = venue_registry or VenueAdapterRegistry.discover(
            connection.adapter_id for connection in frozen.document.execution.connections
        )
        markets = market_registry or MarketSourceRegistry.discover(frozen.document.market_data)

        async def registry_discoverer(_document):
            return signals, venues, markets

    validate_runtime_document(
        frozen.document,
        set(signals.installed_ids()),
        set(venues.installed_ids()),
        set(markets.installed_ids()),
    )
    if frozen.setup_required:
        return Runtime(
            frozen,
            runtime_repository,
            None,
            {},
            signals,
            markets,
            venues,
            routed_events,
            _registry_discoverer=registry_discoverer,
        )

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
        _registry_discoverer=registry_discoverer,
        _session_keys=session_keys,
    )


async def _discover_registry_graph(document, events, repository):
    try:
        llm_gateway_key = (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token
    except CredentialNotConfigured:
        llm_gateway_key = ""
    return (
        SignalComponentRegistry.discover(document, events, llm_gateway_key=llm_gateway_key),
        VenueAdapterRegistry.discover(connection.adapter_id for connection in document.execution.connections),
        MarketSourceRegistry.discover(document.market_data),
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
    services = {
        connection_id: VenueExecutionService(
            session,
            live_order_execution_enabled=frozen.document.execution.live_order_execution_enabled,
        )
        for connection_id, session in sessions.items()
    }
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
        snapshot=frozen,
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
            if not connection.enabled or connection.canary_only:
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
    result = await _close_session_batch(sessions)
    _raise_close_control(result)


async def _close_session_batch(sessions) -> _CloseBatchResult:
    unique = tuple({id(session): session for session in sessions}.values())
    if not unique:
        return _CloseBatchResult((), (), None)

    tasks = tuple(asyncio.create_task(session.close()) for session in unique)
    completion = asyncio.gather(*tasks, return_exceptions=True)
    external_cancellation = None
    try:
        outcomes = await wait_for_owned(completion)
    except asyncio.CancelledError as error:
        external_cancellation = error
        outcomes = completion.result()

    ordinary_failures = tuple(
        session for session, outcome in zip(unique, outcomes, strict=True) if isinstance(outcome, Exception)
    )
    control_failures = tuple(
        (session, outcome)
        for session, outcome in zip(unique, outcomes, strict=True)
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception)
    )
    return _CloseBatchResult(ordinary_failures, control_failures, external_cancellation)


def _raise_close_control(result: _CloseBatchResult) -> None:
    control = _close_control(result)
    if control is not None:
        raise control


def _close_control(result: _CloseBatchResult) -> BaseException | None:
    if result.control_failures:
        return result.control_failures[0][1]
    return result.external_cancellation
