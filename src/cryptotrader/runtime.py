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
from cryptotrader.decision.analysis import SignalAnalysisService
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.decision.read_service import DecisionReadService
from cryptotrader.decision.service import RunService
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
from cryptotrader.tasks import BackgroundTaskManager
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
    journal: MultiVenueCycleStore = field(init=False)
    read_service: DecisionReadService = field(init=False)
    run_service: RunService = field(init=False)
    task_manager: BackgroundTaskManager = field(init=False)

    def __post_init__(self) -> None:
        self._leases_drained.set()
        self.journal = MultiVenueCycleStore(getattr(self.repository, "database_url", None))
        from cryptotrader.signals.evaluation import EvaluationService
        from cryptotrader.signals.evaluation_store import EvaluationStore

        self.evaluation_store = EvaluationStore(getattr(self.repository, "database_url", None))
        self.evaluation_service = EvaluationService(self.journal, self.evaluation_store)
        self.read_service = DecisionReadService(self.journal)
        self.task_manager = BackgroundTaskManager()
        from cryptotrader.backtest.service import BacktestService

        self.backtest_service = (
            BacktestService(repository=self.repository, task_manager=self.task_manager)
            if getattr(self.repository, "database_url", None)
            else None
        )
        self.approvals = BookApprovalStore(
            getattr(self.repository, "database_url", None),
            approval_ttl_minutes=self.snapshot.document.hitl.approval_ttl_minutes,
        )
        self.run_service = RunService(
            analysis_provider=self.analysis_for_revision,
            journal=self.journal,
            task_manager=self.task_manager,
            runtime=self,
            events=self.events,
        )
        self.refresh_owners = None
        self.clear_owners = None
        self.account_store = getattr(self.repository, "account_store", None)
        if self.account_store is not None:
            from cryptotrader.accounts.sync import AccountSyncService

            self.venue_registry.bind_account_store(self.account_store)
            self.account_sync = AccountSyncService(self.account_store, self.account_session)
            from cryptotrader.accounts.operations import AccountOperationService

            self.account_operations = AccountOperationService(self)

        self.alert_owner = None
        if getattr(self.repository, "database_url", None):
            from cryptotrader.alerts.owner import AlertOwner
            from cryptotrader.alerts.recovery import AlertRecovery
            from cryptotrader.alerts.service import AlertService, DeliveryService
            from cryptotrader.alerts.store import AlertStore

            self.alert_store = AlertStore(self.repository.database_url)
            self.alerts = AlertService(self.alert_store, self.notification_config)
            self.deliveries = DeliveryService(self.alert_store, self.notification_config)
            recovery = AlertRecovery(
                self.alerts,
                journal=self.journal,
                approvals=self.approvals,
                accounts=self.account_store,
                operations=self.account_operations.store if self.account_store is not None else None,
                scope_provider=self.alert_scopes,
            )
            self.alert_owner = AlertOwner(recovery, self.deliveries)
            self.events.observe(self.alert_owner)

    async def notification_config(self):
        return (await self.repository.get_existing()).document.notifications

    async def alert_scopes(self):
        from cryptotrader.configuration.catalog import require_environment

        document = (await self.repository.get_existing()).document
        return {
            connection.id: require_environment(connection.adapter_id, connection.environment).capital_scope
            for connection in document.execution.connections
        }

    @asynccontextmanager
    async def account_session(self, connection_id):
        """Explicit read-only lifecycle, independent of execution admission."""
        current = await self.repository.get_existing()
        connection = next((item for item in current.document.execution.connections if item.id == connection_id), None)
        if connection is None:
            raise LookupError("account connection is not configured")
        credentials = (
            await self.repository.reveal_credentials(connection.credential_ref) if connection.credential_ref else None
        )
        # A failed configuration application may have introduced a new registration;
        # read its persisted configuration without replacing the execution graph.
        registry = self.venue_registry
        if connection.adapter_id not in registry.registered_ids():
            registry = VenueAdapterRegistry.discover((connection.adapter_id,))
        registry.bind_account_store(self.account_store)
        session = await registry.require(connection.adapter_id).connect(connection, credentials)
        try:
            yield session
        finally:
            await session.close()

    def approval_reader(self):
        """Journal/approval transitions need no connected account."""
        cycle = _assemble_cycle(
            self.snapshot, self.repository, {}, self.signal_registry, self.market_registry, self.events
        )
        cycle.journal = self.journal
        cycle.approvals = self.approvals
        return cycle

    def _configure_approval_policy(self, snapshot: RuntimeConfigSnapshot) -> None:
        self.approvals.configure_ttl(snapshot.document.hitl.approval_ttl_minutes)

    async def apply_automation(self, document, expected_revision):
        """CAS the one automation flag and publish it through the existing graph barrier."""
        from cryptotrader.runtime_config.repository import RevisionConflict

        snapshot = await self.repository.get_or_create()
        if snapshot.revision != expected_revision:
            raise RevisionConflict(expected_revision, snapshot.revision)
        candidate = await self.prepare_candidate(
            RuntimeConfigSnapshot(expected_revision + 1, document, snapshot.updated_at)
        )
        async with self.application_barrier():
            try:
                saved = await self.repository.replace(expected_revision, document)
            except BaseException:
                await candidate.close()
                raise
            try:
                await self.publish_candidate(candidate, saved)
                if self.refresh_owners is not None:
                    await self.refresh_owners(saved)
                applied = RuntimeConfigSnapshot(
                    saved.revision,
                    saved.document,
                    saved.updated_at,
                    apply_status="applied",
                    applied_revision=saved.revision,
                )
                await self.activate_applied(applied)
                return await self.repository.mark_applied(saved.revision)
            except BaseException:
                await wait_for_owned(asyncio.create_task(self._fail_automation(saved, candidate)))
                raise

    async def _fail_automation(self, saved, candidate):
        cleanup_incomplete = False
        for close in (self.clear_owners, candidate.close):
            if close is not None:
                try:
                    await close()
                except BaseException:
                    cleanup_incomplete = True
        error = "runtime application failed" + (": cleanup incomplete" if cleanup_incomplete else "")
        try:
            failed = await self.repository.mark_failed(saved.revision, error)
        except BaseException:
            failed = RuntimeConfigSnapshot(
                saved.revision,
                saved.document,
                saved.updated_at,
                apply_status="failed",
                applied_revision=saved.applied_revision,
                apply_error=error,
            )
        await self.fail_closed(failed)

    async def analysis_for_revision(self, expected_revision):
        """Pin market/component instances and their exact snapshot without a venue lease."""
        async with self._lifecycle_lock:
            if self._closing or self._closed or self.application_in_progress:
                raise RuntimeLeaseUnavailableError("runtime is unavailable")
            snapshot = await self.repository.get_or_create()
            if snapshot.revision != expected_revision:
                raise ValueError("configuration revision changed")
            if snapshot.revision == self.snapshot.revision:
                signals, markets = self.signal_registry, self.market_registry
            else:
                signals, _, markets = await self._registry_discoverer(snapshot.document)
            return snapshot, SignalAnalysisService(
                market_source=markets.require(snapshot.document.market_data.source_id),
                registry=signals,
                runner=ComponentRunner(self.events),
                fusion=WeightedSignalFusion(),
                decisions=DecisionEngine(),
            )

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

    async def wait_for_execution_idle(self):
        """Caller owns application_barrier; its admission block makes this a real drain."""
        await self._leases_drained.wait()

    @asynccontextmanager
    async def account_operation_lease(self):
        """Pin manual work for removal/shutdown, without blocking unrelated pool admission."""
        async with self._lifecycle_lock:
            if self._closing or self._closed or self._application_owner is not None:
                raise RuntimeLeaseUnavailableError("runtime application is in progress")
            self._active_leases += 1
            self._leases_drained.clear()
        try:
            yield
        finally:
            await wait_for_owned(asyncio.create_task(self._release_cycle_lease()))

    @asynccontextmanager
    async def cycle_lease(self, *, expected_revision=None, confirmed_book_ids=None, origin="manual", pair=None):
        """Publish and pin one frozen runtime graph for an execution owner."""

        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeLeaseUnavailableError("runtime is unavailable")
            if self._application_owner is not None:
                raise RuntimeLeaseUnavailableError("runtime application is in progress")
            cycle = await self._reload_for_cycle_locked(
                expected_revision=expected_revision, confirmed_book_ids=confirmed_book_ids, origin=origin, pair=pair
            )
            if cycle is None:
                raise RuntimeLeaseUnavailableError("runtime is unavailable")
            self._active_leases += 1
            self._leases_drained.clear()
        try:
            yield cycle
        finally:
            await wait_for_owned(asyncio.create_task(self._release_cycle_lease()))

    @asynccontextmanager
    async def execution_lease(self, pair: str, *, expected_revision=None, confirmed_book_ids=None, origin="manual"):
        """Admit exactly one production cycle for a canonical pair.

        Unlike the graph lease this is a strict distributed ownership lease:
        an absent or unhealthy Redis is an execution refusal.
        """
        from cryptotrader.pair import Pair

        canonical_pair = Pair.parse(pair).canonical()
        async with self.cycle_lease(
            expected_revision=expected_revision,
            confirmed_book_ids=confirmed_book_ids,
            origin=origin,
            pair=canonical_pair,
        ) as cycle:
            redis_url = cycle.snapshot.document.infrastructure.redis_url.strip()
            if not redis_url:
                raise RuntimeLeaseUnavailableError("Redis is required for production execution")
            yield cycle

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
            return await build_runtime(
                repository=self.repository, snapshot=prepared, event_sink=self.events, recover_unfinished=False
            )
        signals, venues, markets = await self._registry_discoverer(prepared.document)
        return await build_runtime(
            repository=self.repository,
            snapshot=prepared,
            event_sink=self.events,
            signal_registry=signals,
            venue_registry=venues,
            market_registry=markets,
            recover_unfinished=False,
        )

    async def publish_candidate(self, candidate: Runtime, snapshot: RuntimeConfigSnapshot) -> None:
        """Atomically make an already prepared graph the only executable graph."""
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("runtime is unavailable")
            if self._application_owner is None:
                raise RuntimeError("runtime application barrier is required")
            retired = tuple(self.sessions.values())
            self._configure_approval_policy(snapshot)
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
            self._configure_approval_policy(snapshot)
            self.snapshot = snapshot

    async def fail_closed(self, snapshot: RuntimeConfigSnapshot) -> None:
        """Retain the failed desired document while ensuring no executable graph remains."""
        async with self._lifecycle_lock:
            if self._application_owner is None:
                raise RuntimeError("runtime application barrier is required")
            retired = tuple(self.sessions.values())
            self._configure_approval_policy(snapshot)
            self.snapshot = snapshot
            self.sessions = {}
            self._session_keys = {}
            self.cycle = None
            await self._retire_sessions(retired)

    async def _reload_for_cycle_locked(
        self, *, expected_revision=None, confirmed_book_ids=None, origin="manual", pair=None
    ) -> TradingCycle | None:
        if self._active_leases == 0:
            await self._cleanup_retired_sessions()
        candidate_snapshot = await self.repository.get_or_create()
        if expected_revision is not None and candidate_snapshot.revision != expected_revision:
            raise RuntimeLeaseUnavailableError("configuration revision changed")
        document = candidate_snapshot.document
        if origin in {"scheduled", "trigger"} and (
            not document.scheduler.automation_enabled
            or not (document.scheduler.enabled if origin == "scheduled" else document.triggers.enabled)
        ):
            raise RuntimeLeaseUnavailableError("automation paused")
        if pair is not None and pair not in document.execution.pairs:
            raise RuntimeLeaseUnavailableError("pair is outside execution scope")
        from cryptotrader.decision.readiness import book_scope

        scope = await book_scope(candidate_snapshot, self.repository)
        eligible = {book.book_id for book in scope if book.eligible}
        selected = eligible if confirmed_book_ids is None else set(confirmed_book_ids)
        if (confirmed_book_ids is not None and not selected) or not selected <= eligible:
            raise RuntimeLeaseUnavailableError("confirmed books are not eligible")
        connection_ids = {
            allocation.connection_id
            for book in document.execution.books
            if book.id in selected
            for allocation in book.allocations
            if allocation.enabled
        }
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
        if not candidate_snapshot.operational or not selected:
            retired = tuple(self.sessions.values())
            self._configure_approval_policy(candidate_snapshot)
            self.snapshot, self.sessions, self.cycle, self._session_keys = candidate_snapshot, {}, None, {}
            self.signal_registry, self.venue_registry, self.market_registry = (
                candidate_signals,
                candidate_venues,
                candidate_markets,
            )
            await self._retire_sessions(retired)
            return None

        candidate_sessions, candidate_keys, opened = await _candidate_sessions(
            candidate_snapshot, self.repository, candidate_venues, self.sessions, self._session_keys, connection_ids
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
            candidate_cycle.journal = self.journal
        except BaseException:
            await _close_candidate_sessions(opened)
            raise

        retired = tuple(
            session
            for session in self.sessions.values()
            if all(session is not candidate for candidate in candidate_sessions.values())
        )
        self._configure_approval_policy(candidate_snapshot)
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
        await self.task_manager.shutdown()
        if hasattr(self, "account_operations"):
            await self.account_operations.close()
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
    recover_unfinished: bool = True,
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
        set(signals.registered_ids()),
        set(venues.registered_ids()),
        set(markets.registered_ids()),
    )
    runtime = Runtime(
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
    if recover_unfinished:
        from datetime import UTC, datetime

        await runtime.journal.interrupt_unfinished(datetime.now(UTC))
    return runtime


async def _discover_registry_graph(document, events, repository):
    from cryptotrader.runtime_config.repository import NEWS_PROVIDER_CREDENTIAL_REF

    try:
        llm_gateway_key = (await repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token
    except CredentialNotConfigured:
        llm_gateway_key = ""
    try:
        news_provider_key = (await repository.reveal_token(NEWS_PROVIDER_CREDENTIAL_REF)).token
    except CredentialNotConfigured:
        news_provider_key = ""
    return (
        SignalComponentRegistry.discover(document, events, llm_gateway_key=llm_gateway_key),
        VenueAdapterRegistry.discover(connection.adapter_id for connection in document.execution.connections),
        MarketSourceRegistry.discover(document.market_data, news_provider_key=news_provider_key),
    )


def _validate_snapshot(snapshot, signals, venues, markets) -> None:
    validate_runtime_document(
        snapshot.document,
        set(signals.registered_ids()),
        set(venues.registered_ids()),
        set(markets.registered_ids()),
    )


def _assemble_cycle(snapshot, repository, sessions, signals, markets, event_sink) -> TradingCycle:
    frozen = snapshot
    runtime_repository = repository
    configured_connections = {connection.id: connection for connection in frozen.document.execution.connections}
    services = {
        connection_id: VenueExecutionService(
            session,
            connection=configured_connections[connection_id],
            live_order_execution_enabled=frozen.document.execution.live_order_execution_enabled,
            account_store=getattr(repository, "account_store", None),
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
    from cryptotrader.configuration.parameters import DefaultMarketSourceParameters

    market_parameters = (
        DefaultMarketSourceParameters.model_validate(dict(frozen.document.market_data.parameters))
        if frozen.document.market_data.source_id == "default"
        else DefaultMarketSourceParameters()
    )
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
        portfolios=PortfolioAggregator(getattr(repository, "account_store", None)),
        allocation_policy=allocation,
        book_risk=book_risk,
        connection_risk=connection_risk,
        planner=planner,
        approvals=BookApprovalStore(database_url, approval_ttl_minutes=frozen.document.hitl.approval_ttl_minutes),
        coordinator=ExecutionCoordinator(services),
        journal=MultiVenueCycleStore(database_url),
        events=event_sink,
        exit_requirement=DataRequirements(
            candles=(CandleRequirement(frozen.document.market_data.timeframe, market_parameters.limit),)
        ),
    )


async def _candidate_sessions(
    snapshot,
    repository,
    venue_registry,
    current_sessions,
    current_keys,
    connection_ids,
) -> tuple[dict[str, VenueSession], dict[str, tuple[object, ...]], tuple[VenueSession, ...]]:
    account_store = getattr(repository, "account_store", None)
    if account_store is not None:
        venue_registry.bind_account_store(account_store)
    sessions: dict[str, VenueSession] = {}
    keys: dict[str, tuple[object, ...]] = {}
    opened: list[VenueSession] = []
    try:
        for connection in snapshot.document.execution.connections:
            if connection.id not in connection_ids or not connection.enabled or connection.canary_only:
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
