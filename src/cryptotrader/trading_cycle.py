"""一次平台无关信号驱动多个隔离执行资金池的唯一主链。"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cryptotrader.cycle_events import CycleEvent
from cryptotrader.decision.analysis import AnalysisFailure, SignalAnalysisService
from cryptotrader.decision.models import CycleOutcome, CycleRequest
from cryptotrader.decision.service import configuration_summary
from cryptotrader.execution_ownership import ExecutionOwnership
from cryptotrader.hitl.store import ApprovalNotFound
from cryptotrader.journal.models import (
    BookCycleResult,
    BookHitlSnapshot,
    BookPreparationFailure,
    DecisionRun,
    MultiVenueCycleRecord,
)
from cryptotrader.risk.book_state import BookRiskStateStore
from cryptotrader.risk.models import BookRiskRequest
from cryptotrader.signals.models import DataRequirements

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cryptotrader.decision.models import TargetPosition, TradePlan
    from cryptotrader.execution.models import ExecutionBook
    from cryptotrader.journal.store import MultiVenueCycleStore
    from cryptotrader.signals.fusion import FusedSignal
    from cryptotrader.signals.models import ComponentSignal, SignalContext
    from cryptotrader.venues.protocol import VenueSession


@dataclass(frozen=True)
class _PreparedBook:
    book: ExecutionBook
    result: BookCycleResult


class TradingCycle:
    """Freeze one config snapshot, one market opinion, and independent book outcomes."""

    def __init__(
        self,
        *,
        snapshot,
        repository,
        market_source,
        registry,
        runner,
        fusion,
        decisions,
        exits,
        sessions: Mapping[str, VenueSession],
        portfolios,
        allocation_policy,
        book_risk,
        connection_risk,
        planner,
        approvals,
        coordinator,
        journal: MultiVenueCycleStore,
        events,
        exit_requirement: DataRequirements | None = None,
        clock: Callable[[], datetime] | None = None,
        ownership=None,
        risk_states=None,
    ) -> None:
        self.snapshot = snapshot
        self.repository = repository
        self.market_source = market_source
        self.registry = registry
        self.runner = runner
        self.fusion = fusion
        self.decisions = decisions
        self.exits = exits
        self.sessions = dict(sessions)
        self.portfolios = portfolios
        self.allocation_policy = allocation_policy
        self.book_risk = book_risk
        self.connection_risk = connection_risk
        self.planner = planner
        self.approvals = approvals
        self.coordinator = coordinator
        self.journal = journal
        self.events = events
        self.exit_requirement = exit_requirement or DataRequirements()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.ownership = ownership or ExecutionOwnership(snapshot.document.infrastructure.redis_url)
        self.risk_states = risk_states or BookRiskStateStore(getattr(repository, "account_store", None))
        self.analysis = SignalAnalysisService(
            market_source=market_source,
            registry=registry,
            runner=runner,
            fusion=fusion,
            decisions=decisions,
            clock=self.clock,
        )

    async def run(self, request: CycleRequest) -> CycleOutcome:
        snapshot = self.snapshot
        if not snapshot.operational:
            raise RuntimeError("runtime configuration is not applied")
        created_at = self._now()
        cycle_id = request.decision_id or str(uuid4())
        queued = MultiVenueCycleRecord(
            cycle_id,
            snapshot.revision,
            snapshot.document.market_data.source_id,
            (),
            None,
            None,
            (),
            "queued",
            "not_started",
            False,
            created_at,
            DecisionRun(
                request.pair.canonical(), request.mode, request.origin, configuration_summary(snapshot), None, None
            ),
        )
        if request.decision_id is None:
            await self.journal.save(queued)
        else:
            queued = await self._required_record(request.decision_id)
            if queued.cycle_status != "queued" or queued.config_revision != snapshot.revision:
                raise ValueError("queued decision does not match the execution snapshot")
            created_at = queued.created_at
        running = replace(queued, cycle_status="running")
        await self.journal.replace(running)
        cycle_scope = getattr(self.events, "cycle", None)
        event_scope = cycle_scope(cycle_id, snapshot.revision) if callable(cycle_scope) else nullcontext()
        with event_scope:
            try:
                await self._publish(
                    "cycle_started", cycle_id=cycle_id, config_revision=snapshot.revision, pair=request.pair.canonical()
                )
                result = await self.analysis.analyze(request.pair, snapshot, created_at)
                if result.failure is not None:
                    return await self.save_failed_analysis(result, request, created_at, running=running)
                await self._publish(
                    "context_ready",
                    cycle_id=cycle_id,
                    config_revision=snapshot.revision,
                    market_data_source_id=result.context.market_data_source_id,
                )
                await self._publish(
                    "fusion_completed",
                    cycle_id=cycle_id,
                    config_revision=snapshot.revision,
                    score=result.fused_signal.score,
                )
                await self._publish(
                    "decision_created",
                    cycle_id=cycle_id,
                    config_revision=snapshot.revision,
                    side=result.target_position.side,
                    size_ratio=result.target_position.size_ratio,
                )
                return await self.prepare_and_execute_books(result, request, created_at, running=running)
            except asyncio.CancelledError:
                current = await self.journal.get(cycle_id)
                if current is not None and current.cycle_status in {"queued", "running"}:
                    await self.journal.replace(
                        replace(
                            current,
                            cycle_status="cancelled",
                            run=replace(
                                current.run,
                                finished_at=self._now(),
                                failure=AnalysisFailure(code="cancelled", stage="runtime", message="运行已取消。"),
                            ),
                        )
                    )
                raise
            except Exception:
                current = await self.journal.get(cycle_id)
                if current is not None and current.cycle_status in {"queued", "running"}:
                    await self.journal.replace(
                        replace(
                            current,
                            cycle_status="cycle_failed",
                            run=replace(
                                current.run,
                                finished_at=self._now(),
                                failure=AnalysisFailure(
                                    code="cycle_failed", stage="runtime", message="交易周期未完成。"
                                ),
                            ),
                        )
                    )
                await self._publish(
                    "cycle_failed", cycle_id=cycle_id, config_revision=snapshot.revision, status="cycle_failed"
                )
                raise

    async def save_failed_analysis(self, result, request, created_at, *, running):
        status = "component_failed" if result.failure.code == "component_failed" else "cycle_failed"
        record = replace(
            running,
            component_signals=result.component_signals,
            fused_signal=result.fused_signal,
            target_position=result.target_position,
            cycle_status=status,
            run=replace(running.run, finished_at=self._now(), failure=result.failure),
        )
        await self.journal.replace(record)
        await self._publish(
            "cycle_failed", cycle_id=record.cycle_id, config_revision=record.config_revision, status=status
        )
        return self._outcome(record)

    async def prepare_and_execute_books(self, result, request, created_at, *, running):
        snapshot = self.snapshot
        document = snapshot.document
        profile = document.signals.to_profile(snapshot.revision)
        books = tuple(
            book
            for book in document.execution.books
            if book.enabled and (request.confirmed_book_ids is None or book.id in request.confirmed_book_ids)
        )
        cycle_id = running.cycle_id
        context, signals, fused, target = (
            result.context,
            result.component_signals,
            result.fused_signal,
            result.target_position,
        )
        created_approval_ids = []
        initial_record_saved = False
        journal_write_attempted = False
        durable_record = None
        try:
            plan = self.exits.build_plan(context, target, signals, fused, profile)
            book_results = tuple(
                await asyncio.gather(
                    *(
                        self._process_book(
                            book,
                            request,
                            context,
                            plan,
                            cycle_id=cycle_id,
                            config_revision=snapshot.revision,
                            created_at=created_at,
                            created_approval_ids=created_approval_ids,
                        )
                        for book in books
                    )
                )
            )
            record = self._record(
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                market_data_source_id=context.market_data_source_id,
                signals=signals,
                fused=fused,
                target=target,
                book_results=book_results,
                created_at=created_at,
                run=replace(running.run, finished_at=self._now()),
            )
            journal_write_attempted = True
            await self.journal.replace(record)
            initial_record_saved = True
            durable_record = record
            await self._invalidate_unjournaled_approvals(
                cycle_id,
                created_approval_ids,
                durable_record=record,
            )
            await self._publish(
                "cycle_completed",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                status=record.cycle_status,
                execution_status=record.execution_status,
                requires_attention=record.requires_attention,
            )
            return self._outcome(record)
        except asyncio.CancelledError:
            await self._invalidate_unjournaled_approvals(
                cycle_id,
                created_approval_ids,
                durable_record=durable_record,
                reconcile_ambiguous_write=journal_write_attempted,
            )
            if not initial_record_saved:
                await self._record_or_save_empty(
                    cycle_id,
                    snapshot.revision,
                    context.market_data_source_id if context is not None else document.market_data.source_id,
                    signals,
                    fused,
                    target,
                    created_at,
                    "cancelled",
                )
            await self._publish(
                "cycle_cancelled",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                status="cancelled",
            )
            raise
        except Exception:
            await self._invalidate_unjournaled_approvals(
                cycle_id,
                created_approval_ids,
                durable_record=durable_record,
                reconcile_ambiguous_write=journal_write_attempted,
            )
            if initial_record_saved:
                record = await self._required_record(cycle_id)
            else:
                record = await self._record_or_save_empty(
                    cycle_id,
                    snapshot.revision,
                    context.market_data_source_id if context is not None else document.market_data.source_id,
                    signals,
                    fused,
                    target,
                    created_at,
                    "cycle_failed",
                )
            await self._publish(
                "cycle_failed",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                status="cycle_failed",
            )
            return self._outcome(record)

    async def execute_approved(self, approval_id: str) -> CycleOutcome:
        approval = await self.approvals.get(approval_id)
        if approval is None:
            raise LookupError("approval was not found")
        async with self.ownership.book(approval.book_id):
            return await self._execute_approved_locked(approval_id)

    async def _execute_approved_locked(self, approval_id: str) -> CycleOutcome:
        approval = await self.approvals.get(approval_id)
        if approval is None:
            raise LookupError("approval was not found")
        record = await self._required_record(approval.cycle_id)
        original = self._required_approval_book(record, approval_id, approval.book_id)
        if original.proposal != approval.proposal:
            raise ValueError("approval proposal does not match the journaled approval")
        snapshot = await self.repository.get_or_create()
        if snapshot.revision != approval.config_revision:
            await self.approvals.invalidate(approval_id)
            replacement = await self._persist_approval_transition(record, original, "invalidated")
            await self._publish(
                "book_execution_completed",
                cycle_id=record.cycle_id,
                book_id=original.book_id,
                config_revision=original.config_revision,
                status="approval_rejected",
            )
            return self._outcome(replacement)

        book = self._validated_execution_book(snapshot, approval.proposal)
        from cryptotrader.hitl.store import ApprovalInvalidated

        valid = False
        try:
            portfolio = await self.portfolios.read(book, self.sessions, approval.proposal.pair)
            state = await self.risk_states.update(book.id, tuple(c.account_snapshot for c in portfolio.connections))
            valid = await self.planner.validate_frozen(book, approval.proposal, portfolio, state, self.sessions)
        except asyncio.CancelledError:
            raise
        except Exception:
            valid = False
        if not valid:
            await self.approvals.invalidate(approval_id)
            return self._outcome(await self._persist_approval_transition(record, original, "invalidated"))

        try:
            proposal = await self.approvals.claim_for_execution(
                approval_id,
                current_revision=snapshot.revision,
            )
        except ApprovalInvalidated:
            replacement = await self._persist_approval_transition(record, original, "invalidated")
            return self._outcome(replacement)
        if proposal != approval.proposal:
            raise ValueError("claimed proposal does not match the validated approval")
        terminal = await self._execute_book(
            _PreparedBook(book, original),
            CycleRequest(proposal.pair),
            cycle_id=record.cycle_id,
            config_revision=record.config_revision,
            approval_id=approval_id,
        )
        replacement = await self._persist_book_transition(record, original, terminal)
        return self._outcome(replacement)

    async def _process_book(self, book, request, context, plan, **kwargs):
        from cryptotrader.cycle_lock import ExecutionLeaseUnavailableError

        try:
            async with self.ownership.book(book.id):
                prepared = await self._prepare_book(book, request, context, plan, **kwargs)
                if prepared.result.status != "ready":
                    return prepared.result
                return await self._execute_book(
                    prepared, request, cycle_id=kwargs["cycle_id"], config_revision=kwargs["config_revision"]
                )
        except ExecutionLeaseUnavailableError:
            return self._preparation_failure(book, request, kwargs["config_revision"], "portfolio")

    async def reject_approval(self, approval_id: str) -> CycleOutcome:
        existing = await self.approvals.get(approval_id)
        if existing is None:
            raise LookupError("approval was not found")
        record = await self._required_record(existing.cycle_id)
        original = self._required_approval_book(record, approval_id, existing.book_id)
        if original.proposal != existing.proposal:
            raise ValueError("approval proposal does not match the journaled approval")
        approval = await self.approvals.reject(approval_id)
        replacement = await self._persist_approval_transition(record, original, approval.status)
        return self._outcome(replacement)

    async def _prepare_book(  # noqa: C901 - stages are explicit audit boundaries
        self,
        book: ExecutionBook,
        request: CycleRequest,
        context: SignalContext,
        plan: TradePlan,
        *,
        cycle_id: str,
        config_revision: int,
        created_at: datetime,
        created_approval_ids: list[str],
    ) -> _PreparedBook:
        try:
            for allocation in book.allocations:
                if allocation.enabled:
                    instruments = await self.sessions[allocation.connection_id].list_instruments()
                    if not any(
                        i.pair == request.pair and i.market_type == request.pair.market_type and i.tradable
                        for i in instruments
                    ):
                        raise ValueError("member does not support requested instrument")
            portfolio = await self.portfolios.read(book, self.sessions, request.pair)
            state = await self.risk_states.update(book.id, tuple(c.account_snapshot for c in portfolio.connections))
        except asyncio.CancelledError:
            raise
        except Exception:
            return _PreparedBook(book, self._preparation_failure(book, request, config_revision, "portfolio"))

        try:
            if portfolio.total_equity is not None or plan.target.signed_ratio == 0:
                self.allocation_policy.allocate(plan.target, book, portfolio)
        except Exception:
            return _PreparedBook(
                book,
                self._preparation_failure(
                    book,
                    request,
                    config_revision,
                    "allocation",
                    portfolio=portfolio,
                ),
            )
        try:
            risk_request = BookRiskRequest(
                book,
                portfolio,
                Decimal(str(plan.target.signed_ratio)),
                request.pair,
                state,
            )
        except Exception:
            return _PreparedBook(
                book,
                self._preparation_failure(
                    book,
                    request,
                    config_revision,
                    "risk",
                    portfolio=portfolio,
                ),
            )
        try:
            proposal = await self.planner.propose(
                risk_request,
                self.sessions,
                pair=request.pair,
                stop_loss=self._decimal(plan.stop_loss),
                take_profit=self._decimal(plan.take_profit),
                config_revision=config_revision,
            )
            proposal = replace(
                proposal,
                connection_plans=tuple(replace(item, decision_id=cycle_id) for item in proposal.connection_plans),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _PreparedBook(
                book,
                self._preparation_failure(
                    book,
                    request,
                    config_revision,
                    "planning",
                    portfolio=portfolio,
                ),
            )

        await self._publish(
            "book_proposed",
            cycle_id=cycle_id,
            book_id=book.id,
            config_revision=config_revision,
            ready=proposal.ready,
        )
        if not proposal.ready:
            stage = "risk" if not proposal.risk.passed else "planning"
            return _PreparedBook(
                book,
                self._preparation_failure(
                    book,
                    request,
                    config_revision,
                    stage,
                    portfolio=portfolio,
                    proposal=proposal,
                ),
            )
        if book.hitl_required:
            approval_id = str(uuid4())
            created_approval_ids.append(approval_id)
            try:
                approval = await self.approvals.create(
                    proposal,
                    cycle_id=cycle_id,
                    approval_id=approval_id,
                    created_at=created_at,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return _PreparedBook(
                    book,
                    self._preparation_failure(
                        book,
                        request,
                        config_revision,
                        "planning",
                        portfolio=portfolio,
                    ),
                )
            result = BookCycleResult(
                book.id,
                book.capital_scope,
                config_revision,
                request.pair,
                proposal,
                portfolio,
                BookHitlSnapshot(approval.approval_id, "pending", config_revision),
                None,
                None,
                None,
                None,
                "awaiting_approval",
            )
            await self._publish(
                "book_awaiting_approval",
                cycle_id=cycle_id,
                book_id=book.id,
                config_revision=config_revision,
                approval_id=approval.approval_id,
            )
            return _PreparedBook(book, result)
        return _PreparedBook(
            book,
            BookCycleResult(
                book.id,
                book.capital_scope,
                config_revision,
                request.pair,
                proposal,
                portfolio,
                BookHitlSnapshot(None, "not_required", config_revision),
                None,
                None,
                None,
                None,
                "ready",
            ),
        )

    async def _execute_book(
        self,
        prepared: _PreparedBook,
        request: CycleRequest,
        *,
        cycle_id: str,
        config_revision: int,
        approval_id: str | None = None,
    ) -> BookCycleResult:
        proposal = prepared.result.proposal
        assert proposal is not None
        await self._publish(
            "book_execution_started",
            cycle_id=cycle_id,
            book_id=proposal.book_id,
            config_revision=config_revision,
        )
        execution = await self.coordinator.execute(proposal, frozen=approval_id is not None)
        for result in execution.connection_results:
            await self._publish(
                "connection_execution_completed",
                cycle_id=cycle_id,
                book_id=proposal.book_id,
                connection_id=result.connection_id,
                config_revision=config_revision,
                status=result.status,
                requires_attention=result.requires_attention,
            )
        portfolio_after = None
        portfolio_after_available = False
        reconciliation_required = False
        try:
            portfolio_after = await self.portfolios.read(prepared.book, self.sessions, request.pair)
            portfolio_after_available = True
            await self.risk_states.update(
                prepared.book.id, tuple(p.account_snapshot for p in portfolio_after.connections)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            reconciliation_required = True
        terminal = BookCycleResult(
            prepared.result.book_id,
            prepared.result.capital_scope,
            prepared.result.config_revision,
            prepared.result.pair,
            proposal,
            prepared.result.portfolio_before,
            BookHitlSnapshot(
                approval_id,
                "executed" if approval_id is not None else "not_required",
                prepared.result.config_revision,
            ),
            execution,
            None,
            portfolio_after,
            portfolio_after_available,
            execution.status,
            reconciliation_required,
        )
        await self._publish(
            "book_execution_completed",
            cycle_id=cycle_id,
            book_id=proposal.book_id,
            config_revision=config_revision,
            status=execution.status,
            requires_attention=execution.requires_attention or reconciliation_required,
        )
        return terminal

    @staticmethod
    def _preparation_failure(
        book: ExecutionBook,
        request: CycleRequest,
        config_revision: int,
        stage: str,
        *,
        portfolio=None,
        proposal=None,
    ) -> BookCycleResult:
        return BookCycleResult(
            book.id,
            book.capital_scope,
            config_revision,
            request.pair,
            proposal,
            portfolio,
            BookHitlSnapshot(None, "not_required", config_revision),
            None,
            BookPreparationFailure(stage),
            None,
            None,
            "failed",
        )

    async def _save_empty_record(
        self,
        cycle_id: str,
        config_revision: int,
        market_data_source_id: str,
        signals: tuple[ComponentSignal, ...],
        fused: FusedSignal | None,
        target: TargetPosition | None,
        created_at: datetime,
        status: str,
    ) -> MultiVenueCycleRecord:
        record = MultiVenueCycleRecord(
            cycle_id,
            config_revision,
            market_data_source_id,
            signals,
            fused,
            target,
            (),
            status,
            "not_started",
            False,
            created_at,
            DecisionRun(None, "trading", "manual", configuration_summary(self.snapshot), self._now(), None, ("pair",)),
        )
        await self.journal.save(record)
        return record

    async def _record_or_save_empty(
        self,
        cycle_id: str,
        config_revision: int,
        market_data_source_id: str,
        signals: tuple[ComponentSignal, ...],
        fused: FusedSignal | None,
        target: TargetPosition | None,
        created_at: datetime,
        status: str,
    ) -> MultiVenueCycleRecord:
        durable = await self.journal.get(cycle_id)
        if durable is not None:
            if durable.cycle_status in {"queued", "running"}:
                durable = replace(
                    durable,
                    component_signals=signals,
                    fused_signal=fused,
                    target_position=target,
                    cycle_status=status,
                    run=replace(
                        durable.run,
                        finished_at=self._now(),
                        failure=AnalysisFailure(code=status, stage="execution", message="交易周期未完成。"),
                    ),
                )
                await self.journal.replace(durable)
            return durable
        return await self._save_empty_record(
            cycle_id,
            config_revision,
            market_data_source_id,
            signals,
            fused,
            target,
            created_at,
            status,
        )

    @staticmethod
    def _record(
        *,
        cycle_id: str,
        config_revision: int,
        market_data_source_id: str,
        signals: tuple[ComponentSignal, ...],
        fused: FusedSignal,
        target: TargetPosition,
        book_results: tuple[BookCycleResult, ...],
        created_at: datetime,
        run: DecisionRun,
    ) -> MultiVenueCycleRecord:
        status = TradingCycle._cycle_status(book_results)
        execution_status = TradingCycle._execution_status(book_results)
        attention = any(
            item.reconciliation_required or (item.execution is not None and item.execution.requires_attention)
            for item in book_results
        )
        return MultiVenueCycleRecord(
            cycle_id,
            config_revision,
            market_data_source_id,
            signals,
            fused,
            target,
            book_results,
            status,
            execution_status,
            attention,
            created_at,
            run,
        )

    @staticmethod
    def _cycle_status(books: tuple[BookCycleResult, ...]) -> str:
        statuses = tuple(item.status for item in books)
        for state in ("awaiting_approval", "approval_rejected", "ready"):
            if state in statuses:
                return state
        if all(status == "failed" for status in statuses):
            if all(item.failure is not None and item.failure.stage == "risk" for item in books):
                return "risk_rejected"
            return "failed"
        if all(status == "completed" for status in statuses):
            return "completed"
        return "partial"

    @staticmethod
    def _execution_status(books: tuple[BookCycleResult, ...]) -> str:
        terminal = tuple(item for item in books if item.execution is not None)
        if not terminal:
            return "not_started"
        statuses = tuple(item.execution.status for item in terminal)
        if all(status == "failed" for status in statuses):
            return "failed"
        if len(terminal) != len(books):
            return "partial"
        if all(status == "completed" for status in statuses):
            return "completed"
        return "partial"

    @staticmethod
    def _outcome(record: MultiVenueCycleRecord) -> CycleOutcome:
        return CycleOutcome(
            record.cycle_id,
            record.config_revision,
            record.target_position,
            record.book_results,
            record.cycle_status,
            record.execution_status,
            record.requires_attention,
        )

    async def _required_record(self, cycle_id: str) -> MultiVenueCycleRecord:
        record = await self.journal.get(cycle_id)
        if record is None:
            raise LookupError("cycle was not found")
        return record

    def _validated_execution_book(self, snapshot, proposal):
        if not snapshot.operational or snapshot.revision != proposal.config_revision:
            raise ValueError("approved proposal revision is not active")
        book = next(
            (item for item in snapshot.document.execution.books if item.id == proposal.book_id and item.enabled),
            None,
        )
        if book is None or book.capital_scope != proposal.capital_scope:
            raise ValueError("approved book is absent from its configuration revision")
        enabled_allocations = {item.connection_id for item in book.allocations if item.enabled}
        configured_connections = {item.id for item in snapshot.document.execution.connections if item.enabled}
        planned_connections = {item.connection_id for item in proposal.connection_plans}
        if not planned_connections <= enabled_allocations or not planned_connections <= configured_connections:
            raise ValueError("approved proposal connections do not match the active book")
        if not planned_connections <= self.sessions.keys():
            raise ValueError("approved proposal session is unavailable")
        return book

    async def _persist_approval_transition(
        self,
        record: MultiVenueCycleRecord,
        original: BookCycleResult,
        status: str,
    ) -> MultiVenueCycleRecord:
        if status not in {"rejected", "invalidated"}:
            raise ValueError("unsupported approval transition")
        transitioned = replace(
            original,
            hitl=BookHitlSnapshot(original.hitl.approval_id, status, original.config_revision),
            status="approval_rejected",
        )
        return await self._persist_book_transition(record, original, transitioned)

    async def _persist_book_transition(
        self,
        record: MultiVenueCycleRecord,
        original: BookCycleResult,
        terminal: BookCycleResult,
    ) -> MultiVenueCycleRecord:
        replacement = self._replace_book(record, terminal)
        try:
            await self.journal.replace(replacement)
            return replacement
        except ValueError:
            latest = await self._required_record(record.cycle_id)
            latest_original = self._required_approval_book(
                latest,
                original.hitl.approval_id or "",
                original.book_id,
            )
            prior_by_book = {item.book_id: item for item in record.book_results}
            sibling_changed = any(
                item.book_id != original.book_id and item != prior_by_book[item.book_id] for item in latest.book_results
            )
            if latest_original != original or not sibling_changed:
                raise
            merged = self._replace_book(latest, terminal)
            await self.journal.replace(merged)
            return merged

    async def _invalidate_unjournaled_approvals(
        self,
        cycle_id: str,
        approval_ids: list[str],
        *,
        durable_record: MultiVenueCycleRecord | None = None,
        reconcile_ambiguous_write: bool = False,
    ) -> None:
        if not approval_ids:
            return
        record = durable_record
        if record is None and reconcile_ambiguous_write:
            record = await self.journal.get(cycle_id)
        if record is not None:
            durable_ids = {item.hitl.approval_id for item in record.book_results if item.hitl.approval_id is not None}
        else:
            durable_ids = set()
        pending_cleanup = tuple(
            self.approvals.invalidate(approval_id) for approval_id in approval_ids if approval_id not in durable_ids
        )
        if pending_cleanup:
            outcomes = await asyncio.gather(*pending_cleanup, return_exceptions=True)
            if any(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes):
                raise asyncio.CancelledError
            fatal = next(
                (
                    outcome
                    for outcome in outcomes
                    if isinstance(outcome, BaseException) and not isinstance(outcome, Exception)
                ),
                None,
            )
            if fatal is not None:
                raise fatal
            if any(
                isinstance(outcome, Exception) and not isinstance(outcome, ApprovalNotFound) for outcome in outcomes
            ):
                raise RuntimeError("failed to invalidate unjournaled approval") from None

    @staticmethod
    def _required_approval_book(
        record: MultiVenueCycleRecord,
        approval_id: str,
        book_id: str,
    ) -> BookCycleResult:
        matches = tuple(
            item for item in record.book_results if item.book_id == book_id and item.hitl.approval_id == approval_id
        )
        if len(matches) != 1 or matches[0].status != "awaiting_approval":
            raise ValueError("approval does not match an awaiting book in the cycle")
        return matches[0]

    @staticmethod
    def _replace_book(record: MultiVenueCycleRecord, replacement: BookCycleResult) -> MultiVenueCycleRecord:
        books = tuple(replacement if item.book_id == replacement.book_id else item for item in record.book_results)
        return replace(
            record,
            book_results=books,
            cycle_status=TradingCycle._cycle_status(books),
            execution_status=TradingCycle._execution_status(books),
            requires_attention=any(
                item.reconciliation_required or (item.execution is not None and item.execution.requires_attention)
                for item in books
            ),
        )

    async def _publish(self, name: str, **data: Any) -> None:
        await self.events.publish(CycleEvent(name, data))

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("cycle clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _decimal(value: float | None) -> Decimal | None:
        return None if value is None else Decimal(str(value))

    def _validate_context(self, context: SignalContext, pair, as_of: datetime) -> None:
        if context.pair != pair or context.as_of != as_of:
            raise ValueError("market source context identity mismatch")
        if context.market_data_source_id != self.market_source.id:
            raise ValueError("market source context source_id mismatch")
