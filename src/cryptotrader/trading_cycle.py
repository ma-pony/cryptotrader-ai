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
from cryptotrader.decision.models import CycleOutcome, CycleRequest
from cryptotrader.hitl.store import ApprovalNotFound
from cryptotrader.journal.models import (
    BookCycleResult,
    BookHitlSnapshot,
    BookPreparationFailure,
    MultiVenueCycleRecord,
)
from cryptotrader.risk.models import BookRiskRequest
from cryptotrader.signals.models import DataRequirements
from cryptotrader.signals.runner import ComponentRunError

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

    async def run(self, request: CycleRequest) -> CycleOutcome:
        snapshot = self.snapshot
        if snapshot.setup_required:
            raise RuntimeError("runtime configuration is not active")
        document = snapshot.document
        profile = document.signals.to_profile(snapshot.revision)
        books = tuple(book for book in document.execution.books if book.enabled)
        cycle_id = str(uuid4())
        created_at = self._now()

        context: SignalContext | None = None
        signals: tuple[ComponentSignal, ...] = ()
        fused: FusedSignal | None = None
        target: TargetPosition | None = None
        created_approval_ids: list[str] = []
        initial_record_saved = False
        journal_write_attempted = False
        durable_record: MultiVenueCycleRecord | None = None
        cycle_scope = getattr(self.events, "cycle", None)
        event_scope = cycle_scope(cycle_id, snapshot.revision) if callable(cycle_scope) else nullcontext()
        event_scope.__enter__()
        try:
            await self._publish(
                "cycle_started",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                pair=request.pair.canonical(),
            )
            components = self.registry.enabled(profile)
            requirements = DataRequirements.merge(
                *(component.requirements() for component in components),
                self.exit_requirement,
            )
            context = await self.market_source.collect(request.pair, created_at, requirements)
            self._validate_context(context, request.pair, created_at)
            await self._publish(
                "context_ready",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                market_data_source_id=context.market_data_source_id,
            )
            signals = await self.runner.run(components, context)
            fused = self.fusion.fuse(signals, profile.components)
            await self._publish(
                "fusion_completed",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                score=fused.score,
            )
            target = self.decisions.target_for(fused, profile)
            plan = self.exits.build_plan(context, target, signals, fused, profile)
            await self._publish(
                "decision_created",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                side=target.side,
                size_ratio=target.size_ratio,
            )
            prepared = await asyncio.gather(
                *(
                    self._prepare_book(
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
            book_results = await self._execute_ready_books(
                prepared,
                request,
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
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
            )
            journal_write_attempted = True
            await self.journal.save(record)
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
        except ComponentRunError:
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
                    (),
                    None,
                    None,
                    created_at,
                    "component_failed",
                )
            await self._publish(
                "cycle_failed",
                cycle_id=cycle_id,
                config_revision=snapshot.revision,
                status="component_failed",
            )
            return self._outcome(record)
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
        finally:
            event_scope.__exit__(None, None, None)

    async def execute_approved(self, approval_id: str) -> CycleOutcome:
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
        proposal = await self.approvals.claim_for_execution(
            approval_id,
            current_revision=snapshot.revision,
        )
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
            portfolio = await self.portfolios.read(book, self.sessions, request.pair)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _PreparedBook(book, self._preparation_failure(book, request, config_revision, "portfolio"))

        try:
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
                portfolio.total_equity,
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

    async def _execute_ready_books(
        self,
        prepared: tuple[_PreparedBook, ...] | list[_PreparedBook],
        request: CycleRequest,
        *,
        cycle_id: str,
        config_revision: int,
    ) -> tuple[BookCycleResult, ...]:
        tasks: dict[int, asyncio.Task[BookCycleResult]] = {}
        for index, item in enumerate(prepared):
            if item.result.status == "ready":
                tasks[index] = asyncio.create_task(
                    self._execute_book(
                        item,
                        request,
                        cycle_id=cycle_id,
                        config_revision=config_revision,
                    )
                )
        if tasks:
            values = await asyncio.gather(*tasks.values())
            completed = dict(zip(tasks, values, strict=True))
        else:
            completed = {}
        return tuple(completed.get(index, item.result) for index, item in enumerate(prepared))

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
        execution = await self.coordinator.execute(proposal)
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
        try:
            portfolio_after = await self.portfolios.read(prepared.book, self.sessions, request.pair)
            portfolio_after_available = True
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
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
        )
        await self._publish(
            "book_execution_completed",
            cycle_id=cycle_id,
            book_id=proposal.book_id,
            config_revision=config_revision,
            status=execution.status,
            requires_attention=execution.requires_attention,
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
    ) -> MultiVenueCycleRecord:
        status = TradingCycle._cycle_status(book_results)
        execution_status = TradingCycle._execution_status(book_results)
        attention = any(item.execution is not None and item.execution.requires_attention for item in book_results)
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
        if snapshot.setup_required or snapshot.revision != proposal.config_revision:
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
            requires_attention=any(item.execution is not None and item.execution.requires_attention for item in books),
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
