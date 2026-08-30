"""一次市场观点驱动多个隔离执行资金池的主链契约。"""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.hitl.store import BookApprovalStore
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SignalComponentConfig, SignalConfig, SystemConfig
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import ComponentSignal, DataRequirements, SignalContext
from cryptotrader.trading_cycle import TradingCycle
from tests.factories.runtime_config import connection, runtime_document
from tests.test_multi_venue_journal import _execution, _portfolio, _proposal_for

PAIR = Pair.parse("BTC/USDT:USDT")
NOW = datetime.now(UTC)


def _book(book_id: str, capital_scope: str, connection_ids: tuple[str, str], *, hitl: bool) -> ExecutionBook:
    return ExecutionBook(
        book_id,
        book_id,
        capital_scope,
        True,
        hitl,
        tuple(ConnectionAllocation(connection_id, True, 0.5) for connection_id in connection_ids),
    )


def _snapshot(*books: ExecutionBook, revision: int = 9) -> RuntimeConfigSnapshot:
    connection_ids = tuple(allocation.connection_id for book in books for allocation in book.allocations)
    connections = tuple(
        connection(
            connection_id,
            "live" if connection_id.startswith("live-") else "paper",
            adapter_id="okx" if connection_id.startswith("live-") else "paper",
            credential_ref="credential-ref" if connection_id.startswith("live-") else None,
        )
        for connection_id in connection_ids
    )
    document = runtime_document(
        connections=connections,
        books=books,
        system=SystemConfig(active=True),
        signals=SignalConfig(
            components=(SignalComponentConfig(component_id="fixture", enabled=True, weight=1.0),),
            neutral_threshold=0.2,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
        ),
    )
    return RuntimeConfigSnapshot(revision, document, NOW)


class _Repository:
    def __init__(self, *snapshots: RuntimeConfigSnapshot) -> None:
        self.snapshots = snapshots
        self.calls = 0

    async def get_or_create(self) -> RuntimeConfigSnapshot:
        snapshot = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        return snapshot


class _MutableRepository:
    def __init__(self, snapshot: RuntimeConfigSnapshot) -> None:
        self.current = snapshot
        self.calls = 0

    async def get_or_create(self) -> RuntimeConfigSnapshot:
        self.calls += 1
        return self.current


class _MarketSource:
    id = "default"

    def __init__(self) -> None:
        self.calls = 0

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        self.calls += 1
        return SignalContext(
            pair=pair,
            as_of=as_of,
            market_data_source_id=self.id,
            market_type=pair.market_type,
            current_price=100.0,
            atr=5.0,
            snapshots={},
        )


class _Registry:
    def enabled(self, profile):
        class _Component:
            @staticmethod
            def requirements():
                return DataRequirements()

        return (_Component(),)


class _Runner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, components, context):
        self.calls += 1
        return (ComponentSignal("fixture", "long", 1.0, "fixture"),)


class _Aggregator:
    def __init__(self, proposals, *, failed_books=()) -> None:
        self.proposals = proposals
        self.failed_books = set(failed_books)
        self.calls: dict[str, int] = {}

    async def read(self, book, sessions, pair):
        count = self.calls.get(book.id, 0)
        self.calls[book.id] = count + 1
        if book.id in self.failed_books:
            raise RuntimeError("raw venue secret must not escape")
        proposal = self.proposals[book.id]
        if count == 0:
            return _portfolio(proposal)
        execution = _execution(proposal)
        return _portfolio(proposal, after=True, execution=execution)


class _Planner:
    def __init__(self, proposals) -> None:
        self.proposals = proposals
        self.calls = 0

    async def propose(self, request, sessions, *, pair, stop_loss, take_profit, config_revision):
        self.calls += 1
        proposal = self.proposals[request.book.id]
        assert request.target_exposure == 1
        assert request.peak_equity == request.portfolio.total_equity
        assert config_revision == proposal.config_revision
        return proposal


class _AllocationPolicy:
    def allocate(self, target, book, portfolio):
        assert target.signed_ratio in {-1.0, 0.0, 1.0}
        assert portfolio.book_id == book.id
        return ()


class _Coordinator:
    def __init__(self, *, failed_books=()) -> None:
        self.proposals = []
        self.failed_books = set(failed_books)

    async def execute(self, proposal):
        self.proposals.append(proposal)
        if proposal.book_id in self.failed_books:
            raise RuntimeError("raw venue failure must remain book-local")
        return _execution(proposal)


class _ConcurrentCoordinator:
    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.proposals = []
        self._all_entered = asyncio.Event()

    async def execute(self, proposal):
        self.proposals.append(proposal)
        if len(self.proposals) == self.expected_calls:
            self._all_entered.set()
        await self._all_entered.wait()
        return _execution(proposal)


class _FailingEventSink:
    async def publish(self, event) -> None:
        raise RuntimeError(f"observer failed at {event.name}")


def _cycle(snapshot, *, failed_books=(), execution_failed_books=(), repository=None):
    proposals = {
        book.id: _proposal_for(
            book.id,
            book.capital_scope,
            tuple(allocation.connection_id for allocation in book.allocations),
            PAIR,
        )
        for book in snapshot.document.execution.books
    }
    runner = _Runner()
    coordinator = _Coordinator(failed_books=execution_failed_books)
    journal = MultiVenueCycleStore()
    approvals = BookApprovalStore()
    cycle = TradingCycle(
        snapshot=snapshot,
        repository=repository or _Repository(snapshot),
        market_source=_MarketSource(),
        registry=_Registry(),
        runner=runner,
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        sessions={
            allocation.connection_id: object()
            for book in snapshot.document.execution.books
            for allocation in book.allocations
        },
        portfolios=_Aggregator(proposals, failed_books=failed_books),
        allocation_policy=_AllocationPolicy(),
        book_risk=object(),
        connection_risk=object(),
        planner=_Planner(proposals),
        approvals=approvals,
        coordinator=coordinator,
        journal=journal,
        events=NullCycleEventSink(),
        clock=lambda: NOW,
    )
    return cycle, runner, coordinator, journal, approvals


@pytest.mark.asyncio
async def test_one_signal_pass_drives_simulation_and_live_books():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=False)
    cycle, runner, coordinator, _, _ = _cycle(_snapshot(simulation, live))

    outcome = await cycle.run(CycleRequest(PAIR))

    assert runner.calls == 1
    assert len(coordinator.proposals) == 2
    assert outcome.target_position.side == "long"
    assert {result.book_id for result in outcome.books} == {"simulation", "live"}


@pytest.mark.asyncio
async def test_cycle_started_base_observer_failure_is_journaled_before_propagation():
    from cryptotrader.cycle_events import MultiplexedCycleEventSink

    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    cycle, _, _, journal, _ = _cycle(_snapshot(simulation))
    cycle.events = MultiplexedCycleEventSink(_FailingEventSink())

    with pytest.raises(RuntimeError, match="observer failed at cycle_failed"):
        await cycle.run(CycleRequest(PAIR))

    assert len(journal.records) == 1
    assert journal.records[0].cycle_status == "cycle_failed"


@pytest.mark.asyncio
async def test_book_failure_does_not_block_sibling():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=False)
    cycle, _, coordinator, _, _ = _cycle(_snapshot(simulation, live), failed_books={"simulation"})

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.book("simulation").status == "failed"
    assert outcome.book("simulation").failure.stage == "portfolio"
    assert outcome.book("live").status == "completed"
    assert [proposal.book_id for proposal in coordinator.proposals] == ["live"]
    assert outcome.execution_status == "partial"


@pytest.mark.asyncio
async def test_coordinator_contract_violation_does_not_fabricate_connection_results():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=False)
    cycle, _, coordinator, _, _ = _cycle(
        _snapshot(simulation, live),
        execution_failed_books={"simulation"},
    )

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.status == "cycle_failed"
    assert outcome.books == ()
    assert {proposal.book_id for proposal in coordinator.proposals} == {"simulation", "live"}


@pytest.mark.asyncio
async def test_completed_simulation_is_preserved_while_live_waits_for_approval():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, coordinator, _, _ = _cycle(_snapshot(simulation, live))

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.status == "awaiting_approval"
    assert outcome.book("simulation").status == "completed"
    assert outcome.book("live").status == "awaiting_approval"
    assert [proposal.book_id for proposal in coordinator.proposals] == ["simulation"]


@pytest.mark.asyncio
async def test_cycle_uses_constructor_snapshot_without_repository_reread():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    first = _snapshot(simulation, revision=9)
    repository = _Repository(first, replace(first, revision=10))
    cycle, _, _, _, _ = _cycle(first, repository=repository)

    outcome = await cycle.run(CycleRequest(PAIR))

    assert repository.calls == 0
    assert outcome.config_revision == 9
    assert all(book.config_revision == 9 for book in outcome.books)


@pytest.mark.asyncio
async def test_execute_approved_uses_original_proposal_once_and_replaces_same_cycle():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    snapshot = _snapshot(live)
    cycle, runner, coordinator, journal, approvals = _cycle(snapshot)
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_id = awaiting.book("live").hitl.approval_id
    await approvals.approve(approval_id)

    completed = await cycle.execute_approved(approval_id)

    assert completed.cycle_id == awaiting.cycle_id
    assert runner.calls == 1
    assert len(coordinator.proposals) == 1
    assert coordinator.proposals[0] is awaiting.book("live").proposal
    assert completed.book("live").status == "completed"
    assert len(journal.records) == 1
    assert journal.records[0].cycle_id == awaiting.cycle_id


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_expired_approved_cycle_never_reaches_coordinator(tmp_path, monkeypatch, database):
    import cryptotrader.hitl.store as stores

    book = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=True)
    cycle, _, coordinator, _, _ = _cycle(_snapshot(book))
    approvals = stores.BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'cycle-expiry.db'}" if database else None)
    cycle.approvals = approvals
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_id = awaiting.book("simulation").hitl.approval_id
    await approvals.approve(approval_id)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW + timedelta(minutes=61)

    monkeypatch.setattr(stores, "datetime", Later)
    outcome = await cycle.execute_approved(approval_id)
    assert outcome.book("simulation").status == "approval_rejected"
    assert coordinator.proposals == []
    assert (await approvals.get(approval_id)).claimed_at is None


@pytest.mark.asyncio
async def test_assembled_cycle_uses_snapshot_approval_ttl_and_market_candles():
    from cryptotrader.runtime import _assemble_cycle
    from cryptotrader.runtime_config.models import HitlConfig, MarketDataConfig

    snapshot = _snapshot()
    document = snapshot.document.model_copy(
        update={
            "hitl": HitlConfig(approval_ttl_minutes=5),
            "market_data": MarketDataConfig(parameters={"timeframe": "15m", "limit": 55}),
        }
    )
    snapshot = replace(snapshot, document=document)

    class Markets:
        def require(self, source_id):
            return _MarketSource()

    cycle = _assemble_cycle(snapshot, _Repository(snapshot), {}, _Registry(), Markets(), NullCycleEventSink())
    from tests.test_book_hitl_store import _book_proposal

    await cycle.approvals.create(_book_proposal(), created_at=datetime.now(UTC) - timedelta(minutes=6))
    assert await cycle.approvals.list_pending() == []
    assert cycle.exit_requirement.candles[0].timeframe == "15m"
    assert cycle.exit_requirement.candles[0].limit == 55


@pytest.mark.asyncio
async def test_execute_approved_invalidates_revision_change_without_claiming_or_execution():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    initial = _snapshot(live, revision=9)
    repository = _Repository(replace(initial, revision=10))
    cycle, _, coordinator, journal, approvals = _cycle(initial, repository=repository)
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_id = awaiting.book("live").hitl.approval_id
    await approvals.approve(approval_id)

    invalidated = await cycle.execute_approved(approval_id)

    approval = await approvals.get(approval_id)
    assert approval is not None
    assert approval.status == "invalidated"
    assert approval.claimed_at is None
    assert invalidated.book("live").hitl.status == "invalidated"
    assert invalidated.book("live").status == "approval_rejected"
    assert coordinator.proposals == []
    stored = await journal.get(awaiting.cycle_id)
    assert stored is not None
    assert stored.book_results[0] == invalidated.book("live")


@pytest.mark.asyncio
async def test_execute_approved_reads_revision_after_approval_and_journal_identity_validation():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    initial = _snapshot(live, revision=9)
    repository = _MutableRepository(initial)
    cycle, _, coordinator, journal, approvals = _cycle(initial, repository=repository)
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_id = awaiting.book("live").hitl.approval_id
    await approvals.approve(approval_id)
    original_get = journal.get

    async def get_then_change_revision(candidate_id):
        record = await original_get(candidate_id)
        repository.current = replace(initial, revision=10)
        return record

    journal.get = get_then_change_revision

    invalidated = await cycle.execute_approved(approval_id)

    assert repository.calls == 1
    assert invalidated.book("live").hitl.status == "invalidated"
    approval = await approvals.get(approval_id)
    assert approval is not None
    assert approval.status == "invalidated"
    assert coordinator.proposals == []


@pytest.mark.asyncio
async def test_cancellation_after_approval_creation_invalidates_unjournaled_approval():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, _, journal, approvals = _cycle(_snapshot(live))

    class _CancelAtAwaiting:
        async def publish(self, event):
            if event.name == "book_awaiting_approval":
                raise asyncio.CancelledError

    cycle.events = _CancelAtAwaiting()

    with pytest.raises(asyncio.CancelledError):
        await cycle.run(CycleRequest(PAIR))

    assert len(approvals.records) == 1
    assert approvals.records[0].status == "invalidated"
    assert await approvals.list_pending() == []
    assert journal.records[0].cycle_status == "cancelled"


@pytest.mark.asyncio
async def test_approval_durably_created_before_create_error_is_invalidated_when_not_journaled():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, _, _, _ = _cycle(_snapshot(live))

    class _CommitThenFailStore(BookApprovalStore):
        async def create(self, *args, **kwargs):
            await super().create(*args, **kwargs)
            raise RuntimeError("create response failed")

    approvals = _CommitThenFailStore()
    cycle.approvals = approvals

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.book("live").status == "failed"
    assert outcome.book("live").failure.stage == "planning"
    assert len(approvals.records) == 1
    assert approvals.records[0].status == "invalidated"
    assert await approvals.list_pending() == []


@pytest.mark.asyncio
async def test_successful_journal_save_is_authoritative_for_pending_approval_cleanup():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, _, _, approvals = _cycle(_snapshot(live))

    class _WriteOnlyAfterSaveJournal(MultiVenueCycleStore):
        async def get(self, cycle_id):
            raise RuntimeError("transient journal read outage")

    journal = _WriteOnlyAfterSaveJournal()
    cycle.journal = journal

    outcome = await cycle.run(CycleRequest(PAIR))

    approval_id = outcome.book("live").hitl.approval_id
    assert approval_id is not None
    assert journal.records[0].cycle_status == "awaiting_approval"
    assert (await approvals.get(approval_id)).status == "pending"


@pytest.mark.asyncio
async def test_ambiguous_journal_write_and_failed_read_does_not_invalidate_possible_durable_approval():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, _, _, approvals = _cycle(_snapshot(live))

    class _AmbiguousJournal(MultiVenueCycleStore):
        async def save(self, record):
            await super().save(record)
            raise RuntimeError("ambiguous write response")

        async def get(self, cycle_id):
            raise RuntimeError("durable read unavailable")

    journal = _AmbiguousJournal()
    cycle.journal = journal

    with pytest.raises(RuntimeError, match="durable read unavailable"):
        await cycle.run(CycleRequest(PAIR))

    assert journal.records[0].cycle_status == "awaiting_approval"
    assert len(approvals.records) == 1
    assert approvals.records[0].status == "pending"


@pytest.mark.asyncio
async def test_approval_without_journal_is_not_claimed():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, coordinator, _, approvals = _cycle(_snapshot(live))
    proposal = _proposal_for("live", "real", ("live-first", "live-second"), PAIR)
    approval = await approvals.create(proposal, cycle_id="missing-cycle")
    await approvals.approve(approval.approval_id)

    with pytest.raises(LookupError, match="cycle"):
        await cycle.execute_approved(approval.approval_id)

    assert (await approvals.get(approval.approval_id)).status == "approved"
    assert coordinator.proposals == []


@pytest.mark.asyncio
async def test_approval_mismatched_with_journal_is_not_claimed():
    live = _book("live", "real", ("live-first", "live-second"), hitl=True)
    cycle, _, coordinator, _, approvals = _cycle(_snapshot(live))
    awaiting = await cycle.run(CycleRequest(PAIR))
    proposal = awaiting.book("live").proposal
    mismatch = await approvals.create(
        proposal,
        cycle_id=awaiting.cycle_id,
        approval_id="mismatched-approval",
    )
    await approvals.approve(mismatch.approval_id)

    with pytest.raises(ValueError, match="approval"):
        await cycle.execute_approved(mismatch.approval_id)

    assert (await approvals.get(mismatch.approval_id)).status == "approved"
    assert coordinator.proposals == []


@pytest.mark.asyncio
async def test_concurrent_book_approvals_execute_once_and_merge_both_terminal_facts():
    first = _book("first-book", "simulated", ("first-a", "first-b"), hitl=True)
    second = _book("second-book", "real", ("second-a", "second-b"), hitl=True)
    cycle, _, _, journal, approvals = _cycle(_snapshot(first, second))
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_ids = tuple(book.hitl.approval_id for book in awaiting.books)
    for approval_id in approval_ids:
        await approvals.approve(approval_id)
    coordinator = _ConcurrentCoordinator(expected_calls=2)
    cycle.coordinator = coordinator

    await asyncio.gather(*(cycle.execute_approved(approval_id) for approval_id in approval_ids))

    stored = await journal.get(awaiting.cycle_id)
    assert stored is not None
    assert tuple(book.status for book in stored.book_results) == ("completed", "completed")
    assert {proposal.book_id for proposal in coordinator.proposals} == {"first-book", "second-book"}
    assert len(coordinator.proposals) == 2


def test_cycle_request_and_signal_context_are_hard_cut_over():
    assert [field.name for field in fields(CycleRequest)] == ["pair"]
    assert [field.name for field in fields(SignalContext)] == [
        "pair",
        "as_of",
        "market_data_source_id",
        "market_type",
        "current_price",
        "atr",
        "snapshots",
    ]


def test_runtime_module_is_the_single_bootstrap_surface():
    runtime_module = import_module("cryptotrader.runtime")
    assert callable(runtime_module.build_runtime)
