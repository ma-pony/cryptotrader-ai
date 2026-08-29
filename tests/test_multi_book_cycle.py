"""一次市场观点驱动多个隔离执行资金池的主链契约。"""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime
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
        repository=repository or _Repository(snapshot),
        market_source=_MarketSource(),
        registry=_Registry(),
        runner=runner,
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        sessions={},
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
async def test_unexpected_book_execution_failure_does_not_block_sibling():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    live = _book("live", "real", ("live-first", "live-second"), hitl=False)
    cycle, _, coordinator, _, _ = _cycle(
        _snapshot(simulation, live),
        execution_failed_books={"simulation"},
    )

    outcome = await cycle.run(CycleRequest(PAIR))

    assert outcome.book("simulation").status == "failed"
    assert outcome.book("simulation").execution.requires_attention is True
    assert outcome.book("live").status == "completed"
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
async def test_cycle_keeps_initial_revision_when_repository_changes_mid_run():
    simulation = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    first = _snapshot(simulation, revision=9)
    repository = _Repository(first, replace(first, revision=10))
    cycle, _, _, _, _ = _cycle(first, repository=repository)

    outcome = await cycle.run(CycleRequest(PAIR))

    assert repository.calls == 1
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
