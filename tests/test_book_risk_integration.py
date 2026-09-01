"""Whole-account risk through the assembled cycle, durable journal and HTTP read model."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D  # noqa: N817 - compact exact-value acceptance fixtures
from types import SimpleNamespace

import pytest

from cryptotrader.accounts.models import AccountPosition, Instrument, Money
from cryptotrader.accounts.store import AccountStore
from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.pair import Pair
from cryptotrader.runtime import _assemble_cycle
from tests.fakes.account_session import snapshot
from tests.test_execution_service import _VenueSession
from tests.test_multi_book_cycle import _MarketSource, _Registry, _Repository, _Runner, _snapshot

PAIR = Pair.parse("BTC/USDT:USDT")
OTHER = Pair.parse("ETH/USDT:USDT")


@pytest.fixture(autouse=True)
def strict_redis(monkeypatch):
    from tests.test_book_execution_ownership import StrictRedis

    redis = StrictRedis()
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    return redis


class RiskSession(_VenueSession):
    def __init__(self):
        super().__init__("0")
        self.equity = D("100")
        self.other = D("0")
        self.incomplete = ()
        self.tick = 0

    async def list_instruments(self):
        return tuple(Instrument(pair.canonical(), pair, pair.market_type, True) for pair in (PAIR, OTHER))

    async def fetch_account(self):
        self.tick += 1
        positions = tuple(
            AccountPosition(
                Instrument(pair.canonical(), pair, pair.market_type, True),
                amount,
                abs(amount),
                Money(amount * (self.quote.last if pair == PAIR else D("100")), "USDT"),
                None,
                Money(D("0"), "USDT"),
            )
            for pair, amount in ((PAIR, self.signed_amount), (OTHER, self.other / D("100")))
        )
        value = snapshot("paper-a", amount=str(self.equity or 100))
        return replace(
            value,
            observed_at=value.observed_at + timedelta(seconds=self.tick),
            positions=positions,
            equity=Money(self.equity, "USDT", "equity unavailable" if self.equity is None else None),
            completeness=self.incomplete,
        )


async def assembled(tmp_path, *, hitl=False):
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    book = ExecutionBook("simulation", "模拟池", "simulated", True, hitl, (ConnectionAllocation("paper-a", True, 1.0),))
    initial = _snapshot(book)
    risk = initial.document.risk
    risk = risk.model_copy(
        update={
            "position": risk.position.model_copy(
                update={"max_total_exposure_pct": 0.8, "max_single_pct": 1, "max_margin_used_pct": 1}
            ),
            "loss": risk.loss.model_copy(update={"max_drawdown_pct": 0.1}),
        }
    )
    initial = replace(
        initial,
        document=initial.document.model_copy(
            update={
                "risk": risk,
                "infrastructure": initial.document.infrastructure.model_copy(update={"redis_url": "redis://offline"}),
            }
        ),
    )
    repo = _Repository(initial)
    repo.database_url = f"sqlite+aiosqlite:///{tmp_path / 'risk.db'}"
    await migrate_workbench_schema(repo.database_url)
    repo.account_store = AccountStore(repo.database_url)
    session = RiskSession()
    cycle = _assemble_cycle(
        initial,
        repo,
        {"paper-a": session},
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    cycle.analysis.runner = _Runner()
    return cycle, session, repo


@pytest.mark.asyncio
async def test_assembled_cycle_persists_peak_and_api_reads_same_risk(tmp_path):
    from api.routes.portfolio_books import _account_book

    cycle, session, repo = await assembled(tmp_path, hitl=True)
    first = await cycle.run(CycleRequest(PAIR))
    assert first.book("simulation").status == "awaiting_approval"
    session.equity = D("80")
    # Reassembly models restart / ordinary application; the peak must survive it.
    restarted = _assemble_cycle(
        cycle.snapshot,
        repo,
        {"paper-a": session},
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    restarted.analysis.runner = _Runner()
    outcome = await restarted.run(CycleRequest(PAIR))
    risk = outcome.book("simulation").proposal.risk
    assert risk.capped_target_exposure == D("0")
    record = await restarted.journal.get(outcome.cycle_id)
    assert record.book_results[0].proposal.risk.state.peak_equity == D("100")
    assert record.book_results[0].proposal.risk.state.equity == D("80")
    api_book = await _account_book(
        repo.account_store, cycle.snapshot.document, cycle.snapshot.document.execution.books[0]
    )
    assert api_book.risk_state.peak_equity == D("100")
    assert api_book.risk_state.observed_at == (await repo.account_store.latest("paper-a")).observed_at


@pytest.mark.asyncio
async def test_other_instrument_consumes_whole_pool_budget(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    session.other = D("70")
    cycle.snapshot = replace(
        cycle.snapshot,
        document=cycle.snapshot.document.model_copy(
            update={"signals": cycle.snapshot.document.signals.model_copy(update={"max_target_ratio": 0.4})}
        ),
    )
    outcome = await cycle.run(CycleRequest(PAIR))
    result = outcome.book("simulation")
    assert result.proposal.requested_target_exposure == D("0.4")
    assert result.proposal.risk.capped_target_exposure <= D("0.10")
    assert result.proposal.connection_plans[0].amount <= D("0.10")


def test_journal_order_codec_preserves_exact_client_identity():
    from cryptotrader.execution.codec import _order_from_payload, _order_payload
    from cryptotrader.venues.models import NormalizedOrder

    order = NormalizedOrder("v1", PAIR, "buy", "market", D("1"), D("1"), D("100"), "filled", False, "client1")
    assert _order_from_payload(_order_payload(order)) == order


@pytest.mark.asyncio
async def test_approval_refresh_rejects_new_other_position_before_claim(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    outcome = await cycle.run(CycleRequest(PAIR))
    approval_id = outcome.book("simulation").hitl.approval_id
    await cycle.approvals.approve(approval_id)
    session.other = D("70")
    completed = await cycle.execute_approved(approval_id)
    assert completed.book("simulation").hitl.status == "invalidated"
    assert (await cycle.approvals.get(approval_id)).claimed_at is None
    assert session.orders == []


@pytest.mark.asyncio
async def test_valid_approval_refresh_preserves_frozen_quantity_despite_new_time_and_quote(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    outcome = await cycle.run(CycleRequest(PAIR))
    proposal = outcome.book("simulation").proposal
    approval_id = outcome.book("simulation").hitl.approval_id
    await cycle.approvals.approve(approval_id)
    session.quote = replace(session.quote, bid=D("99"), ask=D("99"), last=D("99"))
    completed = await cycle.execute_approved(approval_id)
    assert completed.book("simulation").status == "completed", completed.book("simulation").execution
    assert session.orders[0].amount == proposal.connection_plans[0].amount


@pytest.mark.asyncio
async def test_frozen_quantity_price_rise_over_budget_invalidates_before_claim(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    outcome = await cycle.run(CycleRequest(PAIR))
    approval_id = outcome.book("simulation").hitl.approval_id
    await cycle.approvals.approve(approval_id)
    session.quote = replace(session.quote, bid=D("105"), ask=D("105"), last=D("105"))
    completed = await cycle.execute_approved(approval_id)
    assert completed.book("simulation").hitl.status == "invalidated"
    assert (await cycle.approvals.get(approval_id)).claimed_at is None
    assert not session.orders


@pytest.mark.asyncio
async def test_safe_flat_requires_actual_available_quantity(tmp_path):
    from cryptotrader.signals.models import ComponentSignal

    cycle, session, _ = await assembled(tmp_path)
    session.signed_amount = D("0.3")
    session.equity = None
    original = session.fetch_account

    async def locked_quantity():
        account = await original()
        return replace(account, positions=tuple(replace(p, available_amount=D("0")) for p in account.positions))

    session.fetch_account = locked_quantity

    class Flat:
        async def run(self, *_):
            return (ComponentSignal("fixture", "neutral", 1.0, "flat"),)

    cycle.analysis.runner = Flat()
    outcome = await cycle.run(CycleRequest(PAIR))
    assert outcome.book("simulation").proposal.requested_target_exposure == 0
    assert not session.orders


@pytest.mark.asyncio
async def test_unknown_equity_can_flatten_known_position_but_never_open(tmp_path):
    from cryptotrader.signals.models import ComponentSignal

    cycle, session, _ = await assembled(tmp_path, hitl=True)
    session.equity = None
    opening = await cycle.run(CycleRequest(PAIR))
    assert opening.book("simulation").proposal is not None
    assert not opening.book("simulation").proposal.risk.passed
    session.signed_amount = D("0.3")

    class Flat:
        async def run(self, *_):
            return (ComponentSignal("fixture", "neutral", 1.0, "flat"),)

    cycle.analysis.runner = Flat()
    awaiting = await cycle.run(CycleRequest(PAIR))
    approval_id = awaiting.book("simulation").hitl.approval_id
    assert approval_id is not None
    await cycle.approvals.approve(approval_id)
    completed = await cycle.execute_approved(approval_id)
    assert completed.book("simulation").status == "completed"
    assert session.orders[0].amount == D("0.3")
    assert session.orders[0].reduce_only


@pytest.mark.asyncio
async def test_normal_paper_account_can_open_and_persist_terminal_journal(tmp_path):
    from cryptotrader.venues.paper import PaperVenueAdapter

    cycle, _, repo = await assembled(tmp_path)
    adapter = PaperVenueAdapter()
    adapter.account_store = repo.account_store
    config = replace(cycle.snapshot.document.execution.connections[0], parameters={"initial_equity": "100"})
    paper = await adapter.connect(config, None)
    await paper.set_quote(PAIR, D("100"))
    assembled_cycle = _assemble_cycle(
        cycle.snapshot,
        repo,
        {"paper-a": paper},
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    assembled_cycle.analysis.runner = _Runner()
    outcome = await assembled_cycle.run(CycleRequest(PAIR))
    assert outcome.book("simulation").status == "completed"
    saved = await assembled_cycle.journal.get(outcome.cycle_id)
    assert saved.book_results[0].execution.connection_results[0].orders[0].client_order_id
    assert (await repo.account_store.latest("paper-a")).positions[0].signed_amount == D("0.8")
    state = await assembled_cycle.risk_states.get("simulation")
    assert state.gross_notional == D("80")
    assert state.observed_at == (await repo.account_store.latest("paper-a")).observed_at
    assert saved.book_results[0].proposal.risk.state.gross_notional == 0


async def test_spot_approval_retains_frozen_receipt_and_final_position_in_journal(tmp_path):
    from api.routes.response_dto import cycle_out
    from cryptotrader.venues.paper import PaperVenueAdapter

    cycle, _, repo = await assembled(tmp_path, hitl=True)
    pair = Pair.parse("BTC/USDT")
    adapter = PaperVenueAdapter()
    adapter.account_store = repo.account_store
    config = replace(cycle.snapshot.document.execution.connections[0], parameters={"initial_equity": "100"})
    paper = await adapter.connect(config, None)
    await paper.set_quote(pair, D("100"))
    active = _assemble_cycle(
        cycle.snapshot,
        repo,
        {"paper-a": paper},
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    active.analysis.runner = _Runner()
    awaiting = await active.run(CycleRequest(pair))
    approval_id = awaiting.book("simulation").hitl.approval_id
    await active.approvals.approve(approval_id)
    await paper.set_quote(pair, D("99"))
    completed = await active.execute_approved(approval_id)
    assert completed.book("simulation").status == "completed"
    saved = await active.journal.get(completed.cycle_id)
    result = saved.book_results[0].execution.connection_results[0]
    assert result.quantity_frozen
    assert result.orders[0].client_order_id
    assert result.orders[0].filled_amount == D("0.8")
    assert result.final_position.position.signed_amount == D("0.8")
    assert result.final_position.position.signed_notional == D("79.2")
    assert cycle_out(saved).books[0].connections[0].execution.quantity_frozen


@pytest.mark.parametrize(("incomplete", "first_amount"), [(False, "0.3"), (True, "0.3"), (True, "0.5")])
async def test_mixed_frozen_targets_invalidate_incomplete_increasing_member_before_claim(
    tmp_path, incomplete, first_amount
):
    from cryptotrader.venues.models import VenueQuote
    from tests.fakes.book_risk import QuotedAccountSession

    book = ExecutionBook(
        "simulation",
        "pool",
        "simulated",
        True,
        True,
        (ConnectionAllocation("a", True, 0.5), ConnectionAllocation("b", True, 0.5)),
    )
    configured = _snapshot(book)
    configured = replace(
        configured,
        document=configured.document.model_copy(
            update={
                "signals": configured.document.signals.model_copy(update={"max_target_ratio": 0.4}),
                "risk": configured.document.risk.model_copy(
                    update={
                        "position": configured.document.risk.position.model_copy(
                            update={"max_total_exposure_pct": 0.8, "max_single_pct": 1, "max_margin_used_pct": 1}
                        )
                    }
                ),
                "infrastructure": configured.document.infrastructure.model_copy(
                    update={"redis_url": "redis://offline"}
                ),
            }
        ),
    )
    repo = _Repository(configured)
    repo.database_url = f"sqlite+aiosqlite:///{tmp_path / 'mixed.db'}"
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    await migrate_workbench_schema(repo.database_url)
    repo.account_store = AccountStore(repo.database_url)
    await repo.account_store.ensure_tables()
    sessions = {"a": QuotedAccountSession("a", first_amount), "b": QuotedAccountSession("b", "0.9")}
    cycle = _assemble_cycle(
        configured,
        repo,
        sessions,
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    original_exits = cycle.exits

    class UnprotectedSpotExits:
        def build_plan(self, *args):
            return replace(original_exits.build_plan(*args), stop_loss=None, take_profit=None)

    cycle.exits = UnprotectedSpotExits()
    cycle.analysis.runner = _Runner()
    pair = Pair.parse("BTC/USDT")
    awaiting = await cycle.run(CycleRequest(pair))
    proposal = awaiting.book(book.id).proposal
    assert proposal is not None, awaiting.book(book.id)
    assert proposal.ready
    assert [p.amount for p in proposal.connection_plans] == [D("0.1"), D("0.5")]
    approval_id = awaiting.book(book.id).hitl.approval_id
    await cycle.approvals.approve(approval_id)
    sessions["a"].quote = VenueQuote(pair, D("200"), D("200"), D("200"))
    sessions["a"].incomplete = ("orders:unavailable",) if incomplete else ()
    outcome = await cycle.execute_approved(approval_id)
    approval = await cycle.approvals.get(approval_id)
    if incomplete and first_amount == "0.3":
        assert outcome.book(book.id).hitl.status == "invalidated"
        assert approval.claimed_at is None
        assert not sessions["a"].orders
        assert not sessions["b"].orders
    else:
        assert outcome.book(book.id).status == "completed"
        assert [sessions[key].orders[0].amount for key in ("a", "b")] == [D("0.1"), D("0.5")]
        assert [sessions[key].signed_amount for key in ("a", "b")] == [D("0.4"), D("0.4")]
        saved = await cycle.journal.get(outcome.cycle_id)
        assert saved.book_results[0].proposal == proposal


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_store", ["risk", "account"])
async def test_post_execution_snapshot_failure_preserves_fill_protection_and_marks_reconciliation(
    tmp_path, failed_store
):
    from api.routes.response_dto import cycle_out

    cycle, session, repo = await assembled(tmp_path)
    store = cycle.risk_states if failed_store == "risk" else repo.account_store
    method = "update" if failed_store == "risk" else "ingest"
    original = getattr(store, method)
    count = 0

    async def fail_after(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected post-execution persistence failure")
        return await original(*args, **kwargs)

    setattr(store, method, fail_after)
    outcome = await cycle.run(CycleRequest(PAIR))
    assert outcome.requires_attention
    saved = await cycle.journal.get(outcome.cycle_id)
    book = saved.book_results[0]
    assert book.reconciliation_required
    assert book.portfolio_after_available is (failed_store == "risk")
    assert book.execution.connection_results[0].status == "completed"
    assert book.execution.connection_results[0].orders[0].filled_amount == D("0.8")
    assert book.execution.connection_results[0].protection.active
    assert len(session.orders) == 1
    assert cycle_out(saved).books[0].reconciliation_required


@pytest.mark.asyncio
async def test_pending_increase_reserves_budget_and_pending_reduce_does_not_release_it(tmp_path):
    from cryptotrader.accounts.models import AccountOrder

    cycle, session, _ = await assembled(tmp_path, hitl=True)
    session.other = D("50")
    fetch = session.fetch_account

    async def account():
        s = await fetch()
        orders = tuple(
            AccountOrder(
                "paper-a",
                str(i),
                Instrument(OTHER.canonical(), OTHER, "swap", True),
                side,
                "limit",
                D("0.2"),
                D("0"),
                None,
                "open",
                reducing,
                False,
                None,
                s.observed_at,
                Money(D("20"), "USDT"),
            )
            for i, (side, reducing) in enumerate((("buy", False), ("sell", True)))
        )
        return replace(s, orders=orders)

    session.fetch_account = account
    outcome = await cycle.run(CycleRequest(PAIR))
    risk = outcome.book("simulation").proposal.risk
    assert risk.state.pending_increase_notional == D("20")
    assert risk.capped_target_exposure == D("0.1")


@pytest.mark.asyncio
async def test_approval_original_amount_exceeding_current_margin_is_invalidated(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    outcome = await cycle.run(CycleRequest(PAIR))
    approval_id = outcome.book("simulation").hitl.approval_id
    await cycle.approvals.approve(approval_id)
    fetch = session.fetch_account

    async def account():
        return replace(await fetch(), available_margin=Money(D("10"), "USDT"))

    session.fetch_account = account
    completed = await cycle.execute_approved(approval_id)
    assert completed.book("simulation").hitl.status == "invalidated"
    assert session.orders == []


@pytest.mark.asyncio
async def test_known_partial_reduction_survives_unrelated_missing_account_facts(tmp_path):
    cycle, session, _ = await assembled(tmp_path, hitl=True)
    session.signed_amount = D("0.8")
    session.incomplete = ("orders:unavailable",)
    cycle.snapshot = replace(
        cycle.snapshot,
        document=cycle.snapshot.document.model_copy(
            update={"signals": cycle.snapshot.document.signals.model_copy(update={"max_target_ratio": 0.4})}
        ),
    )
    outcome = await cycle.run(CycleRequest(PAIR))
    assert outcome.book("simulation").proposal.ready
    assert outcome.book("simulation").proposal.connection_plans[0].reduce_only


@pytest.mark.asyncio
async def test_two_pairs_in_one_pool_wait_for_new_account_facts_while_outside_pool_runs(tmp_path, strict_redis):
    import asyncio
    from unittest.mock import AsyncMock

    from cryptotrader.cycle_events import MultiplexedCycleEventSink
    from cryptotrader.runtime import Runtime
    from cryptotrader.venues.paper import PaperVenueAdapter

    cycle, _, repo = await assembled(tmp_path)
    paper = await PaperVenueAdapter().connect(
        replace(cycle.snapshot.document.execution.connections[0], parameters={"initial_equity": "100"}), None
    )
    for pair in (PAIR, OTHER):
        await paper.set_quote(pair, D("100"))
    cycle = _assemble_cycle(
        cycle.snapshot,
        repo,
        {"paper-a": paper},
        _Registry(),
        SimpleNamespace(require=lambda _: _MarketSource()),
        NullCycleEventSink(),
    )
    cycle.analysis.runner = _Runner()
    entered, release = asyncio.Event(), asyncio.Event()
    original_place = paper.place_order

    async def place(intent):
        entered.set()
        await release.wait()
        return await original_place(intent)

    paper.place_order = place
    runtime = Runtime(
        snapshot=cycle.snapshot,
        repository=repo,
        cycle=cycle,
        sessions={"paper-a": paper},
        signal_registry=_Registry(),
        market_registry=object(),
        venue_registry=SimpleNamespace(bind_account_store=lambda _: None),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    runtime._reload_for_cycle_locked = AsyncMock(return_value=cycle)

    async def run(pair):
        async with runtime.execution_lease(pair.canonical()) as admitted:
            return await admitted.run(CycleRequest(pair))

    first = asyncio.create_task(run(PAIR))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(run(OTHER))
        await asyncio.wait_for(strict_redis.contended.wait(), 2)
        assert not second.done()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside, session, outside_repo = await assembled(outside_dir)
        external_book = replace(outside.snapshot.document.execution.books[0], id="outside")
        outside_snapshot = replace(
            outside.snapshot,
            document=outside.snapshot.document.model_copy(
                update={"execution": outside.snapshot.document.execution.model_copy(update={"books": (external_book,)})}
            ),
        )
        external = _assemble_cycle(
            outside_snapshot,
            outside_repo,
            {"paper-a": session},
            _Registry(),
            SimpleNamespace(require=lambda _: _MarketSource()),
            NullCycleEventSink(),
        )
        external.analysis.runner = _Runner()
        assert (await external.run(CycleRequest(PAIR))).book("outside").status == "completed"
        assert not first.done()
        assert not second.done()
        release.set()
        await asyncio.wait_for(first, 2)
        next_result = await asyncio.wait_for(second, 2)
        risk = next_result.book("simulation").proposal.risk
        assert risk.state.positions_by_instrument[PAIR.canonical()] == D("80")
        assert risk.capped_target_exposure == D("0")
        assert len((await paper.fetch_fills(None)).items) == 1
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        await runtime.close()
