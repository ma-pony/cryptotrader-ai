"""Book proposal planning preflights normalized venue sessions without brand branches."""

from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_DOWN, Decimal

import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import (
    ConnectionPosition,
    OpenVenueState,
    ProtectionState,
    VenueCapabilities,
    VenueQuote,
)
from cryptotrader.venues.protocol import VenueOperationError

PAIR = Pair.parse("BTC/USDT:USDT")
SPOT_PAIR = Pair.parse("BTC/USDT")
CAPABILITIES = VenueCapabilities(frozenset({"spot", "swap"}), True, False, True, frozenset({"market"}))


@pytest.mark.parametrize(
    ("incomplete", "available_margin", "second_amount", "second_price", "valid"),
    [
        (False, "100", "0.9", "100", True),
        (True, "100", "0.9", "100", False),
        (False, "10", "0.9", "100", False),
        (False, "100", "0.4", "100", True),
        (False, "100", "0.4", "300", False),
    ],
)
async def test_frozen_member_prices_use_actual_mixed_targets_for_account_completeness(
    incomplete, available_margin, second_amount, second_price, valid
):
    from cryptotrader.execution.planner import ExecutionPlanner
    from cryptotrader.portfolio.aggregator import PortfolioAggregator
    from cryptotrader.risk.book_state import BookRiskState
    from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
    from cryptotrader.risk.models import BookRiskLimits, BookRiskRequest, ConnectionRiskLimits
    from tests.fakes.book_risk import QuotedAccountSession

    book = ExecutionBook(
        "simulation",
        "pool",
        "simulated",
        True,
        True,
        (ConnectionAllocation("a", True, 0.5), ConnectionAllocation("b", True, 0.5)),
    )
    sessions = {"a": QuotedAccountSession("a", "0.3"), "b": QuotedAccountSession("b", second_amount)}
    aggregator = PortfolioAggregator()
    before = await aggregator.read(book, sessions, SPOT_PAIR)
    state = BookRiskState.from_snapshots(book.id, tuple(p.account_snapshot for p in before.connections))
    planner = ExecutionPlanner(
        book_risk_gate=BookRiskGate(BookRiskLimits(Decimal("1"), Decimal("0.8"), Decimal("0.1"), Decimal("1"))),
        connection_risk_gate=ConnectionRiskGate(ConnectionRiskLimits(Decimal("1"))),
    )
    proposal = await planner.propose(
        BookRiskRequest(book, before, Decimal("0.4"), SPOT_PAIR, state),
        sessions,
        pair=SPOT_PAIR,
        stop_loss=None,
        take_profit=None,
        config_revision=1,
    )
    assert proposal.ready
    expected = [Decimal("0.1"), Decimal("0.5")] if second_amount == "0.9" else [Decimal("0.1")]
    assert [p.amount for p in proposal.connection_plans] == expected
    sessions["a"].quote = VenueQuote(SPOT_PAIR, Decimal("200"), Decimal("200"), Decimal("200"))
    sessions["a"].incomplete = ("orders:unavailable",) if incomplete else ()
    sessions["a"].available_margin = Decimal(available_margin)
    sessions["b"].quote = VenueQuote(SPOT_PAIR, Decimal(second_price), Decimal(second_price), Decimal(second_price))
    fresh = await aggregator.read(book, sessions, SPOT_PAIR)
    fresh_state = BookRiskState.from_snapshots(book.id, tuple(p.account_snapshot for p in fresh.connections))
    assert [p.position.signed_notional for p in fresh.connections] == [
        Decimal("60"),
        Decimal(second_amount) * Decimal(second_price),
    ]
    assert await planner.validate_frozen(book, proposal, fresh, fresh_state, sessions) is valid


def _book() -> ExecutionBook:
    return ExecutionBook(
        "simulation",
        "Simulation",
        "simulated",
        True,
        False,
        (
            ConnectionAllocation("reachable", True, 0.4),
            ConnectionAllocation("unreachable", True, 0.6),
        ),
    )


def _portfolio(
    *,
    current: tuple[str, str] = ("0", "0"),
    amounts: tuple[str, str] | None = None,
    pair: Pair = PAIR,
) -> BookPortfolioSnapshot:
    amounts = amounts or tuple(str(Decimal(notional) / Decimal("100")) for notional in current)
    snapshots = tuple(
        ConnectionPortfolioSnapshot(
            connection_id,
            Decimal("5000"),
            {"USDT": Decimal("5000")},
            ConnectionPosition(pair, Decimal(amount), Decimal(notional), Decimal("100")),
        )
        for connection_id, amount, notional in zip(
            ("reachable", "unreachable"),
            amounts,
            current,
            strict=True,
        )
    )
    return BookPortfolioSnapshot(
        "simulation",
        "simulated",
        Decimal("10000"),
        sum((Decimal(item) for item in current), Decimal("0")),
        snapshots,
    )


def _request(
    *,
    target: str = "0.5",
    current: tuple[str, str] = ("0", "0"),
    amounts: tuple[str, str] | None = None,
    pair: Pair = PAIR,
):
    from cryptotrader.risk.models import BookRiskRequest
    from tests.fakes.book_risk import state_for, with_accounts

    portfolio = with_accounts(_portfolio(current=current, amounts=amounts, pair=pair))
    return BookRiskRequest(
        _book(),
        portfolio,
        Decimal(target),
        pair,
        state_for(portfolio, Decimal("10000")),
    )


class _Session:
    def __init__(
        self,
        snapshot: ConnectionPortfolioSnapshot,
        *,
        unavailable: bool = False,
        normalize_error: bool = False,
        normalize_invariant_error: bool = False,
        state_error: bool = False,
        unsafe_normalization: bool = False,
        protection_ids: tuple[str, ...] = (),
    ) -> None:
        self.connection_id = snapshot.connection_id
        self.capabilities = CAPABILITIES
        self._snapshot = snapshot
        self._unavailable = unavailable
        self._normalize_error = normalize_error
        self._normalize_invariant_error = normalize_invariant_error
        self._state_error = state_error
        self._unsafe_normalization = unsafe_normalization
        self.normalize_calls: list[tuple[Pair, Decimal]] = []
        protections = ()
        if protection_ids:
            protections = (
                ProtectionState(
                    protection_ids,
                    snapshot.position.pair,
                    "long",
                    Decimal("1"),
                    Decimal("90"),
                    Decimal("120"),
                    True,
                    False,
                ),
            )
        self._state = OpenVenueState(snapshot.position, (), protections)

    async def fetch_quote(self, pair):
        if self._unavailable:
            raise VenueOperationError("RAW_SECRET_MARKER venue response")
        return VenueQuote(pair, Decimal("99"), Decimal("101"), Decimal("100"))

    async def list_open_state(self, pair):
        if self._unavailable or self._state_error:
            raise VenueOperationError("RAW_SECRET_MARKER venue response")
        return self._state

    async def normalize_amount(self, pair, base_amount):
        self.normalize_calls.append((pair, base_amount))
        if self._normalize_error:
            raise VenueOperationError("RAW_SECRET_MARKER precision response")
        if self._normalize_invariant_error:
            raise ValueError("normalize contract violation")
        if self._unsafe_normalization:
            return base_amount + Decimal("1")
        return base_amount.quantize(Decimal("0.001"), rounding=ROUND_DOWN)


def _sessions(
    portfolio: BookPortfolioSnapshot,
    *,
    unavailable: str | None = None,
    normalize_error: str | None = None,
    normalize_invariant_error: str | None = None,
    state_error: str | None = None,
    unsafe_normalization: str | None = None,
    protection_ids: tuple[str, ...] = (),
):
    result = {
        snapshot.connection_id: _Session(
            snapshot,
            unavailable=snapshot.connection_id == unavailable,
            normalize_error=snapshot.connection_id == normalize_error,
            normalize_invariant_error=snapshot.connection_id == normalize_invariant_error,
            state_error=snapshot.connection_id == state_error,
            unsafe_normalization=snapshot.connection_id == unsafe_normalization,
            protection_ids=protection_ids if snapshot.connection_id == "reachable" else (),
        )
        for snapshot in portfolio.connections
    }
    return {"unreachable": result["unreachable"], "reachable": result["reachable"]}


def _planner():
    from cryptotrader.execution.planner import ExecutionPlanner
    from cryptotrader.risk.gate import BookRiskGate, ConnectionRiskGate
    from cryptotrader.risk.models import BookRiskLimits, ConnectionRiskLimits

    return ExecutionPlanner(
        book_risk_gate=BookRiskGate(BookRiskLimits(Decimal("1"), Decimal("1"), Decimal("0.2"), Decimal("1"))),
        connection_risk_gate=ConnectionRiskGate(ConnectionRiskLimits(Decimal("1"))),
    )


async def _propose(
    request,
    sessions,
    *,
    pair: Pair | None = None,
    stop_loss: Decimal | None = Decimal("90"),
    take_profit: Decimal | None = Decimal("120"),
):
    return await _planner().propose(
        request,
        sessions,
        pair=pair or request.portfolio.connections[0].position.pair,
        stop_loss=stop_loss,
        take_profit=take_profit,
        config_revision=7,
    )


@pytest.mark.asyncio
async def test_increase_requires_every_connection_to_pass_before_any_plan_is_ready():
    request = _request()
    sessions = _sessions(request.portfolio, unavailable="unreachable")

    proposal = await _propose(request, sessions)

    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert proposal.risk.rejected_by == "connection_preflight"
    assert proposal.unavailable_connections == ("unreachable",)
    assert "RAW_SECRET_MARKER" not in repr(proposal)
    assert sessions["reachable"].normalize_calls


@pytest.mark.asyncio
async def test_risk_reduction_keeps_reachable_connection_plans():
    request = _request(target="0", current=("2000", "3000"), amounts=("19.75", "30.25"))
    sessions = _sessions(request.portfolio, unavailable="unreachable")

    proposal = await _propose(request, sessions)

    assert proposal.ready is True
    assert tuple(plan.connection_id for plan in proposal.connection_plans) == ("reachable",)
    assert proposal.connection_plans[0].reduce_only is True
    assert proposal.connection_plans[0].amount == Decimal("19.750")
    assert proposal.connection_plans[0].post_fill_signed_amount == Decimal("0.000")
    assert proposal.unavailable_connections == ("unreachable",)


@pytest.mark.asyncio
async def test_planner_preserves_allocation_order_and_builds_exact_normalized_plan_fields():
    request = _request(target="0.5", current=("1000", "1000"), amounts=("10", "10"))
    sessions = _sessions(request.portfolio, protection_ids=("old-a", "old-b"))

    proposal = await _propose(request, sessions)

    assert proposal.ready is True
    assert proposal.config_revision == 7
    assert tuple(plan.connection_id for plan in proposal.connection_plans) == ("reachable", "unreachable")
    plan = proposal.connection_plans[0]
    assert plan.current_signed_notional == Decimal("1000")
    assert plan.target_signed_notional == Decimal("2000.00")
    assert plan.delta_signed_notional == Decimal("1000.00")
    assert plan.side == "buy"
    assert plan.execution_price == Decimal("101")
    assert plan.current_signed_amount == Decimal("10")
    assert plan.target_signed_amount == Decimal("2000.00") / Decimal("101")
    assert plan.delta_signed_amount == Decimal("2000.00") / Decimal("101") - Decimal("10")
    assert plan.amount == Decimal("9.801")
    assert plan.post_fill_signed_amount == Decimal("19.801")
    assert plan.reduce_only is False
    assert plan.market_type == "swap"
    assert plan.stop_loss == Decimal("90")
    assert plan.take_profit == Decimal("120")
    assert plan.old_protection_ids == ("old-a", "old-b")
    assert plan.capabilities == CAPABILITIES
    assert sessions["reachable"].normalize_calls == [(PAIR, Decimal("2000.00") / Decimal("101") - Decimal("10"))]


@pytest.mark.asyncio
async def test_sign_flip_and_amount_normalization_failure_discard_all_hidden_partial_plans():
    request = _request(target="-0.5", current=("1000", "1000"), amounts=("10", "10"))
    sessions = _sessions(request.portfolio, normalize_error="unreachable")

    proposal = await _propose(request, sessions, stop_loss=Decimal("110"), take_profit=Decimal("80"))

    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert proposal.risk.rejected_by == "connection_preflight"
    assert proposal.unavailable_connections == ("unreachable",)
    assert proposal.errors == ("connection unreachable: normalize_amount failed",)
    assert proposal.connection_risks[1].operation == "normalize_amount"
    assert "RAW_SECRET_MARKER" not in repr(proposal)


@pytest.mark.asyncio
@pytest.mark.parametrize("pair", [SPOT_PAIR, PAIR])
async def test_flat_close_uses_exact_snapshot_base_amount_across_the_spread(pair):
    request = _request(target="0", current=("1000", "2000"), amounts=("9.876", "20.125"), pair=pair)

    proposal = await _propose(request, _sessions(request.portfolio), pair=pair)

    assert proposal.ready is True
    first, second = proposal.connection_plans
    assert (first.side, first.amount, first.post_fill_signed_amount) == (
        "sell",
        Decimal("9.876"),
        Decimal("0.000"),
    )
    assert (second.side, second.amount, second.post_fill_signed_amount) == (
        "sell",
        Decimal("20.125"),
        Decimal("0.000"),
    )


@pytest.mark.asyncio
async def test_partial_reduction_uses_target_base_amount_and_rounds_without_crossing_target():
    request = _request(target="0.04", current=("1000", "2000"), amounts=("10", "20"))
    sessions = _sessions(request.portfolio)

    proposal = await _propose(request, sessions)

    plan = proposal.connection_plans[0]
    exact_target = Decimal("160.000") / Decimal("99")
    assert plan.side == "sell"
    assert plan.target_signed_amount == exact_target
    assert plan.delta_signed_amount == exact_target - Decimal("10")
    assert plan.amount == Decimal("8.383")
    assert plan.post_fill_signed_amount == Decimal("1.617")
    assert plan.post_fill_signed_amount >= plan.target_signed_amount
    assert sessions["reachable"].normalize_calls == [(PAIR, Decimal("10") - exact_target)]


@pytest.mark.asyncio
async def test_sign_flip_amount_is_exact_close_plus_new_target_leg_before_safe_rounding():
    request = _request(target="-0.1", current=("1000", "2000"), amounts=("10", "20"))

    proposal = await _propose(
        request,
        _sessions(request.portfolio),
        stop_loss=Decimal("110"),
        take_profit=Decimal("80"),
    )

    plan = proposal.connection_plans[0]
    exact_new_short = Decimal("-400.00") / Decimal("99")
    assert plan.side == "sell"
    assert plan.target_signed_amount == exact_new_short
    assert plan.delta_signed_amount == exact_new_short - Decimal("10")
    assert plan.amount == Decimal("14.040")
    assert plan.post_fill_signed_amount == Decimal("-4.040")
    assert plan.post_fill_signed_amount >= plan.target_signed_amount


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "stop_loss", "take_profit"),
    [
        ("0.5", None, None),
        ("0.5", Decimal("101"), None),
        ("0.5", None, Decimal("101")),
        ("-0.5", Decimal("99"), None),
        ("-0.5", None, Decimal("99")),
    ],
)
async def test_derivative_increase_rejects_missing_or_inverted_protection(target, stop_loss, take_profit):
    request = _request(target=target)

    proposal = await _propose(
        request,
        _sessions(request.portfolio),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert proposal.risk.rejected_by == "connection_preflight"
    assert tuple(decision.operation for decision in proposal.connection_risks) == (
        "validate_protection",
        "validate_protection",
    )


@pytest.mark.asyncio
async def test_partial_derivative_reduction_requires_valid_target_side_protection():
    request = _request(target="0.04", current=("1000", "2000"), amounts=("10", "20"))

    proposal = await _propose(
        request,
        _sessions(request.portfolio),
        stop_loss=Decimal("100"),
        take_profit=None,
    )

    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert tuple(decision.operation for decision in proposal.connection_risks) == (
        "validate_protection",
        "validate_protection",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["mixed_pair", "normalize_exception", "unsafe_normalization"])
async def test_reduction_invariants_abort_the_whole_proposal_instead_of_returning_a_sibling_plan(fault):
    request = _request(target="0", current=("1000", "2000"), amounts=("10", "20"))
    sessions = _sessions(request.portfolio)
    broken = sessions["unreachable"]
    if fault == "mixed_pair":
        foreign = ProtectionState(
            ("foreign",),
            Pair.parse("ETH/USDT:USDT"),
            "long",
            Decimal("1"),
            Decimal("90"),
            None,
            True,
            False,
        )
        object.__setattr__(broken._state, "protections", (foreign,))
    elif fault == "normalize_exception":
        broken._normalize_invariant_error = True
    else:
        broken._unsafe_normalization = True

    with pytest.raises(ValueError):
        await _propose(request, sessions)


@pytest.mark.asyncio
async def test_actual_venue_operation_failure_is_safely_classified_while_reduction_proceeds():
    request = _request(target="0", current=("1000", "2000"), amounts=("10", "20"))
    sessions = _sessions(request.portfolio, state_error="unreachable")

    proposal = await _propose(request, sessions)

    assert proposal.ready is True
    assert tuple(plan.connection_id for plan in proposal.connection_plans) == ("reachable",)
    assert proposal.unavailable_connections == ("unreachable",)
    assert tuple((decision.connection_id, decision.operation) for decision in proposal.connection_risks) == (
        ("reachable", ""),
        ("unreachable", "list_open_state"),
    )
    assert proposal.errors == ("connection unreachable: list_open_state failed",)
    assert "RAW_SECRET_MARKER" not in repr(proposal)


@pytest.mark.asyncio
async def test_concurrent_venue_failure_does_not_hide_a_state_invariant_failure():
    request = _request(target="0", current=("1000", "2000"), amounts=("10", "20"))
    sessions = _sessions(request.portfolio)
    broken = sessions["unreachable"]

    async def unavailable_quote(pair):
        raise VenueOperationError("RAW_SECRET_MARKER venue response")

    async def invalid_state(pair):
        raise ValueError("state contract violation")

    broken.fetch_quote = unavailable_quote
    broken.list_open_state = invalid_state

    with pytest.raises(ValueError, match="state contract violation"):
        await _propose(request, sessions)


@pytest.mark.asyncio
async def test_execution_dtos_reject_impossible_prices_transitions_capabilities_and_sets():
    request = _request(target="0.5", current=("1000", "1000"), amounts=("10", "10"))
    proposal = await _propose(request, _sessions(request.portfolio))
    plan = proposal.connection_plans[0]

    with pytest.raises(ValueError, match="ask"):
        replace(plan, execution_price=plan.quote.bid)
    with pytest.raises(ValueError, match="reduce_only"):
        replace(plan, reduce_only=True)
    with pytest.raises(ValueError, match="market order"):
        replace(
            plan,
            capabilities=VenueCapabilities(
                frozenset({"swap"}),
                True,
                False,
                True,
                frozenset({"limit"}),
            ),
        )
    with pytest.raises(ValueError, match="protection"):
        replace(plan, stop_loss=None, take_profit=None)
    with pytest.raises(ValueError, match="post_fill_signed_amount"):
        replace(plan, post_fill_signed_amount=plan.current_signed_amount)
    with pytest.raises(ValueError, match="ready proposal"):
        replace(proposal, ready=False)
    with pytest.raises(ValueError, match="disjoint"):
        replace(proposal, unavailable_connections=(plan.connection_id,))
    with pytest.raises(ValueError, match="configured order"):
        replace(proposal, connection_risks=tuple(reversed(proposal.connection_risks)))
    with pytest.raises(ValueError, match="configured order"):
        replace(proposal, connection_risks=(), connection_plans=())
    with pytest.raises(ValueError, match="errors must match"):
        replace(proposal, errors=("RAW_SECRET_MARKER",))
    with pytest.raises(ValueError, match="venue operation failures"):
        replace(
            proposal,
            connection_plans=(proposal.connection_plans[0],),
            unavailable_connections=("unreachable",),
        )


@pytest.mark.asyncio
async def test_book_proposal_rejects_risk_targets_from_a_foreign_book():
    request = _request(target="0.5", current=("1000", "1000"), amounts=("10", "10"))
    proposal = await _propose(request, _sessions(request.portfolio))
    foreign_risk = replace(
        proposal.risk,
        connection_targets=tuple(replace(target, book_id="foreign") for target in proposal.risk.connection_targets),
    )

    with pytest.raises(ValueError, match="proposal book"):
        replace(proposal, risk=foreign_risk)


@pytest.mark.asyncio
async def test_book_proposal_rejects_a_coherent_plan_for_a_different_risk_target_notional():
    request = _request(target="0.5", current=("1000", "1000"), amounts=("10", "10"))
    proposal = await _propose(request, _sessions(request.portfolio))
    plan = proposal.connection_plans[0]
    foreign_target_notional = Decimal("2100")
    foreign_target_amount = foreign_target_notional / plan.execution_price
    foreign_delta_amount = foreign_target_amount - plan.current_signed_amount
    coherent_foreign_plan = replace(
        plan,
        target_signed_notional=foreign_target_notional,
        delta_signed_notional=foreign_target_notional - plan.current_signed_notional,
        target_signed_amount=foreign_target_amount,
        delta_signed_amount=foreign_delta_amount,
        post_fill_signed_amount=foreign_target_amount,
        amount=foreign_delta_amount,
    )

    with pytest.raises(ValueError, match="risk target"):
        replace(proposal, connection_plans=(coherent_foreign_plan, proposal.connection_plans[1]))
