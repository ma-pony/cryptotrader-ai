"""Book proposal planning preflights normalized venue sessions without brand branches."""

from __future__ import annotations

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

PAIR = Pair.parse("BTC/USDT:USDT")
CAPABILITIES = VenueCapabilities(frozenset({"swap"}), True, False, True, frozenset({"market"}))


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


def _portfolio(*, current: tuple[str, str] = ("0", "0")) -> BookPortfolioSnapshot:
    snapshots = tuple(
        ConnectionPortfolioSnapshot(
            connection_id,
            Decimal("5000"),
            {"USDT": Decimal("5000")},
            ConnectionPosition(PAIR, Decimal(notional) / Decimal("100"), Decimal(notional), Decimal("100")),
        )
        for connection_id, notional in zip(("reachable", "unreachable"), current, strict=True)
    )
    return BookPortfolioSnapshot(
        "simulation",
        "simulated",
        Decimal("10000"),
        sum((Decimal(item) for item in current), Decimal("0")),
        snapshots,
    )


def _request(*, target: str = "0.5", current: tuple[str, str] = ("0", "0")):
    from cryptotrader.risk.models import BookRiskRequest

    return BookRiskRequest(_book(), _portfolio(current=current), Decimal(target), Decimal("10000"))


class _Session:
    def __init__(
        self,
        snapshot: ConnectionPortfolioSnapshot,
        *,
        unavailable: bool = False,
        normalize_error: bool = False,
        protection_ids: tuple[str, ...] = (),
    ) -> None:
        self.connection_id = snapshot.connection_id
        self.capabilities = CAPABILITIES
        self._snapshot = snapshot
        self._unavailable = unavailable
        self._normalize_error = normalize_error
        self.normalize_calls: list[tuple[Pair, Decimal]] = []
        protections = ()
        if protection_ids:
            protections = (
                ProtectionState(
                    protection_ids,
                    PAIR,
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
            raise RuntimeError("RAW_SECRET_MARKER venue response")
        return VenueQuote(pair, Decimal("99"), Decimal("101"), Decimal("100"))

    async def list_open_state(self, pair):
        if self._unavailable:
            raise RuntimeError("RAW_SECRET_MARKER venue response")
        return self._state

    async def normalize_amount(self, pair, base_amount):
        self.normalize_calls.append((pair, base_amount))
        if self._normalize_error:
            raise RuntimeError("RAW_SECRET_MARKER precision response")
        return base_amount.quantize(Decimal("0.001"), rounding=ROUND_DOWN)


def _sessions(
    portfolio: BookPortfolioSnapshot,
    *,
    unavailable: str | None = None,
    normalize_error: str | None = None,
    protection_ids: tuple[str, ...] = (),
):
    result = {
        snapshot.connection_id: _Session(
            snapshot,
            unavailable=snapshot.connection_id == unavailable,
            normalize_error=snapshot.connection_id == normalize_error,
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


async def _propose(request, sessions):
    return await _planner().propose(
        request,
        sessions,
        pair=PAIR,
        stop_loss=Decimal("90"),
        take_profit=Decimal("120"),
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
    request = _request(target="0", current=("2000", "3000"))
    sessions = _sessions(request.portfolio, unavailable="unreachable")

    proposal = await _propose(request, sessions)

    assert proposal.ready is True
    assert tuple(plan.connection_id for plan in proposal.connection_plans) == ("reachable",)
    assert proposal.connection_plans[0].reduce_only is True
    assert proposal.unavailable_connections == ("unreachable",)


@pytest.mark.asyncio
async def test_planner_preserves_allocation_order_and_builds_exact_normalized_plan_fields():
    request = _request(target="0.5", current=("1000", "1000"))
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
    assert plan.amount == Decimal("9.900")
    assert plan.reduce_only is False
    assert plan.market_type == "swap"
    assert plan.stop_loss == Decimal("90")
    assert plan.take_profit == Decimal("120")
    assert plan.old_protection_ids == ("old-a", "old-b")
    assert plan.capabilities == CAPABILITIES
    assert sessions["reachable"].normalize_calls == [(PAIR, Decimal("1000.00") / Decimal("101"))]


@pytest.mark.asyncio
async def test_sign_flip_and_amount_normalization_failure_discard_all_hidden_partial_plans():
    request = _request(target="-0.5", current=("1000", "1000"))
    sessions = _sessions(request.portfolio, normalize_error="unreachable")

    proposal = await _propose(request, sessions)

    assert proposal.ready is False
    assert proposal.connection_plans == ()
    assert proposal.risk.rejected_by == "connection_preflight"
    assert proposal.unavailable_connections == ("unreachable",)
    assert proposal.errors == ("connection unreachable: preflight unavailable",)
    assert "RAW_SECRET_MARKER" not in repr(proposal)
