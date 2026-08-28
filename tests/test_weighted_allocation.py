"""Deterministic execution-book allocation contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.models import ConnectionAllocation, ConnectionTarget, ExecutionBook
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import ConnectionPosition

PAIR = Pair.parse("BTC/USDT:USDT")


def _book(*allocations: ConnectionAllocation, book_id: str = "simulation") -> ExecutionBook:
    return ExecutionBook(book_id, book_id, "simulated", True, False, allocations)


def _connection(connection_id: str, equity: str) -> ConnectionPortfolioSnapshot:
    return ConnectionPortfolioSnapshot(
        connection_id,
        Decimal(equity),
        {"USDT": Decimal(equity)},
        ConnectionPosition(PAIR, Decimal("0"), Decimal("0"), None),
    )


def _portfolio(*connections: ConnectionPortfolioSnapshot, book_id: str = "simulation") -> BookPortfolioSnapshot:
    return BookPortfolioSnapshot(
        book_id,
        "simulated",
        sum((connection.equity for connection in connections), Decimal("0")),
        Decimal("0"),
        connections,
    )


@pytest.mark.parametrize(
    ("side", "ratio", "expected_exposure", "expected_notionals"),
    [
        ("long", 0.5, Decimal("0.5"), (Decimal("20000.00"), Decimal("30000.00"))),
        ("short", 0.5, Decimal("-0.5"), (Decimal("-20000.00"), Decimal("-30000.00"))),
        ("flat", 0.0, Decimal("0.0"), (Decimal("0.00"), Decimal("0.00"))),
    ],
)
def test_weighted_allocation_uses_total_book_equity_signed_exposure_and_explicit_weights(
    side,
    ratio,
    expected_exposure,
    expected_notionals,
):
    from cryptotrader.execution.allocation import WeightedAllocationPolicy

    book = _book(ConnectionAllocation("okx-demo", True, 0.4), ConnectionAllocation("bybit-testnet", True, 0.6))
    portfolio = _portfolio(_connection("okx-demo", "10000"), _connection("bybit-testnet", "90000"))

    targets = WeightedAllocationPolicy().allocate(TargetPosition(side, ratio), book, portfolio)

    assert tuple(item.target_signed_notional for item in targets) == expected_notionals
    assert tuple(item.target_exposure for item in targets) == (expected_exposure, expected_exposure)
    assert tuple(item.weight for item in targets) == (Decimal("0.4"), Decimal("0.6"))
    assert all(item.book_equity == Decimal("100000") for item in targets)


def test_weighted_allocation_preserves_order_skips_disabled_and_does_not_renormalize():
    from cryptotrader.execution.allocation import WeightedAllocationPolicy

    book = _book(
        ConnectionAllocation("second", True, 0.3),
        ConnectionAllocation("disabled", False, 0.5),
        ConnectionAllocation("first", True, 0.2),
    )
    portfolio = _portfolio(_connection("second", "40"), _connection("first", "60"))

    targets = WeightedAllocationPolicy().allocate(TargetPosition("long", 1.0), book, portfolio)

    assert tuple(item.connection_id for item in targets) == ("second", "first")
    assert tuple(item.target_signed_notional for item in targets) == (Decimal("30.0"), Decimal("20.0"))


def test_single_connection_at_one_hundred_percent_is_normal_case():
    from cryptotrader.execution.allocation import WeightedAllocationPolicy

    book = _book(ConnectionAllocation("paper-local", True, 1.0))
    portfolio = _portfolio(_connection("paper-local", "8000"))

    assert WeightedAllocationPolicy().allocate(TargetPosition("long", 0.25), book, portfolio) == (
        ConnectionTarget(
            book_id="simulation",
            connection_id="paper-local",
            weight=Decimal("1.0"),
            book_equity=Decimal("8000"),
            target_exposure=Decimal("0.25"),
            target_signed_notional=Decimal("2000.000"),
        ),
    )


@pytest.mark.parametrize(
    ("book", "portfolio", "message"),
    [
        (
            _book(ConnectionAllocation("one", True, 1.0), book_id="book-a"),
            _portfolio(_connection("one", "100"), book_id="book-b"),
            "book_id",
        ),
        (
            _book(ConnectionAllocation("one", True, 1.0)),
            _portfolio(_connection("other", "100")),
            "connections",
        ),
        (
            _book(ConnectionAllocation("one", True, 0.5), ConnectionAllocation("two", True, 0.5)),
            _portfolio(_connection("two", "50"), _connection("one", "50")),
            "connections",
        ),
    ],
)
def test_weighted_allocation_rejects_mismatched_book_and_portfolio(book, portfolio, message):
    from cryptotrader.execution.allocation import WeightedAllocationPolicy

    with pytest.raises(ValueError, match=message):
        WeightedAllocationPolicy().allocate(TargetPosition("long", 0.5), book, portfolio)


def test_weighted_allocation_rejects_mismatched_capital_scope():
    from cryptotrader.execution.allocation import WeightedAllocationPolicy

    book = _book(ConnectionAllocation("one", True, 1.0))
    portfolio = BookPortfolioSnapshot(
        "simulation",
        "real",
        Decimal("100"),
        Decimal("0"),
        (_connection("one", "100"),),
    )

    with pytest.raises(ValueError, match="capital_scope"):
        WeightedAllocationPolicy().allocate(TargetPosition("long", 0.5), book, portfolio)


def test_connection_target_is_frozen_and_rejects_inconsistent_or_non_finite_values():
    target = ConnectionTarget(
        "simulation",
        "paper-local",
        Decimal("0.4"),
        Decimal("100"),
        Decimal("-0.5"),
        Decimal("-20.0"),
    )

    with pytest.raises(FrozenInstanceError):
        target.target_exposure = Decimal("0")
    with pytest.raises(ValueError, match="target_signed_notional"):
        ConnectionTarget("simulation", "paper-local", Decimal("0.4"), Decimal("100"), Decimal("-0.5"), Decimal("20"))
    with pytest.raises(ValueError, match="target_exposure"):
        ConnectionTarget("simulation", "paper-local", Decimal("0.4"), Decimal("100"), Decimal("NaN"), Decimal("0"))


@pytest.mark.parametrize(
    "values",
    [
        ("", "paper", Decimal("1"), Decimal("100"), Decimal("1"), Decimal("100")),
        ("book", "", Decimal("1"), Decimal("100"), Decimal("1"), Decimal("100")),
        ("book", "paper", Decimal("Infinity"), Decimal("100"), Decimal("1"), Decimal("100")),
        ("book", "paper", Decimal("-0.1"), Decimal("100"), Decimal("1"), Decimal("-10")),
        ("book", "paper", Decimal("1.1"), Decimal("100"), Decimal("1"), Decimal("110")),
        ("book", "paper", Decimal("1"), Decimal("-1"), Decimal("1"), Decimal("-1")),
        ("book", "paper", Decimal("1"), Decimal("100"), Decimal("1.1"), Decimal("110")),
        ("book", "paper", Decimal("1"), Decimal("100"), Decimal("1"), Decimal("Infinity")),
    ],
)
def test_connection_target_rejects_invalid_strict_inputs(values):
    with pytest.raises(ValueError):
        ConnectionTarget(*values)
