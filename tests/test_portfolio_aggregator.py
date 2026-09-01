"""Execution-book portfolio aggregation contracts."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import ConnectionPosition
from tests.fakes.account_session import account_from_portfolio

PAIR = Pair.parse("BTC/USDT:USDT")


def _portfolio(connection_id: str, equity: str, signed_notional: str = "0") -> ConnectionPortfolioSnapshot:
    snapshot = ConnectionPortfolioSnapshot(
        connection_id=connection_id,
        equity=Decimal(equity),
        balances={"USDT": Decimal(equity)},
        position=ConnectionPosition(PAIR, Decimal("0"), Decimal(signed_notional), None),
    )
    return replace(snapshot, account_snapshot=account_from_portfolio(snapshot))


def _book(*allocations: ConnectionAllocation) -> ExecutionBook:
    return ExecutionBook("simulation", "Simulation", "simulated", True, False, allocations)


class _Session:
    def __init__(self, snapshot: object, *, connection_id: str | None = None, error: Exception | None = None):
        self.connection_id = connection_id or getattr(snapshot, "connection_id", "unknown")
        self.snapshot = snapshot
        self.error = error
        self.calls = 0

    async def fetch_account(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return account_from_portfolio(self.snapshot)


async def test_aggregator_reads_enabled_book_connections_concurrently_and_preserves_configuration_order():
    from cryptotrader.portfolio.aggregator import PortfolioAggregator

    started: set[str] = set()
    release = asyncio.Event()

    class ConcurrentSession(_Session):
        async def fetch_account(self):
            self.calls += 1
            started.add(self.connection_id)
            if started == {"bybit-testnet", "okx-demo"}:
                release.set()
            await release.wait()
            return account_from_portfolio(self.snapshot)

    bybit = ConcurrentSession(_portfolio("bybit-testnet", "20000", "-1000"))
    okx = ConcurrentSession(_portfolio("okx-demo", "10000", "2000"))
    disabled = _Session(_portfolio("paper-disabled", "500000", "500000"))
    outside = _Session(_portfolio("bybit-live", "999999", "999999"))
    book = _book(
        ConnectionAllocation("bybit-testnet", True, 0.6),
        ConnectionAllocation("paper-disabled", False, 0.0),
        ConnectionAllocation("okx-demo", True, 0.4),
    )

    result = await asyncio.wait_for(
        PortfolioAggregator().read(
            book,
            {
                "okx-demo": okx,
                "bybit-testnet": bybit,
                "paper-disabled": disabled,
                "bybit-live": outside,
            },
            PAIR,
        ),
        timeout=1,
    )

    assert result == BookPortfolioSnapshot(
        book_id="simulation",
        capital_scope="simulated",
        total_equity=Decimal("30000"),
        total_signed_notional=Decimal("1000"),
        connections=(bybit.snapshot, okx.snapshot),
    )
    assert tuple(item.connection_id for item in result.connections) == ("bybit-testnet", "okx-demo")
    assert bybit.calls == okx.calls == 1
    assert disabled.calls == outside.calls == 0


@pytest.mark.parametrize(
    ("sessions", "connection_id"),
    [
        ({}, "okx-demo"),
        ({"okx-demo": _Session(_portfolio("other", "100"), connection_id="other")}, "okx-demo"),
    ],
)
async def test_aggregator_rejects_missing_or_mismatched_sessions_by_connection_id(sessions, connection_id):
    from cryptotrader.portfolio.aggregator import PortfolioAggregator, PortfolioReadError

    with pytest.raises(PortfolioReadError, match=connection_id):
        await PortfolioAggregator().read(
            _book(ConnectionAllocation("okx-demo", True, 1.0)),
            sessions,
            PAIR,
        )


@pytest.mark.parametrize(
    "session",
    [
        _Session(_portfolio("other", "100"), connection_id="okx-demo"),
        _Session(object(), connection_id="okx-demo"),
        _Session(_portfolio("okx-demo", "100"), error=RuntimeError("venue unavailable")),
    ],
)
async def test_aggregator_fails_the_whole_read_and_names_the_failed_connection(session):
    from cryptotrader.portfolio.aggregator import PortfolioAggregator, PortfolioReadError

    with pytest.raises(PortfolioReadError, match="okx-demo"):
        await PortfolioAggregator().read(
            _book(ConnectionAllocation("okx-demo", True, 1.0)),
            {"okx-demo": session},
            PAIR,
        )


async def test_aggregator_redacts_venue_failure_and_cancels_and_reaps_sibling_reads():
    from cryptotrader.portfolio.aggregator import PortfolioAggregator, PortfolioReadError

    credential_marker = "CREDENTIAL_MARKER_DO_NOT_LEAK_7F3A"
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    sibling_reaped = asyncio.Event()

    class BlockingSession(_Session):
        async def fetch_account(self):
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            finally:
                sibling_reaped.set()

    class FailingSession(_Session):
        async def fetch_account(self):
            await sibling_started.wait()
            raise RuntimeError(credential_marker)

    book = _book(
        ConnectionAllocation("blocking", True, 0.5),
        ConnectionAllocation("failed", True, 0.5),
    )
    sessions = {
        "blocking": BlockingSession(_portfolio("blocking", "50")),
        "failed": FailingSession(_portfolio("failed", "50")),
    }

    with pytest.raises(PortfolioReadError) as exc_info:
        await PortfolioAggregator().read(book, sessions, PAIR)

    error = exc_info.value
    formatted = "".join(traceback.format_exception(error))
    assert str(error) == "failed to read portfolio for connection failed"
    assert credential_marker not in str(error)
    assert credential_marker not in repr(error)
    assert credential_marker not in formatted
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sibling_cancelled.is_set()
    assert sibling_reaped.is_set()


def test_book_portfolio_snapshot_is_frozen_decimal_and_rejects_inconsistent_totals():
    first = _portfolio("first", "40", "8")
    second = _portfolio("second", "60", "-3")
    snapshot = BookPortfolioSnapshot(
        "simulation",
        "simulated",
        Decimal("100"),
        Decimal("5"),
        (first, second),
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.total_equity = Decimal("0")
    with pytest.raises(ValueError, match="total_equity"):
        BookPortfolioSnapshot("simulation", "simulated", Decimal("99"), Decimal("5"), (first, second))
    with pytest.raises(ValueError, match="total_signed_notional"):
        BookPortfolioSnapshot("simulation", "simulated", Decimal("100"), Decimal("4"), (first, second))


@pytest.mark.parametrize(
    "overrides",
    [
        {"book_id": ""},
        {"capital_scope": "paper"},
        {"total_equity": Decimal("NaN")},
        {"total_equity": Decimal("-1")},
        {"total_signed_notional": Decimal("Infinity")},
        {"connections": []},
        {"connections": (_portfolio("same", "50"), _portfolio("same", "50"))},
    ],
)
def test_book_portfolio_snapshot_rejects_invalid_strict_inputs(overrides):
    values = {
        "book_id": "simulation",
        "capital_scope": "simulated",
        "total_equity": Decimal("100"),
        "total_signed_notional": Decimal("0"),
        "connections": (_portfolio("one", "100"),),
    }

    with pytest.raises(ValueError):
        BookPortfolioSnapshot(**(values | overrides))
