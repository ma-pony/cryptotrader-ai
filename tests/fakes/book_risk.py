"""Explicit single-pair account fixtures for old isolated unit tests, never production fallback."""

from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal

from cryptotrader.accounts.models import AccountPosition, Instrument, Money
from cryptotrader.risk.book_state import BookRiskState
from cryptotrader.venues.models import VenueQuote
from tests.fakes.account_session import account_from_portfolio, snapshot
from tests.test_execution_service import SPOT_CAPABILITIES, SPOT_PAIR, _VenueSession


def with_accounts(portfolio):
    return replace(
        portfolio,
        connections=tuple(
            replace(p, account_snapshot=replace(account_from_portfolio(p), capital_scope=portfolio.capital_scope))
            for p in portfolio.connections
        ),
    )


def state_for(portfolio, peak=None):
    value = with_accounts(portfolio)
    return BookRiskState.from_snapshots(
        value.book_id, tuple(p.account_snapshot for p in value.connections), peak_equity=peak
    )


class MemoryRiskStates:
    async def update(self, book_id, snapshots):
        return BookRiskState.from_snapshots(book_id, snapshots)


class NoContentionOwnership:
    @asynccontextmanager
    async def book(self, book_id):
        yield


class QuotedAccountSession(_VenueSession):
    """Offline account with explicit equity and member-specific spot prices."""

    def __init__(self, connection_id, amount, price="100"):
        super().__init__(amount, quote=VenueQuote(SPOT_PAIR, Decimal(price), Decimal(price), Decimal(price)))
        self.connection_id = connection_id
        self.connection = replace(self.connection, id=connection_id)
        self.capabilities = SPOT_CAPABILITIES
        self.incomplete = ()
        self.equity = Decimal("100")
        self.available_margin = Decimal("100")

    async def list_instruments(self):
        return (Instrument(SPOT_PAIR.canonical(), SPOT_PAIR, "spot", True),)

    async def fetch_account(self):
        position = AccountPosition(
            (await self.list_instruments())[0],
            self.signed_amount,
            abs(self.signed_amount),
            Money(self.signed_amount * self.quote.last, "USDT"),
            None,
            Money(Decimal("0"), "USDT"),
        )
        return replace(
            snapshot(self.connection_id, amount="100"),
            positions=(position,),
            equity=Money(self.equity, "USDT", "unknown_equity" if self.equity is None else None),
            available_margin=Money(self.available_margin, "USDT"),
            completeness=self.incomplete,
        )
