"""Fixed UTC account evidence, never an external venue or runtime database."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from cryptotrader.accounts.models import AccountSnapshot, Fill, FillPage, FundingEntry, FundingPage, Instrument, Money
from cryptotrader.pair import Pair

START = datetime(2026, 8, 1, tzinfo=UTC)
END = START + timedelta(days=2)
PAIR = Pair.parse("BTC/USDT")
INSTRUMENT = Instrument("BTC/USDT", PAIR, "spot", True)


def snapshot(connection_id="sim-a", *, amount="1000", currency="USDT", observed_at=END):
    return AccountSnapshot(
        connection_id,
        observed_at,
        "simulated",
        Money(Decimal(amount), currency),
        (Money(Decimal(amount), currency),),
        (),
        (),
        Money(Decimal("0"), currency),
        Money(Decimal(amount), currency),
        (),
    )


def fills(connection_id="sim-a"):
    return tuple(
        Fill(
            connection_id,
            f"fill-{index}",
            f"order-{index}",
            INSTRUMENT,
            side,
            Decimal(amount),
            Decimal(price),
            START + timedelta(hours=index),
            Money(Decimal("1"), "USDT"),
            Money(None, "USDT", "platform_not_provided"),
            "platform",
            f"client{index}",
        )
        for index, (side, amount, price) in enumerate(
            (("buy", "2", "100"), ("buy", "1", "130"), ("sell", "1", "140")), 1
        )
    )


def account_from_portfolio(portfolio):
    """Explicitly denominate the old single-pair test fixture in its quote currency."""
    from cryptotrader.accounts.models import AccountPosition
    from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot

    if not isinstance(portfolio, ConnectionPortfolioSnapshot):
        return portfolio
    position = portfolio.position
    currency = position.pair.settle or position.pair.quote
    return AccountSnapshot(
        portfolio.connection_id,
        END,
        "simulated",
        Money(portfolio.equity, currency),
        tuple(Money(amount, asset) for asset, amount in portfolio.balances.items()),
        (
            AccountPosition(
                Instrument(str(position.pair), position.pair, position.pair.market_type, True),
                position.signed_amount,
                position.signed_amount,
                Money(position.signed_notional, currency),
                position.entry_price,
                Money(None, currency, "fixture valuation unavailable"),
            ),
        ),
        (),
        Money(Decimal("0"), currency),
        Money(portfolio.equity, currency),
        (),
    )


class AccountSession:
    history_from_inception = True

    def __init__(self, connection_id="sim-a"):
        self.connection_id = connection_id
        self.snapshot = snapshot(connection_id)
        self.fills = fills(connection_id)
        self.funding = (
            FundingEntry(connection_id, "fund-1", INSTRUMENT, Money(Decimal("2"), "USDT"), START + timedelta(hours=5)),
        )
        self.error = False
        self.reads = []

    async def fetch_account(self):
        if self.error:
            raise RuntimeError("SECRET: must never be stored")
        return self.snapshot

    async def close(self):
        pass

    async def fetch_fills(self, cursor):
        self.reads.append(("fills", cursor))
        return FillPage(
            self.fills, "fills-end" if cursor is None else "fills-next", True, START if cursor is None else END, END
        )

    async def fetch_funding(self, cursor):
        self.reads.append(("funding", cursor))
        return FundingPage(
            self.funding, "fund-end" if cursor is None else "fund-next", True, START if cursor is None else END, END
        )

    @asynccontextmanager
    async def provide(self, connection_id):
        assert connection_id == self.connection_id
        yield self
