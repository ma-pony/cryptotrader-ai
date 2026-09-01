"""只读取 active Runtime session 的资金池 portfolio API。"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves these annotations at runtime.
from decimal import Decimal  # noqa: TC003 - Pydantic resolves these annotations at runtime.
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from api.routes.accounts import AccountOut, MoneyOut, _parts, account_out
from cryptotrader.accounts.income import sum_money
from cryptotrader.accounts.models import Money
from cryptotrader.accounts.store import effective_memberships, payload

if TYPE_CHECKING:
    from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot

router = APIRouter(prefix="/api/portfolio/books", tags=["portfolio-books"])


class InstrumentRiskOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument: str
    signed_notional: Decimal | None


class BookRiskStateOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: str
    capital_scope: Literal["simulated", "real"]
    valuation_currency: str
    observed_at: datetime
    equity: Decimal | None
    peak_equity: Decimal | None
    positions_by_instrument: list[InstrumentRiskOut]
    pending_increase_notional: Decimal | None
    gross_notional: Decimal | None
    net_notional: Decimal | None
    used_margin: Decimal | None
    available_margin: Decimal | None
    completeness: list[str]


def risk_state_out(state):
    if state is None:
        return None
    value = payload(state)
    value.pop("snapshots")
    value["positions_by_instrument"] = [
        {"instrument": key, "signed_notional": amount}
        for key, amount in sorted(value["positions_by_instrument"].items())
    ]
    return BookRiskStateOut.model_validate(value)


class PositionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair: str
    signed_amount: Decimal
    signed_notional: Decimal
    entry_price: Decimal | None


class AssetBalanceOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: str
    amount: Decimal


class ConnectionPortfolioOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    equity: Decimal | None
    balances: list[AssetBalanceOut]
    position: PositionOut


class BookPortfolioOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str
    capital_scope: Literal["simulated", "real"]
    pair: str
    total_equity: Decimal | None
    total_signed_notional: Decimal
    connections: list[ConnectionPortfolioOut]


class ScopeTotalsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equity: Decimal
    signed_notional: Decimal


class PortfolioScopeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    books: list[BookPortfolioOut]
    totals: ScopeTotalsOut


class PortfolioBooksOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair: str
    simulated: PortfolioScopeOut
    real: PortfolioScopeOut


def connection_portfolio_out(snapshot: ConnectionPortfolioSnapshot) -> ConnectionPortfolioOut:
    position = snapshot.position
    return ConnectionPortfolioOut(
        connection_id=snapshot.connection_id,
        equity=snapshot.equity,
        balances=[AssetBalanceOut(asset=asset, amount=amount) for asset, amount in sorted(snapshot.balances.items())],
        position=PositionOut(
            pair=position.pair.canonical(),
            signed_amount=position.signed_amount,
            signed_notional=position.signed_notional,
            entry_price=position.entry_price,
        ),
    )


def book_portfolio_out(snapshot: BookPortfolioSnapshot) -> BookPortfolioOut:
    return BookPortfolioOut(
        book_id=snapshot.book_id,
        capital_scope=snapshot.capital_scope,
        pair=snapshot.connections[0].position.pair.canonical(),
        total_equity=snapshot.total_equity,
        total_signed_notional=snapshot.total_signed_notional,
        connections=[connection_portfolio_out(item) for item in snapshot.connections],
    )


class AccountBookOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    book_id: str
    label: str
    capital_scope: Literal["simulated", "real"]
    enabled: bool
    total_equity: list[MoneyOut]
    total_signed_notional: list[MoneyOut]
    connections: list[AccountOut]
    risk_state: BookRiskStateOut | None


class AccountScopeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    books: list[AccountBookOut]
    equity: list[MoneyOut]
    signed_notional: list[MoneyOut]


class AccountBooksOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    simulated: AccountScopeOut
    real: AccountScopeOut


async def _account_book(store, document, book):
    from cryptotrader.risk.book_state import BookRiskStateStore

    risk_state = await BookRiskStateStore(store).get(book.id)
    ids = {connection_id for connection_id, book_id in effective_memberships(document).items() if book_id == book.id}
    connections = [
        await account_out(store, document, connection)
        for connection in document.execution.connections
        if connection.id in ids
    ]
    equities, notionals = [], []
    for connection in connections:
        snapshot = await store.latest(connection.connection_id)
        if snapshot is None:
            equities.append(Money(None, "UNKNOWN", "账户尚未同步"))
            notionals.append(Money(None, "UNKNOWN", "账户尚未同步"))
        else:
            equities.append(snapshot.equity)
            notionals.extend(position.signed_notional for position in snapshot.positions)
    return AccountBookOut(
        book_id=book.id,
        label=book.label,
        capital_scope=book.capital_scope,
        enabled=book.enabled,
        total_equity=payload(sum_money(equities)),
        total_signed_notional=payload(sum_money(notionals)),
        connections=connections,
        risk_state=risk_state_out(risk_state),
    )


@router.get("", response_model=AccountBooksOut)
async def list_portfolio_books(request: Request):
    runtime, store = _parts(request)
    document = (await runtime.repository.get_existing()).document
    books = [await _account_book(store, document, book) for book in document.execution.books]
    accounts = [await account_out(store, document, connection) for connection in document.execution.connections]

    def scope(capital_scope):
        # Count a connection once even if it appears in several disabled draft pools.
        ids = {item.connection_id for book in books if book.capital_scope == capital_scope for item in book.connections}
        values = [
            Money(item.snapshot.equity.amount, item.snapshot.equity.currency, item.snapshot.equity.unavailable_reason)
            if item.snapshot
            else Money(None, "UNKNOWN", "账户尚未同步")
            for item in accounts
            if item.connection_id in ids
        ]
        notionals = [
            Money(
                position.signed_notional.amount,
                position.signed_notional.currency,
                position.signed_notional.unavailable_reason,
            )
            for item in accounts
            if item.connection_id in ids and item.snapshot
            for position in item.snapshot.positions
        ]
        return AccountScopeOut(
            books=[book for book in books if book.capital_scope == capital_scope],
            equity=payload(sum_money(values)),
            signed_notional=payload(sum_money(notionals)),
        )

    return AccountBooksOut(simulated=scope("simulated"), real=scope("real"))


@router.get("/{book_id}", response_model=AccountBookOut)
async def get_portfolio_book(book_id: str, request: Request):
    runtime, store = _parts(request)
    document = (await runtime.repository.get_existing()).document
    book = next((item for item in document.execution.books if item.id == book_id), None)
    if book is None:
        raise HTTPException(404, "Portfolio book not found")
    return await _account_book(store, document, book)
