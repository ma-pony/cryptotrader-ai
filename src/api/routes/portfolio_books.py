"""只读取 active Runtime session 的资金池 portfolio API。"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from cryptotrader.pair import Pair
from cryptotrader.portfolio.aggregator import PortfolioReadError

if TYPE_CHECKING:
    from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot

router = APIRouter(prefix="/api/portfolio/books", tags=["portfolio-books"])
_DEFAULT_PAIR = "BTC/USDT"


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
    equity: Decimal
    balances: list[AssetBalanceOut]
    position: PositionOut


class BookPortfolioOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str
    capital_scope: Literal["simulated", "real"]
    pair: str
    total_equity: Decimal
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


def _active_parts(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    cycle = getattr(runtime, "cycle", None) if runtime is not None else None
    aggregator = getattr(cycle, "portfolios", None) if cycle is not None else None
    sessions = getattr(runtime, "sessions", None) if runtime is not None else None
    if runtime is None or cycle is None or aggregator is None or sessions is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not active")
    return runtime, aggregator, sessions


def _resolved_pair(runtime, requested: str | None) -> Pair:
    value = requested
    if value is None:
        pairs = runtime.snapshot.document.scheduler.pairs
        value = str(pairs[0]) if pairs else _DEFAULT_PAIR
    try:
        return Pair.parse(value)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Pair is invalid") from error


async def _read_book(request: Request, book_id: str, requested_pair: str | None):
    runtime, aggregator, sessions = _active_parts(request)
    pair = _resolved_pair(runtime, requested_pair)
    book = next(
        (item for item in runtime.snapshot.document.execution.books if item.enabled and item.id == book_id),
        None,
    )
    if book is None:
        raise HTTPException(status_code=404, detail="Portfolio book not found")
    try:
        snapshot = await aggregator.read(book, sessions, pair)
    except PortfolioReadError as error:
        raise HTTPException(status_code=503, detail="Portfolio session is unavailable") from error
    return pair, snapshot


def _scope(books: list[BookPortfolioOut]) -> PortfolioScopeOut:
    return PortfolioScopeOut(
        books=books,
        totals=ScopeTotalsOut(
            equity=sum((book.total_equity for book in books), Decimal("0")),
            signed_notional=sum((book.total_signed_notional for book in books), Decimal("0")),
        ),
    )


@router.get("", response_model=PortfolioBooksOut)
async def list_portfolio_books(
    request: Request,
    pair: str | None = Query(default=None),
) -> PortfolioBooksOut:
    runtime, aggregator, sessions = _active_parts(request)
    resolved = _resolved_pair(runtime, pair)
    try:
        snapshots = [
            await aggregator.read(book, sessions, resolved)
            for book in runtime.snapshot.document.execution.books
            if book.enabled
        ]
    except PortfolioReadError as error:
        raise HTTPException(status_code=503, detail="Portfolio session is unavailable") from error
    simulated = [book_portfolio_out(item) for item in snapshots if item.capital_scope == "simulated"]
    real = [book_portfolio_out(item) for item in snapshots if item.capital_scope == "real"]
    return PortfolioBooksOut(pair=resolved.canonical(), simulated=_scope(simulated), real=_scope(real))


@router.get("/{book_id}", response_model=BookPortfolioOut)
async def get_portfolio_book(
    book_id: str,
    request: Request,
    pair: str | None = Query(default=None),
) -> BookPortfolioOut:
    _, snapshot = await _read_book(request, book_id, pair)
    return book_portfolio_out(snapshot)
