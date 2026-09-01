"""Account history reads never contact a venue. Only POST sync refreshes facts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from cryptotrader.accounts.income import IncomeService, sum_money
from cryptotrader.accounts.store import effective_memberships, payload
from cryptotrader.accounts.sync import AccountSyncError
from cryptotrader.configuration.catalog import require_environment

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


class StrictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoneyOut(StrictOut):
    amount: Decimal | None
    currency: str
    unavailable_reason: str | None


class InstrumentOut(StrictOut):
    venue_symbol: str
    pair: str | None
    market_type: str
    tradable: bool
    reason: str | None


class PositionOut(StrictOut):
    instrument: InstrumentOut
    signed_amount: Decimal
    available_amount: Decimal | None
    signed_notional: MoneyOut
    entry_price: Decimal | None
    unrealized_pnl: MoneyOut


class OrderOut(StrictOut):
    connection_id: str
    venue_order_id: str
    instrument: InstrumentOut
    side: str
    order_type: str
    amount: Decimal
    filled_amount: Decimal
    average_price: Decimal | None
    status: str
    reduce_only: bool
    protection: bool
    client_order_id: str | None
    observed_at: datetime
    remaining_notional: MoneyOut


class SnapshotOut(StrictOut):
    connection_id: str
    observed_at: datetime
    capital_scope: Literal["simulated", "real"]
    valuation_notes: list[str]
    equity: MoneyOut
    balances: list[MoneyOut]
    positions: list[PositionOut]
    orders: list[OrderOut]
    used_margin: MoneyOut
    available_margin: MoneyOut
    completeness: list[str]


class CoverageOut(StrictOut):
    coverage_start: datetime
    coverage_end: datetime
    complete: bool
    from_inception: bool


class AccountCoverageOut(StrictOut):
    fills: CoverageOut | None = None
    funding: CoverageOut | None = None


class AccountOut(StrictOut):
    connection_id: str
    label: str
    adapter_id: str
    environment: str
    capital_scope: Literal["simulated", "real"]
    enabled: bool
    archived: bool = False
    book_ids: list[str]
    snapshot: SnapshotOut | None
    last_success_at: datetime | None
    last_failure_at: datetime | None
    failure_reason: str | None
    coverage: AccountCoverageOut
    orders: list["StoredOrderOut"]


class AccountsOut(StrictOut):
    items: list[AccountOut]
    simulated: list[MoneyOut]
    real: list[MoneyOut]


class AttributionOut(StrictOut):
    source: str
    book_id: str | None
    decision_id: str | None
    operation_id: str | None


class StoredOrderOut(OrderOut):
    attribution: AttributionOut
    currently_open: bool


AccountOut.model_rebuild()


class FillOut(StrictOut):
    connection_id: str
    venue_fill_id: str
    venue_order_id: str
    instrument: InstrumentOut
    side: str
    amount: Decimal
    price: Decimal
    occurred_at: datetime
    fee: MoneyOut
    realized_pnl: MoneyOut
    source: Literal["platform", "local_calculation"]
    client_order_id: str | None
    attribution: AttributionOut


class FillsOut(StrictOut):
    items: list[FillOut]
    total: int
    offset: int
    limit: int


class FundingOut(StrictOut):
    connection_id: str
    venue_entry_id: str
    instrument: InstrumentOut
    amount: MoneyOut
    occurred_at: datetime


class IncomeOut(StrictOut):
    start: datetime
    end: datetime
    realized_gross: list[MoneyOut]
    fees: list[MoneyOut]
    funding: list[MoneyOut]
    unrealized: list[MoneyOut]
    unrealized_as_of: datetime | None
    net_trading: list[MoneyOut]
    completeness: list[str]
    methodology: str
    items: list[FundingOut]
    total: int
    offset: int
    limit: int


def _parts(request):
    runtime = getattr(request.app.state, "runtime", None)
    store = getattr(getattr(runtime, "repository", None), "account_store", None)
    if runtime is None or store is None:
        raise HTTPException(503, "账户账本不可用")
    return runtime, store


async def _connection(request, connection_id):
    runtime, store = _parts(request)
    document = (await runtime.repository.get_existing()).document
    connection = next((item for item in document.execution.connections if item.id == connection_id), None)
    if connection is None:
        archived = await store.archived(connection_id)
        if archived is None:
            raise HTTPException(404, "账户连接不存在")
        connection = SimpleNamespace(**archived)
    return runtime, store, document, connection


async def account_out(store, document, connection):
    snapshot = await store.latest(connection.id)
    status = await store.status(connection.id)
    status["coverage"] = {
        kind: {key: value for key, value in item.items() if key != "cursor"}
        for kind, item in status["coverage"].items()
    }
    current = {item.venue_order_id for item in snapshot.orders} if snapshot else set()
    orders = []
    book_id = effective_memberships(document).get(connection.id)
    for item in await store.orders(connection.id):
        attribution = await store.attribution(connection.id, item["venue_order_id"], item["client_order_id"], None)
        orders.append(item | {"attribution": attribution, "currently_open": item["venue_order_id"] in current})
    return AccountOut(
        connection_id=connection.id,
        label=connection.label,
        adapter_id=connection.adapter_id,
        environment=connection.environment,
        capital_scope=(
            connection.capital_scope
            if getattr(connection, "archived", False)
            else require_environment(connection.adapter_id, connection.environment).capital_scope
        ),
        enabled=connection.enabled,
        archived=getattr(connection, "archived", False),
        book_ids=[book_id] if book_id is not None else [],
        snapshot=payload(snapshot),
        orders=orders,
        **status,
    )


def _range(start, end):
    end = end or datetime.now(UTC)
    start = start or end - timedelta(days=7)
    if start.tzinfo is None or end.tzinfo is None or start > end:
        raise HTTPException(422, "请选择带时区且起止有序的时间区间")
    return start, end


async def _stored_range(store, connection_id, start, end):
    if end is None:
        snapshot = await store.latest(connection_id)
        end = snapshot.observed_at if snapshot is not None else None
    return _range(start, end)


@router.get("", response_model=AccountsOut)
async def list_accounts(request: Request):
    runtime, store = _parts(request)
    document = (await runtime.repository.get_existing()).document
    items = [await account_out(store, document, connection) for connection in document.execution.connections]

    def totals(scope):
        from cryptotrader.accounts.models import Money

        return payload(
            sum_money(
                Money(
                    item.snapshot.equity.amount, item.snapshot.equity.currency, item.snapshot.equity.unavailable_reason
                )
                if item.snapshot
                else Money(None, "UNKNOWN", "账户尚未同步")
                for item in items
                if item.capital_scope == scope
            )
        )

    return AccountsOut(items=items, simulated=totals("simulated"), real=totals("real"))


@router.get("/{connection_id}", response_model=AccountOut)
async def get_account(connection_id: str, request: Request):
    _, store, document, connection = await _connection(request, connection_id)
    return await account_out(store, document, connection)


@router.post("/{connection_id}/sync", response_model=AccountOut)
async def sync_account(connection_id: str, request: Request):
    runtime, store, document, connection = await _connection(request, connection_id)
    if getattr(connection, "archived", False):
        raise HTTPException(409, "账户已归档，仅可读取历史")  # noqa: RUF001
    service = getattr(runtime, "account_sync", None)
    if service is None:
        raise HTTPException(503, "账户同步尚未启动")
    try:
        await service.sync(connection_id)
    except AccountSyncError as error:
        raise HTTPException(503, str(error)) from None
    return await account_out(store, document, connection)


@router.get("/{connection_id}/fills", response_model=FillsOut)
async def get_fills(
    connection_id: str,
    request: Request,
    symbol: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    _, store, _, _ = await _connection(request, connection_id)
    start, end = await _stored_range(store, connection_id, start, end)
    fills = await store.history(connection_id, start=start, end=end, symbol=symbol)
    items = []
    for item in fills[offset : offset + limit]:
        attribution = await store.attribution(
            connection_id, item.venue_order_id, item.client_order_id, item.occurred_at
        )
        items.append(payload(item) | {"attribution": attribution})
    return FillsOut(items=items, total=len(fills), offset=offset, limit=limit)


@router.get("/{connection_id}/income", response_model=IncomeOut)
async def get_income(
    connection_id: str,
    request: Request,
    symbol: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    _, store, _, _ = await _connection(request, connection_id)
    start, end = await _stored_range(store, connection_id, start, end)
    summary = await IncomeService(store).summary(connection_id, start, end, symbol=symbol)
    entries = await store.history(connection_id, "funding", start=start, end=end, symbol=symbol)
    return IncomeOut(
        start=start,
        end=end,
        **payload(summary),
        items=payload(entries[offset : offset + limit]),
        total=len(entries),
        offset=offset,
        limit=limit,
    )
