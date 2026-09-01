"""Durable account facts; checkpoints and their batches commit together."""
# ruff: noqa: RUF001 -- User-facing Chinese messages retain Chinese punctuation.

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import JSON, DateTime, Integer, String, func, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.accounts.models import (
    AccountOrder,
    AccountPosition,
    AccountSnapshot,
    Fill,
    FundingEntry,
    Instrument,
    Money,
)
from cryptotrader.db import get_async_session
from cryptotrader.migrations.schema import require_tables
from cryptotrader.pair import Pair


class Base(DeclarativeBase):
    pass


class SnapshotRow(Base):
    __tablename__ = "account_snapshots"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class SyncStatusRow(Base):
    __tablename__ = "account_sync_status"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(String(256))


class FillRow(Base):
    __tablename__ = "account_fills"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    venue_fill_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class FundingRow(Base):
    __tablename__ = "account_funding"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    venue_entry_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict] = mapped_column(JSON)


class OrderRow(Base):
    __tablename__ = "account_orders"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    venue_order_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class OrderBindingRow(Base):
    __tablename__ = "account_order_bindings"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    binding_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    client_order_id: Mapped[str | None] = mapped_column(String(255), index=True)
    venue_order_id: Mapped[str | None] = mapped_column(String(255), index=True)
    book_id: Mapped[str | None] = mapped_column(String(255))
    decision_id: Mapped[str | None] = mapped_column(String(255))
    operation_id: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CursorRow(Base):
    __tablename__ = "account_sync_cursors"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class MembershipRow(Base):
    __tablename__ = "book_memberships"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    book_id: Mapped[str] = mapped_column(String(255))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PaperStateRow(Base):
    __tablename__ = "paper_account_states"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class AccountOperationRow(Base):
    __tablename__ = "account_operations"
    operation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)


class ArchivedAccountRow(Base):
    __tablename__ = "archived_accounts"
    connection_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class OperationConflictError(ValueError):
    pass


class AccountOperationStore:
    """Independent operations, sharing only the account database/session foundation."""

    def __init__(self, database_url):
        if not database_url:
            raise ValueError("operations require an explicit database URL")
        self.database_url = database_url

    async def create(self, operation):
        async with await get_async_session(self.database_url) as session, session.begin():
            session.add(
                AccountOperationRow(
                    operation_id=operation.operation_id,
                    status=operation.status,
                    version=0,
                    payload=operation.model_dump(mode="json"),
                )
            )
        return operation

    async def get(self, operation_id):
        from cryptotrader.accounts.models import AccountOperationOut

        async with await get_async_session(self.database_url) as session:
            row = await session.get(AccountOperationRow, operation_id)
            if row is None:
                raise LookupError("人工操作不存在")
            return AccountOperationOut.model_validate(row.payload)

    async def update(self, operation, *, expected_status, expected_version=None):
        operation = operation.model_copy(update={"updated_at": datetime.now(UTC)})
        query = update(AccountOperationRow).where(
            AccountOperationRow.operation_id == operation.operation_id,
            AccountOperationRow.status == expected_status,
        )
        if expected_version is not None:
            query = query.where(AccountOperationRow.version == expected_version)
        async with await get_async_session(self.database_url) as session, session.begin():
            result = await session.execute(
                query.values(
                    status=operation.status,
                    version=operation.plan.version if operation.plan else 0,
                    payload=operation.model_dump(mode="json"),
                )
            )
            if result.rowcount != 1:
                raise OperationConflictError("操作状态或计划版本已变化，请重新读取")
        return operation

    async def invalidate(self, operation_id: str, reason: str):
        operation = await self.get(operation_id)
        if operation.status not in {"preparing", "awaiting_confirmation", "executing"}:
            raise OperationConflictError("操作已结束")
        result = operation.result.model_copy(update={"failure_reason": reason})
        return await self.update(
            operation.model_copy(update={"status": "invalidated", "result": result}), expected_status=operation.status
        )

    async def recover_interrupted(self):
        """Never resubmit orders on process restart; retain all already persisted receipts."""
        async with await get_async_session(self.database_url) as session:
            ids = (
                await session.scalars(
                    select(AccountOperationRow.operation_id).where(
                        AccountOperationRow.status.in_(("preparing", "executing"))
                    )
                )
            ).all()
        for operation_id in ids:
            operation = await self.get(operation_id)
            result = operation.result.model_copy(
                update={
                    "failure_reason": "服务中断，请刷新账户核对实际结果；不会自动重试下单",
                    "reconciliation_required": True,
                }
            )
            await self.update(
                operation.model_copy(update={"status": "failed", "result": result}), expected_status=operation.status
            )

    async def list_all(self):
        from cryptotrader.accounts.models import AccountOperationOut

        async with await get_async_session(self.database_url) as session:
            return [
                AccountOperationOut.model_validate(row.payload)
                for row in (await session.scalars(select(AccountOperationRow))).all()
            ]


def utc(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def payload(value):
    """JSON-safe facts, without binary serialization or executable payloads."""
    if isinstance(value, (datetime, Decimal)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, Pair):
        return value.canonical()
    if hasattr(value, "__dataclass_fields__"):
        return {key: payload(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [payload(item) for item in value]
    return value


def money(value):
    return Money(
        Decimal(value["amount"]) if value["amount"] is not None else None,
        value["currency"],
        value["unavailable_reason"],
    )


def instrument(value):
    return Instrument(**(value | {"pair": Pair.parse(value["pair"]) if value["pair"] else None}))


def fill_from_payload(value):
    return Fill(
        **(
            value
            | {
                "instrument": instrument(value["instrument"]),
                "amount": Decimal(value["amount"]),
                "price": Decimal(value["price"]),
                "occurred_at": datetime.fromisoformat(value["occurred_at"]),
                "fee": money(value["fee"]),
                "realized_pnl": money(value["realized_pnl"]),
            }
        )
    )


def funding_from_payload(value):
    return FundingEntry(
        **(
            value
            | {
                "instrument": instrument(value["instrument"]),
                "amount": money(value["amount"]),
                "occurred_at": datetime.fromisoformat(value["occurred_at"]),
            }
        )
    )


def snapshot_from_payload(value):
    positions = tuple(
        AccountPosition(
            **(
                item
                | {
                    "instrument": instrument(item["instrument"]),
                    "signed_amount": Decimal(item["signed_amount"]),
                    "available_amount": Decimal(item["available_amount"])
                    if item["available_amount"] is not None
                    else None,
                    "entry_price": Decimal(item["entry_price"]) if item["entry_price"] is not None else None,
                    "signed_notional": money(item["signed_notional"]),
                    "unrealized_pnl": money(item["unrealized_pnl"]),
                }
            )
        )
        for item in value["positions"]
    )
    orders = tuple(
        AccountOrder(
            **(
                item
                | {
                    "instrument": instrument(item["instrument"]),
                    "amount": Decimal(item["amount"]),
                    "filled_amount": Decimal(item["filled_amount"]),
                    "average_price": Decimal(item["average_price"]) if item["average_price"] is not None else None,
                    "observed_at": datetime.fromisoformat(item["observed_at"]),
                    "remaining_notional": money(item["remaining_notional"]),
                }
            )
        )
        for item in value["orders"]
    )
    return AccountSnapshot(
        **(
            value
            | {
                "observed_at": datetime.fromisoformat(value["observed_at"]),
                "positions": positions,
                "orders": orders,
                "equity": money(value["equity"]),
                "balances": tuple(money(item) for item in value["balances"]),
                "used_margin": money(value["used_margin"]),
                "available_margin": money(value["available_margin"]),
            }
        )
    )


def effective_memberships(document):
    """Resolve only proven assignments; pausing execution does not remove membership."""
    active, candidates = {}, {}
    for book in document.execution.books:
        for allocation in book.allocations:
            if allocation.enabled:
                candidates.setdefault(allocation.connection_id, set()).add(book.id)
                if book.enabled:
                    active[allocation.connection_id] = book.id
    return {
        connection_id: active.get(connection_id, next(iter(books)))
        for connection_id, books in candidates.items()
        if connection_id in active or len(books) == 1
    }


async def update_memberships(session, document, changed_at):
    """Called only inside the successful configuration CAS transaction."""
    desired = effective_memberships(document)
    current = (await session.scalars(select(MembershipRow).where(MembershipRow.valid_to.is_(None)))).all()
    retained = set()
    for row in current:
        if desired.get(row.connection_id) == row.book_id:
            retained.add(row.connection_id)
        else:
            row.valid_to = changed_at
    for connection_id, book_id in desired.items():
        if connection_id not in retained:
            session.add(MembershipRow(connection_id=connection_id, book_id=book_id, valid_from=changed_at))


class AccountStore:
    def __init__(self, database_url: str, *, clock=None):
        if not database_url:
            raise ValueError("account ledger requires an explicit database URL")
        self.database_url = database_url
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ready = False
        # Owned by this database context; shared by Paper adapters in one runtime.
        self.paper_accounts = {}
        self.paper_locks = {}

    async def ensure_tables(self):
        if not self._ready:
            await require_tables(self.database_url, Base.metadata.tables)
            self._ready = True

    async def latest(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            row = await session.scalar(
                select(SnapshotRow)
                .where(SnapshotRow.connection_id == connection_id)
                .order_by(SnapshotRow.observed_at.desc())
                .limit(1)
            )
            return snapshot_from_payload(row.payload) if row else None

    async def archived(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            row = await session.get(ArchivedAccountRow, connection_id)
            return row.payload if row else None

    async def status(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            row = await session.get(SyncStatusRow, connection_id)
            cursors = (await session.scalars(select(CursorRow).where(CursorRow.connection_id == connection_id))).all()
            return {
                "last_success_at": utc(row.last_success_at) if row and row.last_success_at else None,
                "last_failure_at": utc(row.last_failure_at) if row and row.last_failure_at else None,
                "failure_reason": row.failure_reason if row else None,
                "coverage": {item.kind: item.payload for item in cursors},
            }

    async def sync_states(self):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            return [
                {
                    "connection_id": row.connection_id,
                    "last_success_at": utc(row.last_success_at) if row.last_success_at else None,
                    "last_failure_at": utc(row.last_failure_at) if row.last_failure_at else None,
                    "failure_reason": row.failure_reason,
                }
                for row in (await session.scalars(select(SyncStatusRow))).all()
            ]

    async def failed(self, connection_id, reason):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session, session.begin():
            row = await session.get(SyncStatusRow, connection_id)
            if row is None:
                row = SyncStatusRow(connection_id=connection_id)
                session.add(row)
            row.last_failure_at = self._clock()
            row.failure_reason = reason

    async def ingest(self, snapshot, fills=(), funding=(), *, checkpoints=None):
        await self.ensure_tables()
        connection_id = snapshot.connection_id
        if any(item.connection_id != connection_id for item in (*fills, *funding, *snapshot.orders)):
            raise ValueError("account batch connection identity mismatch")
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.merge(
                SnapshotRow(connection_id=connection_id, observed_at=snapshot.observed_at, payload=payload(snapshot))
            )
            for fill in fills:
                if await session.get(FillRow, (connection_id, fill.venue_fill_id)) is None:
                    session.add(
                        FillRow(
                            connection_id=connection_id,
                            venue_fill_id=fill.venue_fill_id,
                            occurred_at=fill.occurred_at,
                            payload=payload(fill),
                        )
                    )
            for entry in funding:
                if await session.get(FundingRow, (connection_id, entry.venue_entry_id)) is None:
                    session.add(
                        FundingRow(
                            connection_id=connection_id,
                            venue_entry_id=entry.venue_entry_id,
                            occurred_at=entry.occurred_at,
                            payload=payload(entry),
                        )
                    )
            await self._store_orders(session, snapshot)
            await session.flush()
            await self._reconcile_filled_orders(session, connection_id)
            for kind, checkpoint in (checkpoints or {}).items():
                await session.merge(CursorRow(connection_id=connection_id, kind=kind, payload=payload(checkpoint)))
            row = await session.get(SyncStatusRow, connection_id)
            if row is None:
                row = SyncStatusRow(connection_id=connection_id)
                session.add(row)
            row.last_success_at = self._clock()
            row.failure_reason = None

    async def _store_orders(self, session, snapshot):
        for order in snapshot.orders:
            row = await session.get(OrderRow, (snapshot.connection_id, order.venue_order_id))
            if row is None:
                session.add(
                    OrderRow(
                        connection_id=snapshot.connection_id,
                        venue_order_id=order.venue_order_id,
                        payload=payload(order),
                    )
                )
            elif row.payload["status"] not in {"filled", "canceled", "closed", "rejected"}:
                row.payload = payload(order)

    async def _reconcile_filled_orders(self, session, connection_id):
        # Only the sum of durable, deduplicated actual fills proves a complete fill.
        fills = (await session.scalars(select(FillRow).where(FillRow.connection_id == connection_id))).all()
        totals, timestamps = {}, {}
        for row in fills:
            order_id = row.payload["venue_order_id"]
            totals[order_id] = totals.get(order_id, Decimal("0")) + Decimal(row.payload["amount"])
            timestamps[order_id] = max(timestamps.get(order_id, ""), row.payload["occurred_at"])
        orders = (await session.scalars(select(OrderRow).where(OrderRow.connection_id == connection_id))).all()
        for row in orders:
            amount = totals.get(row.venue_order_id, Decimal("0"))
            if amount < Decimal(row.payload["amount"]) or amount == 0:
                continue
            row.payload = {
                **row.payload,
                "status": "filled",
                "filled_amount": str(amount),
                "observed_at": max(row.payload["observed_at"], timestamps[row.venue_order_id]),
            }

    async def fill_count(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            return await session.scalar(
                select(func.count()).select_from(FillRow).where(FillRow.connection_id == connection_id)
            )

    async def record_order(self, connection_id, order):
        await self.ensure_tables()
        fact = AccountOrder(
            connection_id,
            order.id,
            Instrument(str(order.pair), order.pair, order.pair.market_type, True),
            order.side,
            order.order_type,
            order.amount,
            order.filled_amount,
            order.average_price,
            order.status,
            order.reduce_only,
            False,
            order.client_order_id,
            self._clock(),
            Money(Decimal("0"), order.pair.settle or order.pair.quote)
            if order.filled_amount == order.amount
            else Money(None, order.pair.settle or order.pair.quote, "remaining_order_valuation_unavailable"),
        )
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.merge(OrderRow(connection_id=connection_id, venue_order_id=order.id, payload=payload(fact)))

    async def orders(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            rows = (await session.scalars(select(OrderRow).where(OrderRow.connection_id == connection_id))).all()
            return [row.payload for row in rows]

    async def history(self, connection_id, kind="fills", *, start=None, end=None, symbol=None):
        await self.ensure_tables()
        table, decode = (FillRow, fill_from_payload) if kind == "fills" else (FundingRow, funding_from_payload)
        query = select(table).where(table.connection_id == connection_id).order_by(table.occurred_at)
        if start is not None:
            query = query.where(table.occurred_at >= start)
        if end is not None:
            query = query.where(table.occurred_at <= end)
        async with await get_async_session(self.database_url) as session:
            items = [decode(row.payload) for row in (await session.scalars(query)).all()]
        return [
            item
            for item in items
            if symbol is None
            or symbol in (item.instrument.venue_symbol, str(item.instrument.pair) if item.instrument.pair else None)
        ]

    async def bind_order(
        self,
        connection_id,
        client_order_id,
        *,
        book_id,
        decision_id=None,
        operation_id=None,
        venue_order_id=None,
        source="strategy",
    ):
        if not client_order_id and not venue_order_id:
            raise ValueError("an actual client or venue order ID is required")
        binding_key = f"client:{client_order_id}" if client_order_id else f"order:{venue_order_id}"
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session, session.begin():
            row = await session.get(OrderBindingRow, (connection_id, binding_key))
            if row is None:
                row = OrderBindingRow(
                    connection_id=connection_id,
                    binding_key=binding_key,
                    client_order_id=client_order_id,
                    book_id=book_id,
                    decision_id=decision_id,
                    operation_id=operation_id,
                    source=source,
                    created_at=self._clock(),
                )
                session.add(row)
            if venue_order_id is not None:
                row.venue_order_id = venue_order_id

    async def attribution(self, connection_id, order_id, client_id, occurred_at):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            query = select(OrderBindingRow).where(OrderBindingRow.connection_id == connection_id)
            from sqlalchemy import or_

            row = await session.scalar(
                query.where(
                    or_(
                        OrderBindingRow.venue_order_id == order_id,
                        OrderBindingRow.client_order_id == client_id if client_id else False,
                    )
                )
            )
            if row is not None:
                return {
                    "source": row.source,
                    "book_id": row.book_id,
                    "decision_id": row.decision_id,
                    "operation_id": row.operation_id,
                }
            if occurred_at is None:
                return {"source": "external", "book_id": None, "decision_id": None, "operation_id": None}
            memberships = (
                await session.scalars(
                    select(MembershipRow).where(
                        MembershipRow.connection_id == connection_id,
                        MembershipRow.valid_from <= occurred_at,
                        or_(MembershipRow.valid_to.is_(None), MembershipRow.valid_to > occurred_at),
                    )
                )
            ).all()
            return {
                "source": "external",
                "book_id": memberships[0].book_id if len(memberships) == 1 else None,
                "decision_id": None,
                "operation_id": None,
            }

    async def load_paper(self, connection_id):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session:
            row = await session.get(PaperStateRow, connection_id)
            return row.payload if row else None

    async def save_paper(self, connection_id, state):
        await self.ensure_tables()
        async with await get_async_session(self.database_url) as session, session.begin():
            await session.merge(PaperStateRow(connection_id=connection_id, payload=state))
