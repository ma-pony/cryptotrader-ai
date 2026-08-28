"""`trading_cycles` 的 PostgreSQL/SQLite 持久化与内存实现。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import JSON, BigInteger, DateTime, String, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.journal.models import TradingCycleRecord

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleStatus

_ready: set[str] = set()


class _Base(DeclarativeBase):
    pass


class _TradingCycleRow(_Base):
    __tablename__ = "trading_cycles"

    cycle_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pair: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    profile_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )


def _payload(record: TradingCycleRecord) -> dict[str, Any]:
    return {
        "profile_snapshot": dict(record.profile_snapshot),
        "context_summary": dict(record.context_summary),
        "component_signals": [dict(item) for item in record.component_signals],
        "component_error": dict(record.component_error) if record.component_error is not None else None,
        "fused_signal": dict(record.fused_signal) if record.fused_signal is not None else None,
        "target_position": dict(record.target_position) if record.target_position is not None else None,
        "trade_plan": dict(record.trade_plan) if record.trade_plan is not None else None,
        "hitl_result": dict(record.hitl_result) if record.hitl_result is not None else None,
        "risk_result": dict(record.risk_result) if record.risk_result is not None else None,
        "execution_result": dict(record.execution_result) if record.execution_result is not None else None,
    }


def _record(row: _TradingCycleRow) -> TradingCycleRecord:
    payload = row.payload
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return TradingCycleRecord(
        cycle_id=row.cycle_id,
        created_at=created_at,
        pair=row.pair,
        status=cast("CycleStatus", row.status),
        profile_revision=row.profile_revision,
        profile_snapshot=payload["profile_snapshot"],
        context_summary=payload["context_summary"],
        component_signals=tuple(payload["component_signals"]),
        component_error=payload["component_error"],
        fused_signal=payload["fused_signal"],
        target_position=payload["target_position"],
        trade_plan=payload["trade_plan"],
        hitl_result=payload["hitl_result"],
        risk_result=payload["risk_result"],
        execution_result=payload["execution_result"],
    )


class CycleJournalStore:
    """追加并查询完整交易周期。无数据库时使用实例级内存。"""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.records: list[TradingCycleRecord] = []

    async def ensure_table(self) -> None:
        if self.database_url is None or self.database_url in _ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
        _ready.add(self.database_url)

    async def append(self, record: TradingCycleRecord) -> None:
        if self.database_url is None:
            if any(item.cycle_id == record.cycle_id for item in self.records):
                raise ValueError(f"cycle {record.cycle_id!r} already exists")
            self.records.append(record)
            return

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            session.add(
                _TradingCycleRow(
                    cycle_id=record.cycle_id,
                    created_at=record.created_at,
                    pair=record.pair,
                    status=record.status,
                    profile_revision=record.profile_revision,
                    payload=_payload(record),
                )
            )
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise ValueError(f"cycle {record.cycle_id!r} already exists") from exc
        finally:
            await session.close()

    async def get(self, cycle_id: str) -> TradingCycleRecord | None:
        if self.database_url is None:
            return next((item for item in self.records if item.cycle_id == cycle_id), None)

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_TradingCycleRow, cycle_id)
            return _record(row) if row is not None else None
        finally:
            await session.close()

    async def replace(self, record: TradingCycleRecord) -> None:
        """替换同一周期的暂停状态。供 HITL 终态迁移使用。"""
        if self.database_url is None:
            for index, current in enumerate(self.records):
                if current.cycle_id == record.cycle_id:
                    self.records[index] = record
                    return
            raise LookupError(f"cycle {record.cycle_id!r} does not exist")

        await self.ensure_table()
        statement = (
            update(_TradingCycleRow)
            .where(_TradingCycleRow.cycle_id == record.cycle_id)
            .values(
                created_at=record.created_at,
                pair=record.pair,
                status=record.status,
                profile_revision=record.profile_revision,
                payload=_payload(record),
            )
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount != 1:
                await session.rollback()
                raise LookupError(f"cycle {record.cycle_id!r} does not exist")
            await session.commit()
        finally:
            await session.close()

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        pair: str | None = None,
        status: CycleStatus | None = None,
    ) -> list[TradingCycleRecord]:
        if limit < 1 or offset < 0:
            return []
        if self.database_url is None:
            records = self.records
            if pair is not None:
                records = [item for item in records if item.pair == pair]
            if status is not None:
                records = [item for item in records if item.status == status]
            ordered = sorted(records, key=lambda item: item.created_at, reverse=True)
            return ordered[offset : offset + limit]

        await self.ensure_table()
        query = select(_TradingCycleRow)
        if pair is not None:
            query = query.where(_TradingCycleRow.pair == pair)
        if status is not None:
            query = query.where(_TradingCycleRow.status == status)
        query = query.order_by(_TradingCycleRow.created_at.desc()).offset(offset).limit(limit)
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(query)).scalars().all()
            return [_record(row) for row in rows]
        finally:
            await session.close()

    async def count(
        self,
        *,
        pair: str | None = None,
        status: CycleStatus | None = None,
    ) -> int:
        if self.database_url is None:
            records = self.records
            if pair is not None:
                records = [item for item in records if item.pair == pair]
            if status is not None:
                records = [item for item in records if item.status == status]
            return len(records)

        await self.ensure_table()
        query = select(func.count()).select_from(_TradingCycleRow)
        if pair is not None:
            query = query.where(_TradingCycleRow.pair == pair)
        if status is not None:
            query = query.where(_TradingCycleRow.status == status)
        session = await get_async_session(self.database_url)
        try:
            return int((await session.execute(query)).scalar_one())
        finally:
            await session.close()
