"""目标仓位计划的人工审批快照与原子状态迁移。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from sqlalchemy import JSON, BigInteger, DateTime, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.cycle_serialization import (
    cycle_request_from_payload,
    cycle_request_payload,
    signal_context_from_payload,
    signal_context_payload,
    signal_profile_from_payload,
    signal_profile_payload,
    trade_plan_from_payload,
    trade_plan_payload,
)
from cryptotrader.db import get_async_session, get_engine

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleRequest, TradePlan
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.models import SignalContext

ApprovalStatus = Literal["pending", "approved", "rejected"]
_ready: set[str] = set()


class ApprovalStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    cycle_id: str
    pair: str
    profile_revision: int
    profile: SignalProfile
    cycle_request: CycleRequest
    signal_context: SignalContext
    plan: TradePlan
    status: ApprovalStatus
    decision_by: str | None
    created_at: datetime
    decided_at: datetime | None


class _Base(DeclarativeBase):
    pass


class _ApprovalRow(_Base):
    __tablename__ = "trade_plan_approvals"

    approval_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cycle_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    pair: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    profile_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    request_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    context_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    trade_plan_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    decision_by: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _normalize_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _row_to_record(row: _ApprovalRow) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=row.approval_id,
        cycle_id=row.cycle_id,
        pair=row.pair,
        profile_revision=row.profile_revision,
        profile=signal_profile_from_payload(row.request_payload["profile"]),
        cycle_request=cycle_request_from_payload(row.request_payload),
        signal_context=signal_context_from_payload(row.context_payload),
        plan=trade_plan_from_payload(row.trade_plan_payload),
        status=cast("ApprovalStatus", row.status),
        decision_by=row.decision_by,
        created_at=_normalize_datetime(row.created_at),
        decided_at=_normalize_datetime(row.decided_at) if row.decided_at is not None else None,
    )


class ApprovalStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.records: list[ApprovalRecord] = []
        self._lock = asyncio.Lock()

    async def ensure_table(self) -> None:
        if self.database_url is None or self.database_url in _ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
        _ready.add(self.database_url)

    async def create(
        self,
        *,
        cycle_id: str,
        cycle_request: CycleRequest,
        profile: SignalProfile,
        signal_context: SignalContext,
        plan: TradePlan,
        approval_id: str | None = None,
        created_at: datetime | None = None,
    ) -> ApprovalRecord:
        approval_id = approval_id or str(uuid4())
        created_at = created_at or datetime.now(UTC)
        request_data = cycle_request_payload(cycle_request)
        request_data["profile"] = signal_profile_payload(profile)
        context_data = signal_context_payload(signal_context)
        plan_data = trade_plan_payload(plan)
        record = ApprovalRecord(
            approval_id=approval_id,
            cycle_id=cycle_id,
            pair=cycle_request.pair.canonical(),
            profile_revision=profile.revision,
            profile=signal_profile_from_payload(request_data["profile"]),
            cycle_request=cycle_request_from_payload(request_data),
            signal_context=signal_context_from_payload(context_data),
            plan=trade_plan_from_payload(plan_data),
            status="pending",
            decision_by=None,
            created_at=created_at,
            decided_at=None,
        )
        if self.database_url is None:
            async with self._lock:
                if any(item.approval_id == approval_id for item in self.records):
                    raise ValueError(f"approval {approval_id!r} already exists")
                self.records.append(record)
            return record

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            session.add(
                _ApprovalRow(
                    approval_id=record.approval_id,
                    cycle_id=record.cycle_id,
                    pair=record.pair,
                    profile_revision=record.profile_revision,
                    request_payload=request_data,
                    context_payload=context_data,
                    trade_plan_payload=plan_data,
                    status=record.status,
                    decision_by=None,
                    created_at=record.created_at,
                    decided_at=None,
                )
            )
            await session.commit()
            return record
        finally:
            await session.close()

    async def get(self, approval_id: str) -> ApprovalRecord | None:
        if self.database_url is None:
            return next((item for item in self.records if item.approval_id == approval_id), None)
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_ApprovalRow, approval_id)
            return _row_to_record(row) if row is not None else None
        finally:
            await session.close()

    async def list_pending(self) -> list[ApprovalRecord]:
        if self.database_url is None:
            return sorted(
                (item for item in self.records if item.status == "pending"),
                key=lambda item: item.created_at,
                reverse=True,
            )
        await self.ensure_table()
        statement = (
            select(_ApprovalRow).where(_ApprovalRow.status == "pending").order_by(_ApprovalRow.created_at.desc())
        )
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(statement)).scalars().all()
            return [_row_to_record(row) for row in rows]
        finally:
            await session.close()

    async def approve(self, approval_id: str, *, decision_by: str) -> ApprovalRecord:
        return await self._decide(approval_id, "approved", decision_by)

    async def reject(self, approval_id: str, *, decision_by: str) -> ApprovalRecord:
        return await self._decide(approval_id, "rejected", decision_by)

    async def _decide(
        self,
        approval_id: str,
        status: Literal["approved", "rejected"],
        decision_by: str,
    ) -> ApprovalRecord:
        decided_at = datetime.now(UTC)
        if self.database_url is None:
            async with self._lock:
                for index, record in enumerate(self.records):
                    if record.approval_id != approval_id:
                        continue
                    if record.status != "pending":
                        raise ApprovalStateError(f"approval {approval_id!r} is not pending")
                    decided = replace(
                        record,
                        status=status,
                        decision_by=decision_by,
                        decided_at=decided_at,
                    )
                    self.records[index] = decided
                    return decided
            raise LookupError(f"approval {approval_id!r} does not exist")

        await self.ensure_table()
        statement = (
            update(_ApprovalRow)
            .where(
                _ApprovalRow.approval_id == approval_id,
                _ApprovalRow.status == "pending",
            )
            .values(status=status, decision_by=decision_by, decided_at=decided_at)
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount != 1:
                await session.rollback()
                existing = await self.get(approval_id)
                if existing is None:
                    raise LookupError(f"approval {approval_id!r} does not exist")
                raise ApprovalStateError(f"approval {approval_id!r} is not pending")
            await session.commit()
        finally:
            await session.close()
        decided = await self.get(approval_id)
        if decided is None:
            raise LookupError(f"approval {approval_id!r} does not exist")
        return decided
