"""目标仓位计划的人工审批快照与原子状态迁移。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from sqlalchemy import JSON, BigInteger, DateTime, String, and_, case, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
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
from cryptotrader.execution.codec import book_execution_proposal_from_payload, book_execution_proposal_payload
from cryptotrader.hitl.models import BookApproval, BookApprovalStatus

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleRequest, TradePlan
    from cryptotrader.execution.models import BookExecutionProposal
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.signals.models import SignalContext

ApprovalStatus = Literal["pending", "approved", "rejected", "cancelled"]
_ready: set[str] = set()
_book_ready: set[str] = set()


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


class _BookBase(DeclarativeBase):
    pass


class _BookApprovalRow(_BookBase):
    __tablename__ = "book_approvals"

    approval_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    cycle_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    book_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    config_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    proposal_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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

    async def cancel_pending(self, approval_id: str) -> ApprovalRecord | None:
        decided_at = datetime.now(UTC)
        if self.database_url is None:
            async with self._lock:
                for index, record in enumerate(self.records):
                    if record.approval_id != approval_id:
                        continue
                    if record.status != "pending":
                        return record
                    cancelled = replace(record, status="cancelled", decided_at=decided_at)
                    self.records[index] = cancelled
                    return cancelled
            return None

        await self.ensure_table()
        statement = (
            update(_ApprovalRow)
            .where(
                _ApprovalRow.approval_id == approval_id,
                _ApprovalRow.status == "pending",
            )
            .values(status="cancelled", decided_at=decided_at)
        )
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(statement)
            if result.rowcount == 1:
                await session.commit()
            else:
                await session.rollback()
        finally:
            await session.close()
        return await self.get(approval_id)

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


class ApprovalNotFound(LookupError):  # noqa: N818 - 公共契约名称由任务规范锁定
    """审批不存在。"""


class ApprovalNotApproved(ApprovalStateError):  # noqa: N818 - 公共契约名称由任务规范锁定
    """审批尚未批准。"""


class ApprovalRejected(ApprovalStateError):  # noqa: N818 - 公共契约名称由任务规范锁定
    """审批已经拒绝。"""


class ApprovalAlreadyClaimed(ApprovalStateError):  # noqa: N818 - 公共契约名称由任务规范锁定
    """审批已经被执行者领取。"""


class ApprovalInvalidated(ApprovalStateError):  # noqa: N818 - 公共契约名称由任务规范锁定
    """审批因配置 revision 变化而失效。"""


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


_APPROVAL_ENVELOPE_VERSION = 1
_APPROVAL_ENVELOPE_KEYS = {
    "version",
    "approval_id",
    "cycle_id",
    "book_id",
    "config_revision",
    "proposal",
}


def _approval_envelope(record: BookApproval) -> dict[str, Any]:
    return {
        "version": _APPROVAL_ENVELOPE_VERSION,
        "approval_id": record.approval_id,
        "cycle_id": record.cycle_id,
        "book_id": record.book_id,
        "config_revision": record.config_revision,
        "proposal": book_execution_proposal_payload(record.proposal),
    }


def _decode_book_record(row: _BookApprovalRow) -> BookApproval | None:
    try:
        envelope = row.proposal_json
        if type(envelope) is not dict or set(envelope) != _APPROVAL_ENVELOPE_KEYS:
            raise ValueError("invalid approval envelope")
        if type(envelope["version"]) is not int or envelope["version"] != _APPROVAL_ENVELOPE_VERSION:
            raise ValueError("unsupported approval envelope version")
        if (
            envelope["approval_id"] != row.approval_id
            or envelope["cycle_id"] != row.cycle_id
            or envelope["book_id"] != row.book_id
            or envelope["config_revision"] != row.config_revision
        ):
            raise ValueError("approval envelope identity mismatch")
        proposal = book_execution_proposal_from_payload(envelope["proposal"])
        return BookApproval(
            row.approval_id,
            row.cycle_id,
            row.book_id,
            row.config_revision,
            proposal,
            cast("BookApprovalStatus", row.status),
            _utc(row.created_at),
            _utc(row.decided_at) if row.decided_at is not None else None,
            _utc(row.claimed_at) if row.claimed_at is not None else None,
        )
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return None


def _book_record(row: _BookApprovalRow) -> BookApproval:
    record = _decode_book_record(row)
    if record is None:
        raise ValueError("stored approval payload is invalid")
    return record


class BookApprovalStore:
    """只访问 ``book_approvals`` 的 revision-bound 审批存储。"""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.records: list[BookApproval] = []
        self._lock = asyncio.Lock()

    async def ensure_table(self) -> None:
        if self.database_url is None or self.database_url in _book_ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_BookBase.metadata.create_all)
        _book_ready.add(self.database_url)

    async def create(
        self,
        proposal: BookExecutionProposal,
        *,
        cycle_id: str | None = None,
        approval_id: str | None = None,
        created_at: datetime | None = None,
    ) -> BookApproval:
        record = BookApproval(
            approval_id or str(uuid4()),
            cycle_id or str(uuid4()),
            proposal.book_id,
            proposal.config_revision,
            proposal,
            "pending",
            created_at or datetime.now(UTC),
            None,
            None,
        )
        proposal_json = _approval_envelope(record)
        if self.database_url is None:
            async with self._lock:
                if any(item.approval_id == record.approval_id for item in self.records):
                    raise ValueError("approval already exists")
                self.records.append(record)
            return record

        await self.ensure_table()
        session = await get_async_session(self.database_url)
        duplicate = False
        try:
            session.add(
                _BookApprovalRow(
                    approval_id=record.approval_id,
                    cycle_id=record.cycle_id,
                    book_id=record.book_id,
                    config_revision=record.config_revision,
                    proposal_json=proposal_json,
                    status=record.status,
                    created_at=record.created_at,
                    decided_at=None,
                    claimed_at=None,
                )
            )
            await session.commit()
        except IntegrityError:
            await session.rollback()
            duplicate = True
        finally:
            await session.close()
        if duplicate:
            raise ValueError("approval already exists")
        return record

    async def get(self, approval_id: str) -> BookApproval | None:
        if self.database_url is None:
            return next((item for item in self.records if item.approval_id == approval_id), None)
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_BookApprovalRow, approval_id)
        finally:
            await session.close()
        return _book_record(row) if row is not None else None

    async def list_pending(self) -> list[BookApproval]:
        if self.database_url is None:
            return sorted(
                (item for item in self.records if item.status == "pending"),
                key=lambda item: item.created_at,
                reverse=True,
            )
        await self.ensure_table()
        query = (
            select(_BookApprovalRow)
            .where(_BookApprovalRow.status == "pending")
            .order_by(_BookApprovalRow.created_at.desc())
        )
        session = await get_async_session(self.database_url)
        try:
            rows = (await session.execute(query)).scalars().all()
        finally:
            await session.close()
        return [_book_record(row) for row in rows]

    async def approve(self, approval_id: str) -> BookApproval:
        return await self._decide_book(approval_id, "approved")

    async def reject(self, approval_id: str) -> BookApproval:
        return await self._decide_book(approval_id, "rejected")

    async def _decide_book(
        self,
        approval_id: str,
        status: Literal["approved", "rejected"],
    ) -> BookApproval:
        decided_at = datetime.now(UTC)
        if self.database_url is None:
            async with self._lock:
                for index, record in enumerate(self.records):
                    if record.approval_id != approval_id:
                        continue
                    if record.status != "pending":
                        raise ApprovalStateError("approval is not pending")
                    decided = replace(record, status=status, decided_at=decided_at)
                    self.records[index] = decided
                    return decided
            raise ApprovalNotFound("approval was not found")

        await self.ensure_table()
        statement = (
            update(_BookApprovalRow)
            .where(_BookApprovalRow.approval_id == approval_id, _BookApprovalRow.status == "pending")
            .values(status=status, decided_at=decided_at)
            .returning(_BookApprovalRow)
        )
        session = await get_async_session(self.database_url)
        transitioned: BookApproval | None = None
        invalid_payload = False
        try:
            result = await session.execute(statement)
            row = result.scalar_one_or_none()
            if row is not None:
                transitioned = _decode_book_record(row)
            if transitioned is not None:
                await session.commit()
            else:
                await session.rollback()
                invalid_payload = row is not None
        finally:
            await session.close()
        if invalid_payload:
            raise ValueError("stored approval payload is invalid")
        if transitioned is not None:
            return transitioned
        record = await self.get(approval_id)
        if record is None:
            raise ApprovalNotFound("approval was not found")
        raise ApprovalStateError("approval is not pending")

    async def claim_for_execution(
        self,
        approval_id: str,
        *,
        current_revision: int,
    ) -> BookExecutionProposal:
        if type(current_revision) is not int or current_revision < 0:
            raise ValueError("current_revision must be a non-negative integer")
        if self.database_url is None:
            return await self._claim_memory(approval_id, current_revision)
        return await self._claim_database(approval_id, current_revision)

    async def _claim_memory(self, approval_id: str, current_revision: int) -> BookExecutionProposal:
        now = datetime.now(UTC)
        async with self._lock:
            for index, record in enumerate(self.records):
                if record.approval_id != approval_id:
                    continue
                if record.status in {"pending", "approved"} and record.config_revision != current_revision:
                    self.records[index] = replace(record, status="invalidated", decided_at=now)
                    raise ApprovalInvalidated("approval revision is invalid")
                if record.status == "approved":
                    claimed = replace(record, status="executed", claimed_at=now)
                    self.records[index] = claimed
                    return claimed.proposal
                self._raise_claim_state(record)
        raise ApprovalNotFound("approval was not found")

    async def _claim_database(self, approval_id: str, current_revision: int) -> BookExecutionProposal:
        await self.ensure_table()
        now = datetime.now(UTC)
        revision_changed = _BookApprovalRow.config_revision != current_revision
        approved_at_current_revision = and_(
            _BookApprovalRow.status == "approved",
            _BookApprovalRow.config_revision == current_revision,
            _BookApprovalRow.claimed_at.is_(None),
        )
        transition = (
            update(_BookApprovalRow)
            .where(
                _BookApprovalRow.approval_id == approval_id,
                _BookApprovalRow.status.in_(("pending", "approved")),
                or_(revision_changed, approved_at_current_revision),
            )
            .values(
                status=case((revision_changed, "invalidated"), else_="executed"),
                decided_at=case((revision_changed, now), else_=_BookApprovalRow.decided_at),
                claimed_at=case((revision_changed, _BookApprovalRow.claimed_at), else_=now),
            )
            .returning(_BookApprovalRow)
        )
        session = await get_async_session(self.database_url)
        transitioned: BookApproval | None = None
        invalid_payload = False
        try:
            result = await session.execute(transition)
            row = result.scalar_one_or_none()
            if row is not None:
                transitioned = _decode_book_record(row)
            if transitioned is not None:
                await session.commit()
            else:
                await session.rollback()
                invalid_payload = row is not None
        finally:
            await session.close()
        if invalid_payload:
            raise ValueError("stored approval payload is invalid")
        if transitioned is not None:
            if transitioned.status == "invalidated":
                raise ApprovalInvalidated("approval revision is invalid")
            return transitioned.proposal
        record = await self.get(approval_id)
        if record is None:
            raise ApprovalNotFound("approval was not found")
        self._raise_claim_state(record)
        raise AssertionError("unreachable claim state")

    @staticmethod
    def _raise_claim_state(record: BookApproval) -> None:
        if record.status == "rejected":
            raise ApprovalRejected("approval was rejected")
        if record.status == "invalidated":
            raise ApprovalInvalidated("approval revision is invalid")
        if record.status == "executed":
            raise ApprovalAlreadyClaimed("approval was already claimed")
        raise ApprovalNotApproved("approval is not approved")
