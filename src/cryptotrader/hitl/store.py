"""HITL approval request persistence — SQLite/PostgreSQL via shared db.py."""

from __future__ import annotations

import logging
from datetime import datetime

from cryptotrader._compat import UTC
from cryptotrader.db import get_async_session, get_engine

logger = logging.getLogger(__name__)

_table_ready: set[str] = set()
_sa_cache: tuple | None = None


def _sa_models():
    global _sa_cache
    if _sa_cache is not None:
        return _sa_cache

    from sqlalchemy import Column, DateTime, Integer, String, Text
    from sqlalchemy.orm import DeclarativeBase

    class _Base(DeclarativeBase):
        pass

    class _ApprovalRow(_Base):
        __tablename__ = "hitl_approvals"
        approval_id = Column(String(36), primary_key=True)
        pair = Column(String(20), nullable=False, index=True)
        created_at = Column(DateTime(timezone=True), nullable=False)
        expires_at = Column(DateTime(timezone=True), nullable=False)
        trigger_reason = Column(String(50), nullable=False)
        verdict_snapshot = Column(Text, nullable=False)
        agent_analyses_snapshot = Column(Text, nullable=False)
        status = Column(String(20), nullable=False, default="pending", index=True)
        decision_by = Column(String(20), nullable=True)
        decided_at = Column(DateTime(timezone=True), nullable=True)
        comment = Column(Text, nullable=True)
        thread_id = Column(String(100), nullable=False)
        telegram_message_id = Column(Integer, nullable=True)

    _sa_cache = (_Base, _ApprovalRow)
    return _sa_cache


class ApprovalStore:
    @staticmethod
    async def ensure_table(db_url: str) -> None:
        if db_url in _table_ready:
            return
        base, _ = _sa_models()
        engine = await get_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)
        _table_ready.add(db_url)

    @staticmethod
    async def create(
        db_url: str,
        *,
        approval_id: str,
        pair: str,
        expires_at: datetime,
        trigger_reason: str,
        verdict_snapshot: str,
        agent_analyses_snapshot: str,
        thread_id: str,
    ) -> dict:
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        now = datetime.now(UTC)
        session = await get_async_session(db_url)
        try:
            row = row_cls(
                approval_id=approval_id,
                pair=pair,
                created_at=now,
                expires_at=expires_at,
                trigger_reason=trigger_reason,
                verdict_snapshot=verdict_snapshot,
                agent_analyses_snapshot=agent_analyses_snapshot,
                status="pending",
                thread_id=thread_id,
            )
            session.add(row)
            await session.commit()
        finally:
            await session.close()
        return {
            "approval_id": approval_id,
            "pair": pair,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "trigger_reason": trigger_reason,
            "status": "pending",
            "thread_id": thread_id,
        }

    @staticmethod
    async def get(db_url: str, approval_id: str) -> dict | None:
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        from sqlalchemy import select

        session = await get_async_session(db_url)
        try:
            result = await session.execute(select(row_cls).where(row_cls.approval_id == approval_id))
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _row_to_dict(row)
        finally:
            await session.close()

    @staticmethod
    async def list_pending(db_url: str) -> list[dict]:
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        from sqlalchemy import select

        session = await get_async_session(db_url)
        try:
            result = await session.execute(
                select(row_cls).where(row_cls.status == "pending").order_by(row_cls.created_at)
            )
            return [_row_to_dict(r) for r in result.scalars().all()]
        finally:
            await session.close()

    @staticmethod
    async def decide(
        db_url: str,
        approval_id: str,
        *,
        status: str,
        decision_by: str,
        comment: str = "",
    ) -> bool:
        """CAS update: returns True if update succeeded, False if concurrent conflict."""
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        from sqlalchemy import update

        now = datetime.now(UTC)
        session = await get_async_session(db_url)
        try:
            result = await session.execute(
                update(row_cls)
                .where(row_cls.approval_id == approval_id, row_cls.status == "pending")
                .values(
                    status=status,
                    decision_by=decision_by,
                    decided_at=now,
                    comment=comment,
                )
            )
            await session.commit()
            return result.rowcount > 0
        finally:
            await session.close()

    @staticmethod
    async def set_telegram_message_id(db_url: str, approval_id: str, message_id: int) -> None:
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        from sqlalchemy import update

        session = await get_async_session(db_url)
        try:
            await session.execute(
                update(row_cls).where(row_cls.approval_id == approval_id).values(telegram_message_id=message_id)
            )
            await session.commit()
        finally:
            await session.close()

    @staticmethod
    async def expire_stale(db_url: str) -> int:
        """Mark overdue pending approvals as expired. Returns count."""
        await ApprovalStore.ensure_table(db_url)
        _, row_cls = _sa_models()
        from sqlalchemy import update

        now = datetime.now(UTC)
        session = await get_async_session(db_url)
        try:
            result = await session.execute(
                update(row_cls)
                .where(row_cls.status == "pending", row_cls.expires_at < now)
                .values(
                    status="expired",
                    decision_by="timeout",
                    decided_at=now,
                )
            )
            await session.commit()
            count = result.rowcount
            if count > 0:
                logger.warning("Expired %d stale HITL approvals on startup", count)
            return count
        finally:
            await session.close()

    @staticmethod
    async def get_completed_trades_count(db_url: str) -> int:
        """Count completed trades (non-hold) from journal for cold-start detection."""
        from sqlalchemy import text

        session = await get_async_session(db_url)
        try:
            result = await session.execute(
                text("SELECT COUNT(*) FROM decision_commits WHERE verdict->>'action' != 'hold'")
            )
            return result.scalar_one()
        finally:
            await session.close()


def _row_to_dict(row: object) -> dict:
    return {
        "approval_id": row.approval_id,
        "pair": row.pair,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "trigger_reason": row.trigger_reason,
        "verdict_snapshot": row.verdict_snapshot,
        "agent_analyses_snapshot": row.agent_analyses_snapshot,
        "status": row.status,
        "decision_by": row.decision_by,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
        "comment": row.comment,
        "thread_id": row.thread_id,
        "telegram_message_id": row.telegram_message_id,
    }
