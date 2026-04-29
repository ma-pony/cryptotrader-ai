"""Async CRUD service for trigger rules and events."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from cryptotrader.triggers.models import ScheduleRule, TriggerEventRecord

logger = logging.getLogger(__name__)


class TriggerRuleStore:
    """Async CRUD for ScheduleRule and TriggerEventRecord.

    Accepts a callable that returns an ``AsyncSession`` (typically a
    ``functools.partial(get_async_session, url)``).
    """

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def _session(self) -> AsyncSession:
        return await self._session_factory()

    async def list_rules(self, *, enabled_only: bool = False) -> list[ScheduleRule]:
        session = await self._session()
        async with session:
            stmt = select(ScheduleRule).order_by(ScheduleRule.created_at.desc())
            if enabled_only:
                stmt = stmt.where(ScheduleRule.enabled.is_(True))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_rule(self, rule_id: str) -> ScheduleRule | None:
        session = await self._session()
        async with session:
            return await session.get(ScheduleRule, rule_id)

    async def create_rule(self, data: dict[str, Any]) -> ScheduleRule:
        session = await self._session()
        async with session:
            rule = ScheduleRule(**data)
            session.add(rule)
            await session.commit()
            await session.refresh(rule)
            return rule

    async def update_rule(self, rule_id: str, data: dict[str, Any]) -> ScheduleRule | None:
        session = await self._session()
        async with session:
            rule = await session.get(ScheduleRule, rule_id)
            if rule is None:
                return None
            for key, value in data.items():
                setattr(rule, key, value)
            rule.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(rule)
            return rule

    async def toggle_rule(self, rule_id: str) -> ScheduleRule | None:
        session = await self._session()
        async with session:
            rule = await session.get(ScheduleRule, rule_id)
            if rule is None:
                return None
            rule.enabled = not rule.enabled
            rule.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(rule)
            return rule

    async def delete_rule(self, rule_id: str) -> bool:
        session = await self._session()
        async with session:
            result = await session.execute(delete(ScheduleRule).where(ScheduleRule.id == rule_id))
            await session.commit()
            return result.rowcount > 0  # type: ignore[union-attr]

    async def record_event(self, event_data: dict[str, Any]) -> TriggerEventRecord:
        session = await self._session()
        async with session:
            event = TriggerEventRecord(**event_data)
            session.add(event)
            await session.commit()
            await session.refresh(event)
            return event

    async def list_events(
        self, page: int = 1, size: int = 20, *, rule_id: str | None = None
    ) -> tuple[list[TriggerEventRecord], int]:
        session = await self._session()
        async with session:
            stmt = select(TriggerEventRecord).order_by(TriggerEventRecord.triggered_at.desc())
            count_stmt = select(func.count()).select_from(TriggerEventRecord)
            if rule_id:
                stmt = stmt.where(TriggerEventRecord.rule_id == rule_id)
                count_stmt = count_stmt.where(TriggerEventRecord.rule_id == rule_id)
            total = (await session.execute(count_stmt)).scalar() or 0
            offset = (page - 1) * size
            stmt = stmt.offset(offset).limit(size)
            result = await session.execute(stmt)
            return list(result.scalars().all()), total

    async def cleanup_expired_rules(self) -> int:
        now = datetime.now(UTC)
        session = await self._session()
        async with session:
            stmt = (
                delete(ScheduleRule)
                .where(ScheduleRule.ttl_expires_at.isnot(None))
                .where(ScheduleRule.ttl_expires_at < now)
                .where(ScheduleRule.created_by == "agent")
            )
            result = await session.execute(stmt)
            await session.commit()
            count = result.rowcount or 0  # type: ignore[union-attr]
            if count > 0:
                logger.info("Cleaned up %d expired agent rules", count)
            return count

    async def count_rules(self) -> int:
        session = await self._session()
        async with session:
            result = await session.execute(select(func.count()).select_from(ScheduleRule))
            return result.scalar() or 0

    async def get_last_triggered_at(self, rule_id: str) -> datetime | None:
        session = await self._session()
        async with session:
            stmt = (
                select(func.max(TriggerEventRecord.triggered_at))
                .where(TriggerEventRecord.rule_id == rule_id)
                .where(TriggerEventRecord.cooldown_skipped.is_(False))
            )
            result = await session.execute(stmt)
            return result.scalar()

    @staticmethod
    async def ensure_tables(database_url: str) -> None:
        """Create tables using the provided database URL."""
        from cryptotrader.db import get_engine
        from cryptotrader.triggers.models import Base

        engine = await get_engine(database_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
