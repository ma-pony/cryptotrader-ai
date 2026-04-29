"""SQLAlchemy ORM models for price trigger rules and events."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from cryptotrader._compat import UTC


class Base(DeclarativeBase):
    pass


class ScheduleRule(Base):
    __tablename__ = "schedule_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)
    pair: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    ttl_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    schedule_depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    events: Mapped[list[TriggerEventRecord]] = relationship(back_populates="rule", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_schedule_rules_enabled_pair", "enabled", "pair"),)


class TriggerEventRecord(Base):
    __tablename__ = "trigger_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("schedule_rules.id", ondelete="CASCADE"), nullable=False
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    trigger_reason: Mapped[str] = mapped_column(String(500), nullable=False)
    price_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    analysis_commit_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schedule_depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cooldown_skipped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    rule: Mapped[ScheduleRule] = relationship(back_populates="events")

    __table_args__ = (Index("ix_trigger_events_rule_id_triggered_at", "rule_id", "triggered_at"),)
