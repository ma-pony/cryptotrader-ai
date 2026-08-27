"""Persistence for the single global signal profile."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session, get_engine
from cryptotrader.profiles.models import ComponentWeight, SignalProfile

_GLOBAL_ID = "global"
_ready: set[str] = set()


class _Base(DeclarativeBase):
    pass


class _SignalProfileRow(_Base):
    __tablename__ = "signal_profile"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _profile_config(profile: SignalProfile) -> dict[str, Any]:
    return {
        "components": [
            {
                "component_id": item.component_id,
                "enabled": item.enabled,
                "weight": item.weight,
            }
            for item in profile.components
        ],
        "neutral_threshold": profile.neutral_threshold,
        "max_target_ratio": profile.max_target_ratio,
        "atr_stop_multiplier": profile.atr_stop_multiplier,
        "reward_ratio": profile.reward_ratio,
        "hitl_required": profile.hitl_required,
    }


def _to_profile(row: _SignalProfileRow) -> SignalProfile:
    config = row.config
    return SignalProfile(
        revision=row.revision,
        components=tuple(ComponentWeight(**item) for item in config["components"]),
        neutral_threshold=float(config["neutral_threshold"]),
        max_target_ratio=float(config["max_target_ratio"]),
        atr_stop_multiplier=float(config["atr_stop_multiplier"]),
        reward_ratio=float(config["reward_ratio"]),
        hitl_required=bool(config["hitl_required"]),
    )


class SignalProfileRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def ensure_table(self) -> None:
        if self.database_url in _ready:
            return
        engine = await get_engine(self.database_url)
        async with engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)
        _ready.add(self.database_url)

    async def get(self) -> SignalProfile | None:
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_SignalProfileRow, _GLOBAL_ID)
            return _to_profile(row) if row is not None else None
        finally:
            await session.close()

    async def get_or_create(self, default: SignalProfile) -> SignalProfile:
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            row = await session.get(_SignalProfileRow, _GLOBAL_ID)
            if row is None:
                row = _SignalProfileRow(
                    id=_GLOBAL_ID,
                    revision=1,
                    config=_profile_config(default),
                    updated_at=datetime.now(UTC),
                )
                session.add(row)
                await session.commit()
            return _to_profile(row)
        finally:
            await session.close()

    async def replace(self, profile: SignalProfile) -> SignalProfile:
        await self.ensure_table()
        session = await get_async_session(self.database_url)
        try:
            result = await session.execute(
                select(_SignalProfileRow).where(_SignalProfileRow.id == _GLOBAL_ID).with_for_update()
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise LookupError("global signal profile does not exist")
            row.revision += 1
            row.config = _profile_config(profile)
            row.updated_at = datetime.now(UTC)
            await session.commit()
            return _to_profile(row)
        finally:
            await session.close()
