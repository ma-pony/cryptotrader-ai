"""Separate evaluation projections; never rewrites the canonical prediction Journal."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from decimal import Decimal  # noqa: TC003 - Pydantic resolves these fields at runtime.
from typing import Literal

from pydantic import AwareDatetime, Field
from sqlalchemy import JSON, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cryptotrader.db import get_async_session
from cryptotrader.migrations.schema import require_tables
from cryptotrader.signals.presentation import EvaluationReference, PredictionComparison, PresentationModel

EvaluationStatus = Literal["pending", "evaluated", "missing_market", "not_directional", "skipped", "failed"]


class EvaluationRecord(PresentationModel):
    decision_id: str
    component_id: str
    pair: str | None
    mode: Literal["analysis", "trading", "backtest"]
    config_revision: int
    interval: str | None
    created_at: AwareDatetime
    status: EvaluationStatus
    direction: Literal["long", "short", "neutral"]
    reference: EvaluationReference | None
    actual_price: Decimal | None = None
    actual_time: AwareDatetime | None = None
    hit: bool | None = None
    return_ratio: Decimal | None = None
    reason: str | None = None
    comparisons: tuple[PredictionComparison, ...] = ()
    cost: Decimal | None = Field(default=None, ge=0)


class EvaluationGroup(PresentationModel):
    component_id: str
    pair: str | None
    mode: Literal["analysis", "trading", "backtest"]
    config_revision: int
    interval: str | None
    total: int
    pending: int
    matured_directional: int
    hits: int
    neutral: int
    skipped: int
    failed: int
    missing_market: int
    hit_rate: float | None


class EvaluationSummary(PresentationModel):
    groups: tuple[EvaluationGroup, ...]


class EvaluationList(PresentationModel):
    items: tuple[EvaluationRecord, ...]
    summary: EvaluationSummary
    total: int
    limit: int
    offset: int
    has_next: bool


class EvaluationBase(DeclarativeBase):
    pass


class EvaluationRow(EvaluationBase):
    __tablename__ = "component_evaluations"
    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    component_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)


class EvaluationStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self._records = {}
        self._ready = False
        self._lock = asyncio.Lock()

    async def ensure_table(self):
        if self.database_url is not None and not self._ready:
            await require_tables(self.database_url, EvaluationBase.metadata.tables)
            self._ready = True

    async def upsert(self, record: EvaluationRecord):
        record = EvaluationRecord.model_validate(record)
        async with self._lock:
            if self.database_url is None:
                self._records[record.decision_id, record.component_id] = record
                return
            await self.ensure_table()
            async with await get_async_session(self.database_url) as session:
                await session.merge(
                    EvaluationRow(
                        decision_id=record.decision_id,
                        component_id=record.component_id,
                        payload=record.model_dump(mode="json"),
                    )
                )
                await session.commit()

    async def get(self, decision_id, component_id):
        if self.database_url is None:
            return self._records.get((decision_id, component_id))
        await self.ensure_table()
        async with await get_async_session(self.database_url) as session:
            row = await session.get(EvaluationRow, (decision_id, component_id))
            return None if row is None else EvaluationRecord.model_validate(row.payload)

    async def list(self, filters=None):
        filters = filters or {}
        if self.database_url is None:
            records = list(self._records.values())
        else:
            await self.ensure_table()
            async with await get_async_session(self.database_url) as session:
                rows = (await session.execute(select(EvaluationRow))).scalars().all()
                records = [EvaluationRecord.model_validate(row.payload) for row in rows]
        return sorted(
            (r for r in records if all(value is None or getattr(r, key) == value for key, value in filters.items())),
            key=lambda r: (r.created_at, r.decision_id),
            reverse=True,
        )

    async def summary(self, filters=None) -> EvaluationSummary:
        from cryptotrader.signals.evaluation import direction_hit_rate

        grouped = defaultdict(list)
        for row in await self.list(filters):
            grouped[row.component_id, row.pair, row.mode, row.config_revision, row.interval].append(row)
        groups = []
        for key, rows in grouped.items():
            counts = {
                status: sum(r.status == status for r in rows)
                for status in ("pending", "evaluated", "not_directional", "skipped", "failed", "missing_market")
            }
            hits = sum(r.hit is True for r in rows)
            groups.append(
                EvaluationGroup(
                    component_id=key[0],
                    pair=key[1],
                    mode=key[2],
                    config_revision=key[3],
                    interval=key[4],
                    total=len(rows),
                    pending=counts["pending"],
                    matured_directional=counts["evaluated"],
                    hits=hits,
                    neutral=counts["not_directional"],
                    skipped=counts["skipped"],
                    failed=counts["failed"],
                    missing_market=counts["missing_market"],
                    hit_rate=direction_hit_rate(hits, counts["evaluated"]),
                )
            )
        return EvaluationSummary(groups=tuple(groups))
