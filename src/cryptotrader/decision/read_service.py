"""Persisted decisions, independent of the currently executable runtime graph."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from api.routes.response_dto import (
    BookCycleOut,
    ComponentSignalOut,
    FusedSignalOut,
    JsonEntryOut,
    TargetPositionOut,
    cycle_out,
    json_entries_out,
)
from cryptotrader.decision.analysis import AnalysisFailure


class DecisionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str
    pair: str | None
    mode: Literal["analysis", "trading", "backtest"]
    origin: Literal["manual", "scheduled", "trigger", "backtest"] | None
    config_revision: int
    config_snapshot: list[JsonEntryOut]
    created_at: AwareDatetime
    finished_at: AwareDatetime | None
    status: str
    components: list[ComponentSignalOut]
    fusion: FusedSignalOut | None
    target: TargetPositionOut | None
    books: list[BookCycleOut]
    failure: AnalysisFailure | None
    incomplete_fields: list[str]


class DecisionListOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DecisionOut]
    total: int
    limit: int
    offset: int
    has_next: bool


def decision_out(record) -> DecisionOut:
    shared = cycle_out(record)
    return DecisionOut(
        decision_id=record.cycle_id,
        pair=record.run.pair,
        mode=record.run.mode,
        origin=record.run.origin,
        config_revision=record.config_revision,
        config_snapshot=json_entries_out(record.run.config_snapshot),
        created_at=record.created_at,
        finished_at=record.run.finished_at,
        status=record.cycle_status,
        components=shared.shared_signals.components,
        fusion=shared.shared_signals.fused,
        target=shared.shared_signals.target_position,
        books=shared.books,
        failure=record.run.failure,
        incomplete_fields=list(record.run.incomplete_fields),
    )


class DecisionReadService:
    def __init__(self, journal):
        self.journal = journal

    async def get(self, decision_id: str) -> DecisionOut | None:
        record = await self.journal.get(decision_id)
        return None if record is None else decision_out(record)

    async def list(
        self,
        *,
        pair=None,
        mode=None,
        origin=None,
        revision=None,
        started_at=None,
        ended_at=None,
        limit=50,
        offset=0,
        component_id=None,
    ) -> DecisionListOut:
        filters = {
            "pair": pair,
            "mode": mode,
            "origin": origin,
            "revision": revision,
            "started_at": started_at,
            "ended_at": ended_at,
            "component_id": component_id,
        }
        records = await self.journal.list(limit=limit, offset=offset, **filters)
        total = await self.journal.count(**filters)
        return DecisionListOut(
            items=[decision_out(record) for record in records],
            total=total,
            limit=limit,
            offset=offset,
            has_next=offset + limit < total,
        )
