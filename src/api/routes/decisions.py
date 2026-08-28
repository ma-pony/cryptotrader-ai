"""Cycle Journal list and detail endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from cryptotrader.journal.models import TradingCycleRecord

CycleStatusValue = Literal[
    "completed",
    "no_change",
    "awaiting_approval",
    "approval_rejected",
    "component_failed",
    "risk_rejected",
    "execution_failed",
    "cancelled",
]

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


class DecisionListItem(BaseModel):
    cycle_id: str
    ts: str
    pair: str
    pair_display: str
    market_type: str
    status: CycleStatusValue
    profile_revision: int
    price: float = 0.0
    fused_score: float | None = None
    target_position: dict[str, Any] | None = None
    component_error: dict[str, str] | None = None
    risk_result: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None


class PaginatedDecisions(BaseModel):
    items: list[DecisionListItem]
    total: int
    page: int
    size: int
    has_next: bool


class DecisionDetailOut(BaseModel):
    cycle_id: str
    ts: str
    pair: str
    pair_display: str
    market_type: str
    status: CycleStatusValue
    profile_revision: int
    profile: dict[str, Any]
    context: dict[str, Any]
    components: list[dict[str, Any]]
    component_error: dict[str, str] | None = None
    fusion: dict[str, Any] | None = None
    target_position: dict[str, Any] | None = None
    trade_plan: dict[str, Any] | None = None
    hitl_result: dict[str, Any] | None = None
    risk_result: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None


def _store(request: Request | None = None):
    if request is not None:
        shared = getattr(request.app.state, "cycle_journal_store", None)
        if shared is not None:
            return shared
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import CycleJournalStore

    database_url = load_config().infrastructure.database_url or None
    return CycleJournalStore(database_url)


def _pair_meta(pair: str) -> tuple[str, str]:
    from cryptotrader.pair import Pair

    parsed = Pair.parse(pair)
    return parsed.display(), parsed.market_type


def _mapping(value) -> dict[str, Any] | None:
    return dict(value) if value is not None else None


def _list_item(record: TradingCycleRecord) -> DecisionListItem:
    pair_display, market_type = _pair_meta(record.pair)
    fusion = record.fused_signal or {}
    score = fusion.get("score")
    return DecisionListItem(
        cycle_id=record.cycle_id,
        ts=record.created_at.isoformat(),
        pair=record.pair,
        pair_display=pair_display,
        market_type=market_type,
        status=record.status,
        profile_revision=record.profile_revision,
        price=float(record.context_summary.get("current_price", 0.0) or 0.0),
        fused_score=float(score) if score is not None else None,
        target_position=_mapping(record.target_position),
        component_error=dict(record.component_error) if record.component_error is not None else None,
        risk_result=_mapping(record.risk_result),
        execution_result=_mapping(record.execution_result),
    )


def _detail(record: TradingCycleRecord) -> DecisionDetailOut:
    pair_display, market_type = _pair_meta(record.pair)
    return DecisionDetailOut(
        cycle_id=record.cycle_id,
        ts=record.created_at.isoformat(),
        pair=record.pair,
        pair_display=pair_display,
        market_type=market_type,
        status=record.status,
        profile_revision=record.profile_revision,
        profile=dict(record.profile_snapshot),
        context=dict(record.context_summary),
        components=[dict(item) for item in record.component_signals],
        component_error=dict(record.component_error) if record.component_error is not None else None,
        fusion=_mapping(record.fused_signal),
        target_position=_mapping(record.target_position),
        trade_plan=_mapping(record.trade_plan),
        hitl_result=_mapping(record.hitl_result),
        risk_result=_mapping(record.risk_result),
        execution_result=_mapping(record.execution_result),
    )


@router.get("", response_model=PaginatedDecisions)
async def list_decisions(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    pair: str | None = None,
    status: CycleStatusValue | None = None,
) -> PaginatedDecisions:
    store = _store(request)
    offset = (page - 1) * size
    records = await store.list(limit=size, offset=offset, pair=pair, status=status)
    total = await store.count(pair=pair, status=status)
    return PaginatedDecisions(
        items=[_list_item(record) for record in records],
        total=total,
        page=page,
        size=size,
        has_next=offset + len(records) < total,
    )


@router.get("/{cycle_id}", response_model=DecisionDetailOut)
async def get_decision(cycle_id: str, request: Request) -> DecisionDetailOut:
    record = await _store(request).get(cycle_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trading cycle {cycle_id} not found")
    return _detail(record)
