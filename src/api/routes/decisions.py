"""产品 decisions 视图读取 canonical multi-venue Journal 契约。"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - FastAPI evaluates route annotations at runtime.
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from cryptotrader.decision.read_service import DecisionListOut, DecisionOut

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


@router.get("", response_model=DecisionListOut)
async def list_decisions(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    component_id: str | None = Query(None),
    pair: str | None = None,
    mode: Literal["analysis", "trading", "backtest"] | None = None,
    origin: Literal["manual", "scheduled", "trigger", "backtest"] | None = None,
    revision: int | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> DecisionListOut:
    if any(value is not None and value.utcoffset() is None for value in (started_at, ended_at)):
        raise HTTPException(422, "Filter times must include a timezone")
    return await request.app.state.runtime.read_service.list(
        limit=limit,
        offset=offset,
        component_id=component_id,
        pair=pair,
        mode=mode,
        origin=origin,
        revision=revision,
        started_at=started_at,
        ended_at=ended_at,
    )


@router.get("/{decision_id}", response_model=DecisionOut)
async def get_decision(decision_id: str, request: Request) -> DecisionOut:
    record = await request.app.state.runtime.read_service.get(decision_id)
    if record is None:
        raise HTTPException(404, "Decision was not found")
    return record
