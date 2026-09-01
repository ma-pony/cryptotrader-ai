"""Read-only projections of evaluations for the canonical component history."""

from typing import Literal

from fastapi import APIRouter, Query, Request

from cryptotrader.signals.evaluation_store import EvaluationList, EvaluationStatus

router = APIRouter(prefix="/api/components", tags=["components"])


@router.get("/{component_id}/evaluations", response_model=EvaluationList)
async def evaluations(
    component_id: str,
    request: Request,
    pair: str | None = None,
    mode: Literal["analysis", "trading", "backtest"] | None = None,
    config_revision: int | None = None,
    interval: str | None = None,
    status: EvaluationStatus | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> EvaluationList:
    store = request.app.state.runtime.evaluation_store
    filters = {
        "component_id": component_id,
        "pair": pair,
        "mode": mode,
        "config_revision": config_revision,
        "interval": interval,
    }
    rows = await store.list({**filters, "status": status})
    return EvaluationList(
        items=tuple(rows[offset : offset + limit]),
        summary=await store.summary(filters),
        total=len(rows),
        limit=limit,
        offset=offset,
        has_next=offset + limit < len(rows),
    )
