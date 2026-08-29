"""产品 decisions 视图读取 canonical multi-venue Journal 契约。"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.routes.cycles import CycleOut, PaginatedCyclesOut, get_cycle_record, list_cycle_page

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


@router.get("", response_model=PaginatedCyclesOut)
async def list_decisions(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> PaginatedCyclesOut:
    return await list_cycle_page(request, page, size)


@router.get("/{cycle_id}", response_model=CycleOut)
async def get_decision(cycle_id: str, request: Request) -> CycleOut:
    return await get_cycle_record(request, cycle_id)
