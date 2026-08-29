"""多平台周期 Journal 的 canonical list/detail API。"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves this response field at runtime.
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from api.routes.portfolio_books import (
    BookPortfolioOut,
    ConnectionPortfolioOut,
    book_portfolio_out,
    connection_portfolio_out,
)
from api.routes.response_dto import (
    BookRiskDecisionOut,
    ConnectionExecutionPlanOut,
    ConnectionExecutionResultOut,
    ConnectionRiskDecisionOut,
    JsonEntryOut,
    book_risk_out,
    connection_execution_out,
    connection_risk_out,
    json_entries_out,
    plan_out,
)

router = APIRouter(prefix="/api/cycles", tags=["cycles"])


class ComponentSignalOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    direction: str
    confidence: float
    reasoning: str
    details: list[JsonEntryOut]


class ComponentContributionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    weight: float
    signed_score: float
    weighted_score: float


class FusedSignalOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float
    reasoning: str
    contributions: list[ComponentContributionOut]


class TargetPositionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    side: str
    size_ratio: float


class SharedSignalsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: list[ComponentSignalOut]
    fused: FusedSignalOut | None
    target_position: TargetPositionOut | None


class BookHitlOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str | None
    status: str
    config_revision: int


class BookExecutionSummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    requires_attention: bool
    reallocated: bool


class CycleConnectionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connection_id: str
    portfolio_before: ConnectionPortfolioOut | None
    portfolio_after: ConnectionPortfolioOut | None
    risk: ConnectionRiskDecisionOut | None
    plan: ConnectionExecutionPlanOut | None
    execution: ConnectionExecutionResultOut | None
    unavailable: bool


class BookFailureOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str


class BookCycleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str
    capital_scope: Literal["simulated", "real"]
    config_revision: int
    pair: str
    market_type: str
    status: str
    hitl: BookHitlOut
    failure: BookFailureOut | None
    requested_target_exposure: str | None
    target_exposure: str | None
    risk: BookRiskDecisionOut | None
    ready: bool | None
    errors: list[str]
    execution: BookExecutionSummaryOut | None
    portfolio_before: BookPortfolioOut | None
    portfolio_after: BookPortfolioOut | None
    portfolio_after_available: bool | None
    connections: list[CycleConnectionOut]


class CycleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_id: str
    config_revision: int
    market_data_source_id: str
    shared_signals: SharedSignalsOut
    books: list[BookCycleOut]
    cycle_status: str
    execution_status: str
    requires_attention: bool
    created_at: datetime


class PaginatedCyclesOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CycleOut]
    total: int
    page: int
    size: int
    has_next: bool


def _journal(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    cycle = getattr(runtime, "cycle", None) if runtime is not None else None
    journal = getattr(cycle, "journal", None) if cycle is not None else None
    if journal is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not active")
    return journal


def _connection_ids(book) -> tuple[str, ...]:
    ordered: list[str] = []
    sources = []
    if book.portfolio_before is not None:
        sources.append(item.connection_id for item in book.portfolio_before.connections)
    if book.proposal is not None:
        sources.append(item.connection_id for item in book.proposal.risk.connection_targets)
        sources.append(item.connection_id for item in book.proposal.connection_plans)
        sources.append(iter(book.proposal.unavailable_connections))
    if book.execution is not None:
        sources.append(item.connection_id for item in book.execution.connection_results)
    if book.portfolio_after is not None:
        sources.append(item.connection_id for item in book.portfolio_after.connections)
    for source in sources:
        for connection_id in source:
            if connection_id not in ordered:
                ordered.append(connection_id)
    return tuple(ordered)


def _book_out(book) -> BookCycleOut:
    proposal = book.proposal
    before = (
        {item.connection_id: connection_portfolio_out(item) for item in book.portfolio_before.connections}
        if book.portfolio_before is not None
        else {}
    )
    after = (
        {item.connection_id: connection_portfolio_out(item) for item in book.portfolio_after.connections}
        if book.portfolio_after is not None
        else {}
    )
    risks = (
        {item.connection_id: connection_risk_out(item) for item in proposal.connection_risks}
        if proposal is not None
        else {}
    )
    plans = {item.connection_id: plan_out(item) for item in proposal.connection_plans} if proposal is not None else {}
    executions = (
        {item.connection_id: connection_execution_out(item) for item in book.execution.connection_results}
        if book.execution is not None
        else {}
    )
    unavailable = set(proposal.unavailable_connections if proposal is not None else ())
    connections = [
        CycleConnectionOut(
            connection_id=connection_id,
            portfolio_before=before.get(connection_id),
            portfolio_after=after.get(connection_id),
            risk=risks.get(connection_id),
            plan=plans.get(connection_id),
            execution=executions.get(connection_id),
            unavailable=connection_id in unavailable,
        )
        for connection_id in _connection_ids(book)
    ]
    execution = (
        BookExecutionSummaryOut(
            status=book.execution.status,
            requires_attention=book.execution.requires_attention,
            reallocated=book.execution.reallocated,
        )
        if book.execution is not None
        else None
    )
    return BookCycleOut(
        book_id=book.book_id,
        capital_scope=book.capital_scope,
        config_revision=book.config_revision,
        pair=book.pair.canonical(),
        market_type=book.pair.market_type,
        status=book.status,
        hitl=BookHitlOut(
            approval_id=book.hitl.approval_id,
            status=book.hitl.status,
            config_revision=book.hitl.config_revision,
        ),
        failure=BookFailureOut(stage=book.failure.stage) if book.failure is not None else None,
        requested_target_exposure=(str(proposal.requested_target_exposure) if proposal is not None else None),
        target_exposure=str(proposal.target_exposure) if proposal is not None else None,
        risk=book_risk_out(proposal.risk) if proposal is not None else None,
        ready=proposal.ready if proposal is not None else None,
        errors=list(proposal.errors if proposal is not None else ()),
        execution=execution,
        portfolio_before=book_portfolio_out(book.portfolio_before) if book.portfolio_before is not None else None,
        portfolio_after=book_portfolio_out(book.portfolio_after) if book.portfolio_after is not None else None,
        portfolio_after_available=book.portfolio_after_available,
        connections=connections,
    )


def cycle_out(record) -> CycleOut:
    return CycleOut(
        cycle_id=record.cycle_id,
        config_revision=record.config_revision,
        market_data_source_id=record.market_data_source_id,
        shared_signals=SharedSignalsOut(
            components=[
                ComponentSignalOut(
                    component_id=item.component_id,
                    direction=item.direction,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                    details=json_entries_out(item.details),
                )
                for item in record.component_signals
            ],
            fused=(
                FusedSignalOut(
                    score=record.fused_signal.score,
                    reasoning=record.fused_signal.reasoning,
                    contributions=[
                        ComponentContributionOut(
                            component_id=item.component_id,
                            weight=item.weight,
                            signed_score=item.signed_score,
                            weighted_score=item.weighted_score,
                        )
                        for item in record.fused_signal.contributions
                    ],
                )
                if record.fused_signal is not None
                else None
            ),
            target_position=(
                TargetPositionOut(
                    side=record.target_position.side,
                    size_ratio=record.target_position.size_ratio,
                )
                if record.target_position is not None
                else None
            ),
        ),
        books=[_book_out(item) for item in record.book_results],
        cycle_status=record.cycle_status,
        execution_status=record.execution_status,
        requires_attention=record.requires_attention,
        created_at=record.created_at,
    )


async def list_cycle_page(request: Request, page: int, size: int) -> PaginatedCyclesOut:
    journal = _journal(request)
    offset = (page - 1) * size
    records = await journal.list(limit=size, offset=offset)
    total = await journal.count()
    return PaginatedCyclesOut(
        items=[cycle_out(item) for item in records],
        total=total,
        page=page,
        size=size,
        has_next=offset + len(records) < total,
    )


async def get_cycle_record(request: Request, cycle_id: str) -> CycleOut:
    record = await _journal(request).get(cycle_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Cycle not found")
    return cycle_out(record)


@router.get("", response_model=PaginatedCyclesOut)
async def list_cycles(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> PaginatedCyclesOut:
    return await list_cycle_page(request, page, size)


@router.get("/{cycle_id}", response_model=CycleOut)
async def get_cycle(cycle_id: str, request: Request) -> CycleOut:
    return await get_cycle_record(request, cycle_id)
