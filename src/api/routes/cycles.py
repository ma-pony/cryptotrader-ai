"""多平台周期 Journal 的 canonical list/detail API。"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves this response field at runtime.
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from api.routes.portfolio_books import (
    BookPortfolioOut,
    ConnectionPortfolioOut,
    book_portfolio_out,
    connection_portfolio_out,
)
from cryptotrader.cycle_serialization import component_signal_payload, fused_signal_payload, target_payload
from cryptotrader.execution.codec import book_execution_proposal_payload, book_execution_result_payload

router = APIRouter(prefix="/api/cycles", tags=["cycles"])


class SharedSignalsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: list[dict[str, Any]]
    fused: dict[str, Any] | None
    target_position: dict[str, Any] | None


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
    risk: dict[str, Any] | None
    plan: dict[str, Any] | None
    execution: dict[str, Any] | None
    unavailable: bool


class BookCycleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str
    capital_scope: Literal["simulated", "real"]
    config_revision: int
    pair: str
    market_type: str
    status: str
    hitl: BookHitlOut
    failure: dict[str, str] | None
    requested_target_exposure: str | None
    target_exposure: str | None
    risk: dict[str, Any] | None
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
    proposal_payload = book_execution_proposal_payload(book.proposal) if book.proposal is not None else None
    execution_payload = book_execution_result_payload(book.execution) if book.execution is not None else None
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
    risks = {item["connection_id"]: item for item in (proposal_payload or {}).get("connection_risks", [])}
    plans = {item["connection_id"]: item for item in (proposal_payload or {}).get("connection_plans", [])}
    executions = {item["connection_id"]: item for item in (execution_payload or {}).get("connection_results", [])}
    unavailable = set((proposal_payload or {}).get("unavailable_connections", []))
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
            status=execution_payload["status"],
            requires_attention=execution_payload["requires_attention"],
            reallocated=execution_payload["reallocated"],
        )
        if execution_payload is not None
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
        failure={"stage": book.failure.stage} if book.failure is not None else None,
        requested_target_exposure=(proposal_payload or {}).get("requested_target_exposure"),
        target_exposure=(proposal_payload or {}).get("target_exposure"),
        risk=(proposal_payload or {}).get("risk"),
        ready=(proposal_payload or {}).get("ready"),
        errors=list((proposal_payload or {}).get("errors", [])),
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
            components=[component_signal_payload(item) for item in record.component_signals],
            fused=fused_signal_payload(record.fused_signal) if record.fused_signal is not None else None,
            target_position=target_payload(record.target_position) if record.target_position is not None else None,
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
