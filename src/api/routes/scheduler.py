"""Canonical scheduler status + trigger rule CRUD endpoints.

Protected routes (``/api/scheduler/*``):

- ``GET /api/scheduler/status`` — contract endpoint (FR-802).
- ``GET/POST /api/scheduler/rules`` — trigger rule CRUD.
- ``GET /api/scheduler/triggers`` — trigger event history.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo, field_validator

if TYPE_CHECKING:
    from cryptotrader.scheduler import Scheduler
    from cryptotrader.triggers.store import TriggerRuleStore

logger = logging.getLogger(__name__)
api_router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def _get_scheduler(request: Request) -> Scheduler | None:
    """Return the Scheduler instance from app.state, or None if not registered."""
    return getattr(request.app.state, "scheduler", None)


# ---------------------------------------------------------------------------
# Contract endpoint — /api/scheduler/status (FR-802)
# ---------------------------------------------------------------------------


class SchedulerContractStatus(BaseModel):
    """Data-model §2 SchedulerStatus shape for the React Dashboard."""

    enabled: bool
    config_revision: int
    pairs: list[str]
    enabled_books: list[str]
    next_pair: str | None
    next_run_at: datetime | None
    redis_available: bool


def _next_trading_run(scheduler: Scheduler | None) -> tuple[str | None, datetime | None]:
    """Return ``(next_pair, next_run_at)`` from the live scheduler, or (None, None)."""
    if scheduler is None:
        return (None, None)

    # spec 013: scheduler.pairs is list[Pair]; SchedulerContractStatus.next_pair
    # is str. Coerce via canonical() — never let a Pair object escape into
    # the API response (production observation 2026-05-02: SchedulerContractStatus
    # validation 500'd because triggers got enabled and the page started fetching
    # /api/scheduler/status, which had been silently broken behind the disabled flag).
    raw_pair = scheduler.pairs[0] if scheduler.pairs else None
    next_pair: str | None
    if raw_pair is None:
        next_pair = None
    elif hasattr(raw_pair, "canonical"):
        next_pair = raw_pair.canonical()
    else:
        next_pair = str(raw_pair)

    next_run: datetime | None = None
    for raw in scheduler.jobs:
        if raw.get("id") != "trading_cycle":
            continue
        raw_ts = raw.get("next_run_time")
        if isinstance(raw_ts, datetime):
            next_run = raw_ts
        elif isinstance(raw_ts, str):
            try:
                next_run = datetime.fromisoformat(raw_ts)
            except ValueError:
                logger.debug("Cannot parse trading_cycle next_run_time %r", raw_ts)
        break
    return (next_pair, next_run)


@api_router.get("/status", response_model=SchedulerContractStatus)
async def scheduler_status_v2(request: Request) -> SchedulerContractStatus:
    """Return scheduler status in the data-model contract shape (FR-802)."""
    from cryptotrader.risk.state import RedisStateManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    config = runtime.snapshot.document
    rsm = RedisStateManager(config.infrastructure.redis_url)

    scheduler = _get_scheduler(request)
    next_pair, next_run_at = _next_trading_run(scheduler)

    return SchedulerContractStatus(
        enabled=bool(getattr(config.scheduler, "enabled", False)),
        config_revision=runtime.snapshot.revision,
        pairs=[pair.canonical() if hasattr(pair, "canonical") else str(pair) for pair in config.execution.pairs],
        enabled_books=[book.id for book in config.execution.books if book.enabled],
        next_pair=next_pair,
        next_run_at=next_run_at,
        redis_available=bool(rsm.available),
    )


# ---------------------------------------------------------------------------
# Trigger rule CRUD models
# ---------------------------------------------------------------------------


class _TriggerParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True)


class _PriceParameters(_TriggerParameters):
    direction: Literal["above", "below"]
    price: float = Field(gt=0)


class _ChangeParameters(_TriggerParameters):
    window_minutes: int = Field(ge=1)
    threshold_pct: float = Field(gt=0)


class _CandleParameters(_TriggerParameters):
    interval: str = Field(min_length=1)
    consecutive_count: int = Field(ge=1)
    direction: Literal["bearish", "bullish"]


class _FundingParameters(_TriggerParameters):
    threshold_pct: float = Field(gt=0)


class ScheduleRuleIn(BaseModel):
    """Create/update request body for a trigger rule."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=255)
    trigger_type: str = Field(..., pattern=r"^(price_threshold|pct_change|candle_pattern|funding_rate)$")
    pair: str = Field(..., min_length=3, max_length=20)
    parameters: dict[str, Any]
    cooldown_minutes: int = Field(default=30, ge=1, le=1440)

    @field_validator("parameters")
    @classmethod
    def validate_active_parameters(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        parameter_model = {
            "price_threshold": _PriceParameters,
            "pct_change": _ChangeParameters,
            "candle_pattern": _CandleParameters,
            "funding_rate": _FundingParameters,
        }.get(info.data.get("trigger_type"))
        return parameter_model.model_validate(value).model_dump() if parameter_model else value


class _ScheduleRuleOut(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True, strict=True)

    id: str
    name: str
    pair: str
    cooldown_minutes: int
    enabled: bool
    ttl_expires_at: datetime | None
    created_by: str
    schedule_depth: int
    created_at: datetime
    updated_at: datetime
    in_cooldown: bool = False
    last_triggered_at: datetime | None = None


class _PriceScheduleRuleOut(_ScheduleRuleOut):
    trigger_type: Literal["price_threshold"]
    parameters: _PriceParameters


class _ChangeScheduleRuleOut(_ScheduleRuleOut):
    trigger_type: Literal["pct_change"]
    parameters: _ChangeParameters


class _CandleScheduleRuleOut(_ScheduleRuleOut):
    trigger_type: Literal["candle_pattern"]
    parameters: _CandleParameters


class _FundingScheduleRuleOut(_ScheduleRuleOut):
    trigger_type: Literal["funding_rate"]
    parameters: _FundingParameters


ScheduleRuleOut = Annotated[
    _PriceScheduleRuleOut | _ChangeScheduleRuleOut | _CandleScheduleRuleOut | _FundingScheduleRuleOut,
    Field(discriminator="trigger_type"),
]
_schedule_rule_adapter = TypeAdapter(ScheduleRuleOut)


class PriceSnapshotOut(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True)

    pair: str = Field(min_length=1)
    price: float
    ts: float


class TriggerEventOut(BaseModel):
    """Response model for a trigger event."""

    model_config = ConfigDict(extra="forbid", from_attributes=True, strict=True)

    id: str
    rule_id: str
    triggered_at: datetime
    trigger_reason: str
    price_snapshot: PriceSnapshotOut
    analysis_commit_id: str | None
    schedule_depth: int
    cooldown_skipped: bool


class PaginatedTriggerEvents(BaseModel):
    items: list[TriggerEventOut]
    total: int
    page: int
    size: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_trigger_store(request: Request) -> TriggerRuleStore:
    store = getattr(request.app.state, "trigger_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Trigger engine not initialized")
    return store


async def _enrich_rule(rule: Any, request: Request) -> ScheduleRuleOut:
    """Convert ORM rule to response model with runtime fields."""
    store = _get_trigger_store(request)
    last_triggered = await store.get_last_triggered_at(rule.id)
    cooldown_key = f"trigger:cooldown:{rule.id}"

    from cryptotrader.risk.state import RedisStateManager

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    config = runtime.snapshot.document
    rsm = RedisStateManager(config.infrastructure.redis_url)
    in_cooldown = (await rsm.get(cooldown_key)) is not None

    return _schedule_rule_adapter.validate_python(
        {
            "id": rule.id,
            "name": rule.name,
            "trigger_type": rule.trigger_type,
            "pair": rule.pair,
            "parameters": rule.parameters,
            "cooldown_minutes": rule.cooldown_minutes,
            "enabled": rule.enabled,
            "ttl_expires_at": rule.ttl_expires_at,
            "created_by": rule.created_by,
            "schedule_depth": rule.schedule_depth,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at,
            "in_cooldown": in_cooldown,
            "last_triggered_at": last_triggered,
        }
    )


# ---------------------------------------------------------------------------
# Rule CRUD endpoints — /api/scheduler/rules
# ---------------------------------------------------------------------------


@api_router.get("/rules", response_model=list[ScheduleRuleOut])
async def list_rules(
    request: Request,
    enabled: bool | None = Query(default=None),
) -> list[ScheduleRuleOut]:
    store = _get_trigger_store(request)
    rules = await store.list_rules(enabled_only=bool(enabled))
    return [await _enrich_rule(r, request) for r in rules]


@api_router.post("/rules", response_model=ScheduleRuleOut, status_code=201)
async def create_rule(request: Request, body: ScheduleRuleIn) -> ScheduleRuleOut:
    store = _get_trigger_store(request)

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not initialized")
    config = runtime.snapshot.document
    count = await store.count_rules()
    if count >= config.triggers.max_rules:
        raise HTTPException(status_code=422, detail=f"Maximum {config.triggers.max_rules} rules reached")

    rule = await store.create_rule(body.model_dump())
    _reload_engine(request)
    return await _enrich_rule(rule, request)


@api_router.get("/rules/{rule_id}", response_model=ScheduleRuleOut)
async def get_rule(request: Request, rule_id: str) -> ScheduleRuleOut:
    store = _get_trigger_store(request)
    rule = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return await _enrich_rule(rule, request)


@api_router.put("/rules/{rule_id}", response_model=ScheduleRuleOut)
async def update_rule(request: Request, rule_id: str, body: ScheduleRuleIn) -> ScheduleRuleOut:
    store = _get_trigger_store(request)
    rule = await store.update_rule(rule_id, body.model_dump())
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    _reload_engine(request)
    return await _enrich_rule(rule, request)


@api_router.patch("/rules/{rule_id}/toggle", response_model=ScheduleRuleOut)
async def toggle_rule(request: Request, rule_id: str) -> ScheduleRuleOut:
    store = _get_trigger_store(request)
    rule = await store.toggle_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    _reload_engine(request)
    return await _enrich_rule(rule, request)


@api_router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(request: Request, rule_id: str) -> Response:
    store = _get_trigger_store(request)
    deleted = await store.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")
    _reload_engine(request)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Trigger event endpoints — /api/scheduler/triggers
# ---------------------------------------------------------------------------


@api_router.get("/triggers", response_model=PaginatedTriggerEvents)
async def list_triggers(
    request: Request,
    rule_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> PaginatedTriggerEvents:
    store = _get_trigger_store(request)
    events, total = await store.list_events(page, size, rule_id=rule_id)
    return PaginatedTriggerEvents(
        items=[TriggerEventOut.model_validate(e, from_attributes=True) for e in events],
        total=total,
        page=page,
        size=size,
    )


@api_router.get("/triggers/{event_id}", response_model=TriggerEventOut)
async def get_trigger_event(request: Request, event_id: str) -> TriggerEventOut:
    store = _get_trigger_store(request)
    session = await store._session()
    async with session:
        from cryptotrader.triggers.models import TriggerEventRecord

        event = await session.get(TriggerEventRecord, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Trigger event not found")
        return TriggerEventOut.model_validate(event, from_attributes=True)


def _reload_engine(request: Request) -> None:
    """Fire-and-forget rule reload on the trigger engine."""
    engine = getattr(request.app.state, "trigger_engine", None)
    if engine is not None:
        import asyncio

        task = asyncio.create_task(engine.reload_rules())
        task.add_done_callback(lambda t: t.result() if not t.cancelled() and not t.exception() else None)
