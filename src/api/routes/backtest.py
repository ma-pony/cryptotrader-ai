# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""Durable research runs. GET reads history without invoking a model."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from cryptotrader.accounts.store import payload
from cryptotrader.backtest.comparison import compare_runs
from cryptotrader.backtest.models import BacktestParams
from cryptotrader.cycle_serialization import json_value
from cryptotrader.decision.read_service import DecisionOut
from cryptotrader.tasks import TaskManagerClosedError, TooManyTasksError

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class BacktestRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: str = "queued"


class StrictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class BacktestModelEvidenceOut(StrictOut):
    requested_model: str | None
    actual_model: str | None
    actual_model_reason: str | None
    prompt_hash: str
    prompt_version: str
    status: Literal["started", "completed", "failed"]


class BacktestMetricsOut(StrictOut):
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float | None
    fill_count: int
    closed_trade_count: int


class BacktestEquityPointOut(StrictOut):
    ts: datetime
    equity: float


class BacktestDecisionSummaryOut(StrictOut):
    cycle_id: str
    status: str
    config_revision: int


class BacktestMoneyOut(StrictOut):
    amount: Decimal | None
    currency: str
    unavailable_reason: str | None


class BacktestInstrumentOut(StrictOut):
    venue_symbol: str
    pair: str | None
    market_type: str
    tradable: bool
    reason: str | None


class BacktestFillOut(StrictOut):
    connection_id: str
    venue_fill_id: str
    venue_order_id: str
    instrument: BacktestInstrumentOut
    side: str
    amount: Decimal
    price: Decimal
    occurred_at: datetime
    fee: BacktestMoneyOut
    realized_pnl: BacktestMoneyOut
    source: Literal["platform", "local_calculation"]
    client_order_id: str | None


class BacktestClosedTradeOut(StrictOut):
    pair: str
    opened_at: datetime
    closed_at: datetime
    side: Literal["long", "short"]
    gross_pnl: Decimal
    fees: Decimal
    funding: Decimal
    net_pnl: Decimal
    fill_ids: list[str]


class BacktestFundingOut(StrictOut):
    connection_id: str
    venue_entry_id: str
    instrument: BacktestInstrumentOut
    amount: BacktestMoneyOut
    occurred_at: datetime


class BacktestResultOut(StrictOut):
    metrics: BacktestMetricsOut
    equity_curve: list[BacktestEquityPointOut]
    decisions: list[DecisionOut | BacktestDecisionSummaryOut]
    decision_ids: list[str]
    fills: list[BacktestFillOut]
    closed_trades: list[BacktestClosedTradeOut]
    fees: Decimal
    funding: Decimal
    funding_entries: list[BacktestFundingOut]
    cost_assumptions: dict[str, JsonValue]
    unmodeled_costs: list[str]
    data_coverage: dict[str, JsonValue]


class BacktestRunOut(StrictOut):
    run_id: str
    params: BacktestParams
    config_snapshot: dict[str, JsonValue] | None
    status: Literal["queued", "running", "completed", "failed", "canceled", "interrupted"]
    progress: float
    started_at: datetime
    finished_at: datetime | None
    error: str | None
    incomplete_fields: list[str]
    model_evidence: list[BacktestModelEvidenceOut]
    result: BacktestResultOut | None


class BacktestRunsOut(StrictOut):
    items: list[BacktestRunOut]
    limit: int
    offset: int
    has_next: bool


class BacktestDifferenceOut(StrictOut):
    left: JsonValue
    right: JsonValue


class BacktestDifferenceReasonOut(BacktestDifferenceOut):
    reason: str


class BacktestComparisonOut(StrictOut):
    comparable: bool
    condition_differences: dict[str, BacktestDifferenceOut | BacktestDifferenceReasonOut]
    configuration_differences: dict[str, BacktestDifferenceOut | BacktestDifferenceReasonOut]
    left: BacktestRunOut
    right: BacktestRunOut


class BacktestCancelOut(StrictOut):
    canceled: Literal[True]


def _service(request):
    runtime = getattr(request.app.state, "runtime", None)
    service = getattr(runtime, "backtest_service", None)
    if service is None:
        raise HTTPException(503, "研究服务未初始化")
    return service


def _equity_curve_to_dicts(curve):
    return [{"ts": point.time.isoformat(), "equity": float(point.equity)} for point in curve]


def _result_to_dict(result):
    return json_value(
        {
            "metrics": {
                "total_return_pct": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown,
                "win_rate": result.win_rate,
                "fill_count": result.fill_count,
                "closed_trade_count": result.closed_trade_count,
            },
            "equity_curve": _equity_curve_to_dicts(result.equity_curve),
            "decisions": result.decisions,
            "decision_ids": result.decision_ids,
            "fills": payload(result.fills),
            "closed_trades": payload(result.closed_trades),
            "fees": str(result.fees),
            "funding": str(result.funding),
            "funding_entries": payload(result.funding_entries),
            "cost_assumptions": result.cost_assumptions,
            "unmodeled_costs": result.unmodeled_costs,
            "data_coverage": result.data_coverage,
        }
    )


def _run_to_dict(run, *, include_result=True):
    return {
        "run_id": run.run_id,
        "params": run.params.model_dump(mode="json"),
        "config_snapshot": run.config_snapshot,
        "status": run.status,
        "progress": run.progress,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
        "incomplete_fields": run.incomplete_fields,
        "model_evidence": run.model_evidence,
        "result": _result_to_dict(run.result) if include_result and run.result is not None else None,
    }


async def _get(store, identity):
    try:
        run = await store.get(identity)
    except ValueError as error:
        if "migrate_backtest_snapshots" in str(error):
            raise HTTPException(409, "存量回测快照尚未显式迁移，历史记录暂不可读。") from None
        raise
    if run is None:
        raise HTTPException(404, "回测不存在")
    return run


@router.post("/runs", response_model=BacktestRunResponse, status_code=202)
async def run_backtest(params: BacktestParams, request: Request):
    try:
        run_id = await _service(request).start(params)
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError:
        raise HTTPException(422, "配置快照不完整或不可复用，请检查已保存配置。") from None
    except (TaskManagerClosedError, TooManyTasksError):
        raise HTTPException(409, "任务暂时不可排入，请稍后重试。") from None
    return BacktestRunResponse(run_id=run_id)


@router.get("/runs", response_model=BacktestRunsOut)
async def list_runs(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    try:
        rows = await _service(request).store.list(limit + 1, offset)
    except ValueError as error:
        if "migrate_backtest_snapshots" in str(error):
            raise HTTPException(409, "存量回测快照尚未显式迁移，历史记录暂不可读。") from None
        raise
    return {
        "items": [_run_to_dict(row, include_result=False) for row in rows[:limit]],
        "limit": limit,
        "offset": offset,
        "has_next": len(rows) > limit,
    }


@router.get("/runs/compare", response_model=BacktestComparisonOut)
async def compare(request: Request, left: str, right: str):
    store = _service(request).store
    comparison = compare_runs(await _get(store, left), await _get(store, right))
    return {
        "comparable": comparison.comparable,
        "condition_differences": comparison.condition_differences,
        "configuration_differences": comparison.configuration_differences,
        "left": _run_to_dict(comparison.left),
        "right": _run_to_dict(comparison.right),
    }


@router.get("/runs/{run_id}", response_model=BacktestRunOut)
async def get_backtest_run(run_id: str, request: Request):
    return _run_to_dict(await _get(_service(request).store, run_id))


@router.delete("/runs/{run_id}", response_model=BacktestCancelOut)
async def cancel_backtest_run(run_id: str, request: Request):
    try:
        await _service(request).cancel(run_id)
    except LookupError:
        raise HTTPException(404, "回测不存在") from None
    except ValueError:
        raise HTTPException(409, "回测已结束或正在停止") from None
    return {"canceled": True}
