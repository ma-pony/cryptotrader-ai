"""Backtest run / status / cancel / sessions endpoints (FR-805/FR-806)."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# ── Request / response models ──


class BacktestParams(BaseModel):
    start: str
    end: str
    pair: str
    initial_capital: float = Field(ge=100)
    mode: Literal["rules", "llm"]
    session_name: str | None = None

    @model_validator(mode="after")
    def _validate_dates(self) -> BacktestParams:
        try:
            start_d = date.fromisoformat(self.start)
            end_d = date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError(f"start/end must be YYYY-MM-DD: {exc}") from exc
        if start_d >= end_d:
            raise ValueError("start must be before end")
        if end_d > datetime.now(UTC).date():
            raise ValueError("end must be ≤ today")
        return self


class BacktestRunResponse(BaseModel):
    run_id: str


class BacktestRunStatus(BaseModel):
    run_id: str
    params: BacktestParams
    status: Literal["queued", "running", "completed", "failed", "canceled"]
    progress: float
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    result: dict | None = None


class BacktestCancelResponse(BaseModel):
    canceled: bool


class BacktestSessionsList(BaseModel):
    sessions: list[str]


# ── In-process run registry ──

_RUNS: dict[str, dict[str, Any]] = {}
_TASKS: dict[str, asyncio.Task[Any]] = {}


def _new_run_id() -> str:
    return f"run_{secrets.token_hex(4)}"


def _spawn_run(params: BacktestParams) -> str:
    """Schedule a backtest in the background. Returns the new run_id."""
    from cryptotrader.task_registry import add_background_task

    run_id = _new_run_id()
    _RUNS[run_id] = {
        "run_id": run_id,
        "params": params.model_dump(),
        "status": "running",
        "progress": 0.0,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "error": None,
        "result": None,
    }

    task = add_background_task(_execute_backtest(run_id, params), name=f"backtest:{run_id}")
    _TASKS[run_id] = task
    return run_id


async def _execute_backtest(run_id: str, params: BacktestParams) -> None:
    from cryptotrader.backtest.engine import BacktestEngine

    try:
        engine = BacktestEngine(
            pair=params.pair,
            start=params.start,
            end=params.end,
            initial_capital=params.initial_capital,
            use_llm=(params.mode == "llm"),
        )
        result = await engine.run()
        _RUNS[run_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "finished_at": datetime.now(UTC).isoformat(),
                "result": _result_to_dict(result),
            }
        )
    except asyncio.CancelledError:
        _RUNS[run_id].update(
            {
                "status": "canceled",
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        raise
    except Exception as exc:
        logger.exception("Backtest %s failed", run_id)
        _RUNS[run_id].update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            }
        )


def _result_to_dict(result: Any) -> dict:
    """Translate cryptotrader BacktestResult → contract data-model §3 BacktestResult."""
    return {
        "metrics": {
            "total_return_pct": float(getattr(result, "total_return", 0.0) or 0.0),
            "sharpe": float(getattr(result, "sharpe_ratio", 0.0) or 0.0),
            "max_drawdown_pct": float(getattr(result, "max_drawdown", 0.0) or 0.0),
            "win_rate": float(getattr(result, "win_rate", 0.0) or 0.0),
            "trades_count": len(getattr(result, "trades", []) or []),
        },
        "equity_curve": _equity_curve_to_dicts(getattr(result, "equity_curve", []) or []),
        "decisions": list(getattr(result, "decisions", []) or []),
    }


def _equity_curve_to_dicts(curve: list) -> list[dict]:
    """BacktestResult.equity_curve is list[float]; synthesize ts."""
    if not curve:
        return []
    base = datetime.now(UTC)
    return [{"ts": base.isoformat(), "equity": float(v)} for v in curve]


def _get_run(run_id: str) -> dict | None:
    return _RUNS.get(run_id)


def _cancel_run(run_id: str) -> bool:
    """Cancel an in-flight run. Returns False if run is missing or already terminated."""
    info = _RUNS.get(run_id)
    if info is None:
        return False
    if info["status"] not in ("queued", "running"):
        return False
    task = _TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    info["status"] = "canceled"
    info["finished_at"] = datetime.now(UTC).isoformat()
    return True


# ── Sessions persistence helpers ──

_SESSIONS_DIR = Path.home() / ".cryptotrader" / "backtest_sessions"


def _load_session(name: str) -> dict | None:
    """Load a saved backtest session by name. Returns None if missing."""
    session_dir = _SESSIONS_DIR / name
    if not session_dir.is_dir():
        return None

    result_path = session_dir / "result.json"
    params_path = session_dir / "params.json"
    if not result_path.exists():
        return None

    try:
        with open(result_path) as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load session result %s", result_path, exc_info=True)
        return None

    params: dict = {}
    if params_path.exists():
        try:
            with open(params_path) as f:
                params = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.debug("Failed to load session params %s", params_path, exc_info=True)

    saved_at = datetime.fromtimestamp(result_path.stat().st_mtime, tz=UTC).isoformat()
    return {
        "name": name,
        "params": params,
        "result": result,
        "saved_at": saved_at,
    }


# ── Routes ──


@router.post("/run", response_model=BacktestRunResponse, status_code=202)
async def run_backtest(params: BacktestParams) -> BacktestRunResponse:
    run_id = _spawn_run(params)
    return BacktestRunResponse(run_id=run_id)


@router.get("/runs/{run_id}", response_model=BacktestRunStatus)
async def get_backtest_run(run_id: str) -> BacktestRunStatus:
    info = _get_run(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
    return BacktestRunStatus(**info)


@router.delete("/runs/{run_id}", response_model=BacktestCancelResponse)
async def cancel_backtest_run(run_id: str) -> BacktestCancelResponse:
    # Try to cancel first — successful cancel implies the run exists and was active.
    if _cancel_run(run_id):
        return BacktestCancelResponse(canceled=True)
    # Cancel failed: distinguish missing vs already-terminated.
    if _get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")
    raise HTTPException(status_code=409, detail="Backtest run already terminated")


@router.get("/sessions", response_model=BacktestSessionsList)
async def list_backtest_sessions() -> BacktestSessionsList:
    from cryptotrader.backtest import session as session_mod

    return BacktestSessionsList(sessions=session_mod.list_sessions())


@router.get("/sessions/{name}")
async def get_backtest_session(name: str) -> dict:
    loaded = _load_session(name)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"Session {name} not found")
    return loaded
