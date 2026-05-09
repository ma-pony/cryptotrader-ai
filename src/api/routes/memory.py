"""spec 018 — Memory API Routes（US-Z5）。

FR-Z40: GET /api/memory/rules — 返回 pattern rules 摘要列表
FR-Z41: GET /api/memory/cases — 返回 cases 摘要列表（含 IVE）
FR-Z42: GET /api/memory/transitions — 返回 FSM 状态转换事件
FR-Z43: GET /api/memory/archived — 返回 archived patterns 列表

Cache-Control: max-age=30 for rules/cases/transitions; max-age=300 for archived.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])

# 有效 maturity 值
_VALID_MATURITIES = {"observed", "probationary", "active", "deprecated", "archived"}
# 有效 agent IDs
_VALID_AGENTS = {"tech", "chain", "news", "macro"}

# memory root（从 EvolvingMemoryProvider 默认值一致）
_MEMORY_ROOT = Path("agent_memory")


# ── Response models ───────────────────────────────────────────────────────────


class PnLTrackOut(BaseModel):
    successes: int
    losses: int
    total_pnl: float


class RuleItem(BaseModel):
    name: str
    agent: str
    description: str
    maturity: str
    importance: float
    access_count: int
    last_accessed_at: str | None
    pnl_track: PnLTrackOut
    regime_tags: list[str]
    fundamental_failure_streak: int
    version: int
    manually_edited: bool


class RulesList(BaseModel):
    items: list[RuleItem]
    total: int


class TradeExecutionOut(BaseModel):
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    actual_exit_price: float | None = None
    fill_status: str | None = None
    hit_sl: bool | None = None


class IVEClassificationOut(BaseModel):
    failure_type: str
    confidence: float
    reasoning: str


class CaseItem(BaseModel):
    cycle_id: str
    timestamp: str
    pair: str
    verdict_action: str
    final_pnl: float | None
    trade_execution: TradeExecutionOut | None
    ive_classification: IVEClassificationOut | None


class CasesList(BaseModel):
    items: list[CaseItem]
    total: int


class TransitionItem(BaseModel):
    rule_id: str
    agent_id: str
    old_state: str
    new_state: str
    triggered_by: str
    timestamp: str


class TransitionsList(BaseModel):
    items: list[TransitionItem]
    total: int


class ArchivedRuleItem(BaseModel):
    name: str
    agent: str
    archived_at: str | None
    fundamental_failure_streak: int
    final_pnl_track: PnLTrackOut


class ArchivedList(BaseModel):
    items: list[ArchivedRuleItem]
    total: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_provider() -> Any:
    """加载 EvolvingMemoryProvider 实例（新建只读实例）。"""
    from cryptotrader.learning.evolution.provider import EvolvingMemoryProvider

    return EvolvingMemoryProvider(memory_root=_MEMORY_ROOT)


def _pnl_track_out(pnl_track: Any) -> PnLTrackOut:
    """把 PnLTrack dataclass 转为 response model。"""
    wins = getattr(pnl_track, "wins", 0)
    cases = getattr(pnl_track, "cases", 0)
    losses = max(0, cases - wins)
    avg_pnl = getattr(pnl_track, "avg_pnl", 0.0)
    total_pnl = avg_pnl * cases
    return PnLTrackOut(successes=wins, losses=losses, total_pnl=round(total_pnl, 4))


def _dt_str(dt: Any) -> str | None:
    """把 datetime 转为 ISO 字符串，None-safe。"""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    return str(dt)


def _build_trade_execution_out(trade_ex: dict | None) -> TradeExecutionOut | None:
    """把 trade_execution dict 转为 response model。"""
    if not trade_ex:
        return None
    return TradeExecutionOut(
        entry_price=trade_ex.get("entry_price"),
        stop_loss=trade_ex.get("stop_loss"),
        take_profit=trade_ex.get("take_profit"),
        actual_exit_price=trade_ex.get("actual_exit_price"),
        fill_status=trade_ex.get("fill_status"),
        hit_sl=trade_ex.get("hit_sl"),
    )


def _build_ive_out(ive: dict | None) -> IVEClassificationOut | None:
    """把 ive_classification dict 转为 response model。"""
    if not ive:
        return None
    return IVEClassificationOut(
        failure_type=ive.get("failure_type", "noise"),
        confidence=float(ive.get("confidence", 0.0)),
        reasoning=ive.get("reasoning", ""),
    )


def _parse_aware_dt(raw: str | None, default: datetime) -> datetime | None:
    """解析 ISO 字符串为 timezone-aware datetime，失败返回 None（由调用方返回 400）。"""
    if raw is None:
        return default
    dt = datetime.fromisoformat(raw)  # raises ValueError if invalid
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _case_passes_filters(case: Any, from_dt: datetime, to_dt: datetime, agent: str | None) -> bool:
    """检查 case 是否在时间窗口内且符合 agent 过滤。"""
    ts = case.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if not (from_dt <= ts <= to_dt):
        return False
    if agent is not None:
        refs = list(case.applied_patterns or [])
        if not any(ref.startswith(f"{agent}::") for ref in refs):
            return False
    return True


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/rules", response_model=RulesList)
async def get_memory_rules(
    agent: str | None = Query(default=None, description="tech/chain/news/macro"),
    status: str | None = Query(default=None, description="observed/probationary/active/deprecated/archived"),
) -> JSONResponse:
    """FR-Z40: 返回 pattern rules 摘要列表。"""
    # 参数验证
    if status is not None and status not in _VALID_MATURITIES:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": f"status must be one of {sorted(_VALID_MATURITIES)}"},
        )
    if agent is not None and agent not in _VALID_AGENTS:
        return JSONResponse(
            status_code=404,
            content={"error": "agent_not_found"},
        )

    try:
        from cryptotrader.learning.evolution.provider import _load_pattern_from_path

        agents_to_scan = [agent] if agent else list(_VALID_AGENTS)
        items: list[RuleItem] = []

        for agent_id in agents_to_scan:
            pattern_dir = _MEMORY_ROOT / agent_id / "patterns"
            if not pattern_dir.exists():
                continue
            for path in sorted(pattern_dir.glob("*.md")):
                rule = _load_pattern_from_path(path)
                if rule is None:
                    continue
                if status is not None and rule.maturity != status:
                    continue
                items.append(
                    RuleItem(
                        name=rule.name,
                        agent=rule.agent,
                        description=rule.description,
                        maturity=rule.maturity,
                        importance=rule.importance,
                        access_count=rule.access_count,
                        last_accessed_at=_dt_str(rule.last_accessed_at),
                        pnl_track=_pnl_track_out(rule.pnl_track),
                        regime_tags=list(rule.regime_tags or []),
                        fundamental_failure_streak=rule.fundamental_failure_streak,
                        version=rule.version,
                        manually_edited=rule.manually_edited,
                    )
                )

        response = RulesList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/rules failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/cases", response_model=CasesList)
async def get_memory_cases(
    from_: str | None = Query(default=None, alias="from", description="ISO8601 start time"),
    to: str | None = Query(default=None, description="ISO8601 end time"),
    agent: str | None = Query(default=None, description="filter by agent in analyses"),
) -> JSONResponse:
    """FR-Z41: 返回 cases 摘要列表（按 timestamp 倒序，含 IVE）。"""
    now = datetime.now(UTC)
    try:
        from_dt = _parse_aware_dt(from_, now - timedelta(days=7))
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'from' datetime format"},
        )
    try:
        to_dt = _parse_aware_dt(to, now)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'to' datetime format"},
        )

    try:
        from cryptotrader.learning.evolution.provider import _load_case_from_path

        cases_dir = _MEMORY_ROOT / "cases"
        items: list[CaseItem] = []

        if cases_dir.exists():
            for path in sorted(cases_dir.glob("*.md")):
                case = _load_case_from_path(path)
                if case is None:
                    continue
                if not _case_passes_filters(case, from_dt, to_dt, agent):
                    continue

                items.append(
                    CaseItem(
                        cycle_id=case.cycle_id,
                        timestamp=_dt_str(case.timestamp) or "",
                        pair=case.pair,
                        verdict_action=case.verdict_action,
                        final_pnl=case.final_pnl,
                        trade_execution=_build_trade_execution_out(case.trade_execution),
                        ive_classification=_build_ive_out(case.ive_classification),
                    )
                )

        # 按 timestamp 倒序
        items.sort(key=lambda c: c.timestamp, reverse=True)
        response = CasesList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/cases failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/transitions", response_model=TransitionsList)
async def get_memory_transitions(
    since: str | None = Query(default=None, description="ISO8601 start time; defaults to 24h ago"),
) -> JSONResponse:
    """FR-Z42: 返回 FSM 状态转换事件（从 provider 触发一次 evaluate_all_rules）。"""
    now = datetime.now(UTC)
    try:
        since_dt = datetime.fromisoformat(since) if since else now - timedelta(hours=24)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=UTC)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_query", "detail": "Invalid 'since' datetime format"},
        )

    try:
        provider = _load_provider()
        transitions = provider.evaluate_all_rules()

        items: list[TransitionItem] = []
        for t in transitions:
            ts_raw = getattr(t, "timestamp", now)
            ts_str = _dt_str(ts_raw) or now.isoformat()
            # filter by since
            try:
                ts_dt = datetime.fromisoformat(ts_str)
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
                if ts_dt < since_dt:
                    continue
            except (ValueError, TypeError):
                pass

            items.append(
                TransitionItem(
                    rule_id=f"{getattr(t, 'agent_id', '')}::{getattr(t, 'rule_id', '')}",
                    agent_id=getattr(t, "agent_id", ""),
                    old_state=getattr(t, "old_state", ""),
                    new_state=getattr(t, "new_state", ""),
                    triggered_by=getattr(t, "triggered_by", ""),
                    timestamp=ts_str,
                )
            )

        response = TransitionsList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=30"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/transitions failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )


@router.get("/archived", response_model=ArchivedList)
async def get_memory_archived() -> JSONResponse:
    """FR-Z43: 返回 archived patterns 列表（从 .archived/ 子目录读取）。"""
    try:
        from cryptotrader.learning.evolution.provider import _load_pattern_from_path

        items: list[ArchivedRuleItem] = []

        for agent_id in _VALID_AGENTS:
            archived_dir = _MEMORY_ROOT / agent_id / "patterns" / ".archived"
            if not archived_dir.exists():
                continue
            for path in sorted(archived_dir.glob("*.md")):
                rule = _load_pattern_from_path(path)
                if rule is None:
                    continue
                archived_at = _dt_str(rule.last_modified_at)
                items.append(
                    ArchivedRuleItem(
                        name=rule.name,
                        agent=rule.agent,
                        archived_at=archived_at,
                        fundamental_failure_streak=rule.fundamental_failure_streak,
                        final_pnl_track=_pnl_track_out(rule.pnl_track),
                    )
                )

        response = ArchivedList(items=items, total=len(items))
        return JSONResponse(
            content=response.model_dump(),
            headers={"Cache-Control": "max-age=300"},
        )

    except Exception as exc:
        logger.warning("GET /api/memory/archived failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "memory_io_error", "detail": str(exc)},
        )
