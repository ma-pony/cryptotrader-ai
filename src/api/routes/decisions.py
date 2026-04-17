"""Decisions list + detail endpoints (FR-803/FR-804)."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


# ── Response models ──


class VerdictSlim(BaseModel):
    action: str
    size: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    source: str = "ai"


class DecisionListItem(BaseModel):
    commit_hash: str
    ts: str
    pair: str
    price: float = 0.0
    verdict: VerdictSlim
    is_filled: bool = False
    trace_id: str | None = None


class PaginatedDecisions(BaseModel):
    items: list[DecisionListItem]
    total: int
    page: int
    size: int
    has_next: bool


class AgentAnalysisOut(BaseModel):
    name: str
    score: float
    confidence: float
    reasoning: str
    is_mock: bool = False


class DebateRoundOut(BaseModel):
    round: int
    bull_message: str
    bear_message: str


class RiskCheckOut(BaseModel):
    name: str
    passed: bool
    reason: str | None = None
    threshold: float | str | None = None


class RiskGateOut(BaseModel):
    passed: bool
    checks: list[RiskCheckOut] = Field(default_factory=list)


class ExecutionOut(BaseModel):
    order_id: str
    status: str
    fill_price: float = 0.0
    fill_size: float = 0.0
    fee: float = 0.0
    slippage_bps: float = 0.0
    exchange: str = "paper"


class NodeTimelineEntryOut(BaseModel):
    node: str
    start_ms: int
    duration_ms: int


class ExperienceMemoryRefOut(BaseModel):
    memory_id: str = ""
    success_patterns: list[dict] = Field(default_factory=list)
    forbidden_zones: list[dict] = Field(default_factory=list)
    strategic_insights: list[Any] = Field(default_factory=list)


class DecisionDetailOut(BaseModel):
    commit_hash: str
    ts: str
    pair: str
    price: float
    agent_analyses: list[AgentAnalysisOut]
    debate_rounds: list[DebateRoundOut]
    verdict: VerdictSlim
    risk_gate: RiskGateOut
    execution: ExecutionOut | None
    node_timeline: list[NodeTimelineEntryOut]
    experience_memory_ref: ExperienceMemoryRefOut
    trace_id: str | None = None


def _verdict_to_slim(v: Any) -> VerdictSlim:
    if v is None:
        return VerdictSlim(action="hold")
    return VerdictSlim(
        action=getattr(v, "action", "hold"),
        size=float(getattr(v, "position_scale", 0.0) or 0.0),
        confidence=float(getattr(v, "confidence", 0.0) or 0.0),
        reasoning=getattr(v, "reasoning", "") or "",
        source=getattr(v, "verdict_source", "ai") or "ai",
    )


def _commit_to_list_item(c: Any) -> DecisionListItem:
    snapshot = c.snapshot_summary or {}
    return DecisionListItem(
        commit_hash=c.hash,
        ts=c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else str(c.timestamp),
        pair=c.pair,
        price=float(snapshot.get("price", 0.0) or 0.0),
        verdict=_verdict_to_slim(c.verdict),
        is_filled=bool(c.fill_price is not None and c.fill_price > 0) or bool(c.order),
        trace_id=c.trace_id,
    )


def _serialize_analyses(analyses: dict) -> list[AgentAnalysisOut]:
    out = []
    for name, a in (analyses or {}).items():
        direction = getattr(a, "direction", "neutral")
        # Map direction (string) → numeric score for the frontend
        score = {"bullish": 0.6, "bearish": -0.6, "neutral": 0.0}.get(direction, 0.0)
        out.append(
            AgentAnalysisOut(
                name=name,
                score=score,
                confidence=float(getattr(a, "confidence", 0.0) or 0.0),
                reasoning=getattr(a, "reasoning", "") or "",
                is_mock=bool(getattr(a, "is_mock", False)),
            )
        )
    return out


def _serialize_challenges(challenges: list) -> list[DebateRoundOut]:
    out = []
    for ch in challenges or []:
        if not isinstance(ch, dict):
            continue
        out.append(
            DebateRoundOut(
                round=int(ch.get("round", 0) or 0),
                bull_message=ch.get("bull", "") or "",
                bear_message=ch.get("bear", "") or "",
            )
        )
    return out


def _serialize_risk_gate(gate: Any) -> RiskGateOut:
    if gate is None:
        return RiskGateOut(passed=False, checks=[])
    checks: list[RiskCheckOut] = []
    rejected_by = getattr(gate, "rejected_by", "") or ""
    reason = getattr(gate, "reason", "") or ""
    if rejected_by:
        checks.append(RiskCheckOut(name=rejected_by, passed=False, reason=reason or None))
    return RiskGateOut(passed=bool(gate.passed), checks=checks)


def _serialize_execution(commit: Any) -> ExecutionOut | None:
    order = getattr(commit, "order", None)
    if order is None:
        return None
    return ExecutionOut(
        order_id=getattr(order, "exchange_id", "") or order.__class__.__name__,
        status=str(getattr(order, "status", "")),
        fill_price=float(commit.fill_price or order.price or 0.0),
        fill_size=float(getattr(order, "amount", 0.0) or 0.0),
        slippage_bps=float((commit.slippage or 0.0) * 10000),
        exchange=getattr(order, "exchange_id", "") or "paper",
    )


def _serialize_node_trace(entries: list) -> list[NodeTimelineEntryOut]:
    out: list[NodeTimelineEntryOut] = []
    cumulative = 0
    for e in entries or []:
        duration = int(getattr(e, "duration_ms", 0) or 0)
        out.append(
            NodeTimelineEntryOut(
                node=getattr(e, "node", "unknown"),
                start_ms=cumulative,
                duration_ms=duration,
            )
        )
        cumulative += duration
    return out


def _serialize_experience(em: Any) -> ExperienceMemoryRefOut:
    if not em:
        return ExperienceMemoryRefOut()
    if isinstance(em, dict):
        data = em
    elif is_dataclass(em) and not isinstance(em, type):
        data = asdict(cast("Any", em))
    else:
        data = {}

    def _to_dict(rule: Any) -> dict:
        if isinstance(rule, dict):
            return rule
        if is_dataclass(rule) and not isinstance(rule, type):
            return asdict(cast("Any", rule))
        return {}

    return ExperienceMemoryRefOut(
        memory_id=str(data.get("memory_id", "") or ""),
        success_patterns=[_to_dict(r) for r in data.get("success_patterns", []) or []],
        forbidden_zones=[_to_dict(r) for r in data.get("forbidden_zones", []) or []],
        strategic_insights=list(data.get("strategic_insights", []) or []),
    )


# ── Routes ──


@router.get("", response_model=PaginatedDecisions)
async def list_decisions(
    pair: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
) -> PaginatedDecisions:
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import JournalStore

    config = load_config()
    store = JournalStore(config.infrastructure.database_url)

    # JournalStore.log returns most-recent-first up to `limit`. We oversample
    # by `page * size + 1` so we can detect whether more rows exist.
    limit = page * size + 1
    commits = await store.log(limit=limit, pair=pair)
    # Defensive: from/to filters are not applied at store level; do it here
    if from_ or to:
        from datetime import datetime as _dt

        def _parse(s: str) -> _dt:
            return _dt.fromisoformat(s.replace("Z", "+00:00"))

        commits = [
            c for c in commits if (not from_ or c.timestamp >= _parse(from_)) and (not to or c.timestamp <= _parse(to))
        ]

    offset = (page - 1) * size
    page_commits = commits[offset : offset + size]
    # Full page is the cue for "more may exist"; short page is the last page.
    has_next = len(page_commits) == size

    return PaginatedDecisions(
        items=[_commit_to_list_item(c) for c in page_commits],
        total=offset + len(page_commits),
        page=page,
        size=size,
        has_next=has_next,
    )


@router.get("/{commit_hash}", response_model=DecisionDetailOut)
async def get_decision(commit_hash: str) -> DecisionDetailOut:
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import JournalStore

    config = load_config()
    store = JournalStore(config.infrastructure.database_url)
    commit = await store.show(commit_hash)
    if commit is None:
        raise HTTPException(status_code=404, detail=f"Commit {commit_hash} not found")

    snapshot = commit.snapshot_summary or {}
    return DecisionDetailOut(
        commit_hash=commit.hash,
        ts=commit.timestamp.isoformat() if hasattr(commit.timestamp, "isoformat") else str(commit.timestamp),
        pair=commit.pair,
        price=float(snapshot.get("price", 0.0) or 0.0),
        agent_analyses=_serialize_analyses(commit.analyses),
        debate_rounds=_serialize_challenges(commit.challenges),
        verdict=_verdict_to_slim(commit.verdict),
        risk_gate=_serialize_risk_gate(commit.risk_gate),
        execution=_serialize_execution(commit),
        node_timeline=_serialize_node_trace(commit.node_trace),
        experience_memory_ref=_serialize_experience(commit.experience_memory),
        trace_id=commit.trace_id,
    )
