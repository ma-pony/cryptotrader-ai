"""Decisions list + detail endpoints (FR-803/FR-804)."""

from __future__ import annotations

import logging
import time as _time
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
    # Alignment with frontend prototype (2026-04-24):
    pnl: float | None = None
    debate_status: str = ""  # "skipped-consensus" | "skipped-confusion" | "1-round" | "2-round"
    reject_reason: str | None = None


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


class DebateTurnOut(BaseModel):
    """Single agent's utterance in a single round (matches frontend prototype)."""

    round: int
    from_agent: str = Field(alias="from")
    to_agent: str | None = Field(default=None, alias="to")
    before_direction: str
    before_confidence: float
    after_direction: str
    after_confidence: float
    move: str
    reasoning: str = ""
    new_findings: str = ""
    errored: bool = False

    model_config = {"populate_by_name": True}


class DebateGateOut(BaseModel):
    decision: str  # "debate" | "skipped-consensus" | "skipped-confusion"
    reason: str = ""
    strength: float = 0.0
    mean_score: float = 0.0
    dispersion: float = 0.0


class ConsensusMetricsOut(BaseModel):
    strength: float = 0.0
    mean_score: float = 0.0
    dispersion: float = 0.0
    skip_threshold: float = 0.5
    confusion_threshold: float = 0.05


class LatencyBreakdownOut(BaseModel):
    data_ms: float = 0.0
    agents_ms: float = 0.0
    debate_ms: float = 0.0
    verdict_ms: float = 0.0
    risk_ms: float = 0.0
    execute_ms: float = 0.0
    other_ms: float = 0.0
    total_ms: float = 0.0


class TokenUsageOut(BaseModel):
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cache_hits: float = 0.0
    calls: float = 0.0
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)


class AgentBiasOut(BaseModel):
    """One agent's 30-day bias profile (derived from journal.calibrate.detect_biases)."""

    agent_id: str
    accuracy: float
    neutral_rate: float
    bullish_rate: float
    bearish_rate: float
    avg_conf_when_right: float
    avg_conf_when_wrong: float
    sample_size: int
    warnings: list[str] = Field(default_factory=list)


class BiasOut(BaseModel):
    """Rolling bias snapshot at decision time.

    - ``agents``: per-agent bias stats over the last ``window_days``
    - ``summary``: human-readable one-liner for the Decision Detail hero
    - ``severity``: ``"low"|"medium"|"high"`` based on worst-agent warnings
    - ``window_days``: observation window used
    """

    agents: list[AgentBiasOut] = Field(default_factory=list)
    summary: str = ""
    severity: str = "low"
    window_days: int = 30


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
    # Alignment with frontend prototype (2026-04-24):
    debate_turns: list[DebateTurnOut] = Field(default_factory=list)
    debate_gate: DebateGateOut | None = None
    consensus_metrics: ConsensusMetricsOut | None = None
    latency_breakdown: LatencyBreakdownOut = Field(default_factory=LatencyBreakdownOut)
    token_usage: TokenUsageOut = Field(default_factory=TokenUsageOut)
    pnl: float | None = None
    retrospective: str | None = None
    debate_skip_reason: str = ""
    bias: BiasOut | None = None


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


def _debate_status(c: Any) -> str:
    """Derive the debate pipeline label for the list view."""
    skip = (getattr(c, "debate_skip_reason", "") or "").strip()
    rounds = int(getattr(c, "debate_rounds", 0) or 0)
    if skip == "consensus":
        return "skipped-consensus"
    if skip == "confusion":
        return "skipped-confusion"
    if rounds <= 0:
        return "skipped"
    if rounds == 1:
        return "1-round"
    return f"{rounds}-round"


def _commit_to_list_item(c: Any) -> DecisionListItem:
    snapshot = c.snapshot_summary or {}
    gate = getattr(c, "risk_gate", None)
    reject_reason: str | None = None
    if gate is not None and not getattr(gate, "passed", True):
        rejected_by = getattr(gate, "rejected_by", "") or ""
        reason = getattr(gate, "reason", "") or ""
        reject_reason = f"{rejected_by} · {reason}" if rejected_by and reason else (rejected_by or reason or None)
    return DecisionListItem(
        commit_hash=c.hash,
        ts=c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else str(c.timestamp),
        pair=c.pair,
        price=float(snapshot.get("price", 0.0) or 0.0),
        verdict=_verdict_to_slim(c.verdict),
        is_filled=bool(c.fill_price is not None and c.fill_price > 0) or bool(c.order),
        trace_id=c.trace_id,
        pnl=float(c.pnl) if getattr(c, "pnl", None) is not None else None,
        debate_status=_debate_status(c),
        reject_reason=reject_reason,
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
    """Legacy bull/bear rounds — kept for backwards compatibility.

    The prototype now prefers per-agent turns (see ``_serialize_turns``).
    """
    out = []
    for ch in challenges or []:
        if not isinstance(ch, dict):
            continue
        # Skip new-format turn entries (they have from/to/before/after keys).
        if "from" in ch and "before" in ch:
            continue
        out.append(
            DebateRoundOut(
                round=int(ch.get("round", 0) or 0),
                bull_message=ch.get("bull", "") or "",
                bear_message=ch.get("bear", "") or "",
            )
        )
    return out


def _serialize_turns(challenges: list) -> list[DebateTurnOut]:
    """Serialize the new per-agent turn entries populated by nodes/debate.py."""
    out: list[DebateTurnOut] = []
    for ch in challenges or []:
        if not isinstance(ch, dict):
            continue
        if not ("from" in ch and "before" in ch and "after" in ch):
            # Legacy / bull-bear format — skip.
            continue
        before = ch.get("before") or {}
        after = ch.get("after") or {}
        out.append(
            DebateTurnOut(
                round=int(ch.get("round", 0) or 0),
                **{"from": ch.get("from", ""), "to": ch.get("to")},
                before_direction=str(before.get("direction", "neutral")),
                before_confidence=float(before.get("confidence", 0.0) or 0.0),
                after_direction=str(after.get("direction", "neutral")),
                after_confidence=float(after.get("confidence", 0.0) or 0.0),
                move=str(ch.get("move", "保持")),
                reasoning=str(ch.get("reasoning", "") or ""),
                new_findings=str(ch.get("new_findings", "") or ""),
                errored=bool(ch.get("errored", False)),
            )
        )
    return out


def _serialize_debate_gate(c: Any) -> DebateGateOut:
    """Build the gate card for the Debate UI from commit observability fields."""
    cm = getattr(c, "consensus_metrics", None)
    strength = float(getattr(cm, "strength", 0.0) or 0.0) if cm is not None else 0.0
    mean_score = float(getattr(cm, "mean_score", 0.0) or 0.0) if cm is not None else 0.0
    dispersion = float(getattr(cm, "dispersion", 0.0) or 0.0) if cm is not None else 0.0
    skip = (getattr(c, "debate_skip_reason", "") or "").strip()
    if skip == "consensus":
        decision = "skipped-consensus"
        reason = f"strong consensus (strength={strength:.2f})"
    elif skip == "confusion":
        decision = "skipped-confusion"
        reason = f"shared confusion (|mean|={abs(mean_score):.2f}, dispersion={dispersion:.2f})"
    else:
        decision = "debate"
        reason = f"divergence triggered debate (dispersion={dispersion:.2f})"
    return DebateGateOut(
        decision=decision,
        reason=reason,
        strength=strength,
        mean_score=mean_score,
        dispersion=dispersion,
    )


def _serialize_consensus(cm: Any) -> ConsensusMetricsOut | None:
    if cm is None:
        return None
    return ConsensusMetricsOut(
        strength=float(getattr(cm, "strength", 0.0) or 0.0),
        mean_score=float(getattr(cm, "mean_score", 0.0) or 0.0),
        dispersion=float(getattr(cm, "dispersion", 0.0) or 0.0),
        skip_threshold=float(getattr(cm, "skip_threshold", 0.5) or 0.5),
        confusion_threshold=float(getattr(cm, "confusion_threshold", 0.05) or 0.05),
    )


def _serialize_latency(breakdown: Any) -> LatencyBreakdownOut:
    d = breakdown if isinstance(breakdown, dict) else {}

    def _f(k: str) -> float:
        v = d.get(k, 0.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    return LatencyBreakdownOut(
        data_ms=_f("data"),
        agents_ms=_f("agents"),
        debate_ms=_f("debate"),
        verdict_ms=_f("verdict"),
        risk_ms=_f("risk"),
        execute_ms=_f("execute"),
        other_ms=_f("other"),
        total_ms=_f("total"),
    )


# Module-level cache: bias stats are a 30-day rolling window — they don't change
# per-commit, so caching 60s shaves a 1000-row journal scan off every Decision Detail
# GET. Keyed by the store identity so test fixtures with fresh stores don't bleed.
_BIAS_CACHE_TTL = 60.0
_bias_cache: dict[int, tuple[float, BiasOut | None]] = {}


async def _build_bias(store: Any) -> BiasOut | None:
    """Compute 30-day rolling bias snapshot from journal data, cached 60s per store.

    Returns ``None`` when no agent has enough samples (journal.calibrate.detect_biases
    skips agents with ``<3`` settled commits).
    """
    from cryptotrader.journal.calibrate import _build_agent_warnings, detect_biases

    cache_key = id(store)
    now = _time.monotonic()
    cached = _bias_cache.get(cache_key)
    if cached and now - cached[0] < _BIAS_CACHE_TTL:
        return cached[1]

    try:
        stats = await detect_biases(store, days=30)
    except Exception:
        logger.debug("detect_biases failed", exc_info=True)
        _bias_cache[cache_key] = (now, None)
        return None
    if not stats:
        _bias_cache[cache_key] = (now, None)
        return None

    agents: list[AgentBiasOut] = []
    all_warnings: list[str] = []
    for agent_id, s in stats.items():
        warnings = _build_agent_warnings(s)
        all_warnings.extend(warnings)
        agents.append(
            AgentBiasOut(
                agent_id=agent_id,
                accuracy=float(s.get("accuracy", 0.0) or 0.0),
                neutral_rate=float(s.get("neutral_rate", 0.0) or 0.0),
                bullish_rate=float(s.get("bullish_rate", 0.0) or 0.0),
                bearish_rate=float(s.get("bearish_rate", 0.0) or 0.0),
                avg_conf_when_right=float(s.get("avg_conf_when_right", 0.0) or 0.0),
                avg_conf_when_wrong=float(s.get("avg_conf_when_wrong", 0.0) or 0.0),
                sample_size=int(s.get("sample_size", 0) or 0),
                warnings=warnings,
            )
        )

    # Severity heuristic: 0 warnings → low; 1-2 → medium; 3+ → high
    severity = "low"
    if len(all_warnings) >= 3:
        severity = "high"
    elif len(all_warnings) >= 1:
        severity = "medium"

    # Summary — pick the most bullish/bearish skewed agent as the representative lead.
    lead = max(agents, key=lambda a: abs(a.bullish_rate - a.bearish_rate), default=None)
    if lead is not None and abs(lead.bullish_rate - lead.bearish_rate) > 0.3:
        direction = "做多" if lead.bullish_rate > lead.bearish_rate else "做空"
        pct = round(max(lead.bullish_rate, lead.bearish_rate) * 100, 0)
        summary = f"过去 30 天 {lead.agent_id} {int(pct)}% {direction}倾向 · 注意确认偏差"
    elif all_warnings:
        summary = "检测到偏差: " + "; ".join(all_warnings[:2])
    else:
        summary = "过去 30 天无显著偏差"

    result = BiasOut(agents=agents, summary=summary, severity=severity, window_days=30)
    _bias_cache[cache_key] = (now, result)
    return result


def _serialize_tokens(usage: Any) -> TokenUsageOut:
    d = usage if isinstance(usage, dict) else {}

    def _f(k: str) -> float:
        v = d.get(k, 0.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    by_model_raw = d.get("by_model") or {}
    by_model: dict[str, dict[str, float]] = {}
    if isinstance(by_model_raw, dict):
        for model_name, stats in by_model_raw.items():
            if isinstance(stats, dict):
                by_model[model_name] = {str(k): float(v) for k, v in stats.items() if isinstance(v, int | float)}
    return TokenUsageOut(
        input_tokens=_f("input_tokens"),
        output_tokens=_f("output_tokens"),
        cache_hits=_f("cache_hits"),
        calls=_f("calls"),
        cost_usd=_f("cost_usd"),
        by_model=by_model,
    )


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
        # Entries may be plain dicts (JSONB roundtrip from Postgres / registry)
        # or dataclass instances (in-process callers). Support both.
        if isinstance(e, dict):
            node = e.get("node") or "unknown"
            duration = int(e.get("duration_ms") or 0)
        else:
            node = getattr(e, "node", None) or "unknown"
            duration = int(getattr(e, "duration_ms", 0) or 0)
        out.append(
            NodeTimelineEntryOut(
                node=node,
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
    # Defensive: from/to filters are not applied at store level; do it here.
    # Use coerce_timestamp from _utils to guarantee both sides are tz-aware before
    # compare — naive ISO strings (e.g. "2026-04-01") get promoted to UTC.
    if from_ or to:
        from api.routes._utils import coerce_timestamp

        from_dt = coerce_timestamp(from_) if from_ else None
        to_dt = coerce_timestamp(to) if to else None
        commits = [
            c
            for c in commits
            if (
                (from_dt is None or (coerce_timestamp(c.timestamp) or from_dt) >= from_dt)
                and (to_dt is None or (coerce_timestamp(c.timestamp) or to_dt) <= to_dt)
            )
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
        debate_turns=_serialize_turns(commit.challenges),
        debate_gate=_serialize_debate_gate(commit),
        consensus_metrics=_serialize_consensus(commit.consensus_metrics),
        latency_breakdown=_serialize_latency(getattr(commit, "latency_breakdown", {})),
        token_usage=_serialize_tokens(getattr(commit, "token_usage", {})),
        pnl=(float(commit.pnl) if commit.pnl is not None else None),
        retrospective=getattr(commit, "retrospective", None),
        debate_skip_reason=getattr(commit, "debate_skip_reason", "") or "",
        bias=await _build_bias(store),
    )
