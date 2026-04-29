"""Build decision commits for the journal."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from cryptotrader._compat import UTC
from cryptotrader.models import (
    AgentAnalysis,
    CommitObservability,
    DecisionCommit,
    GateResult,
    NodeTraceEntry,
    Order,
    TradeVerdict,
)


def generate_hash(data: dict) -> str:
    """SHA256 of json-serialized data, return first 16 chars."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_commit(
    pair: str,
    snapshot_summary: dict[str, Any],
    analyses: dict[str, AgentAnalysis],
    debate_rounds: int,
    divergence: float,
    verdict: TradeVerdict | None,
    risk_gate: GateResult | None,
    order: Order | None,
    parent_hash: str | None,
    fill_price: float | None = None,
    slippage: float | None = None,
    portfolio_after: dict[str, Any] | None = None,
    challenges: list[dict] | None = None,
    *,
    observability: CommitObservability | None = None,
    # Legacy per-field kwargs (deprecated — prefer ``observability``):
    trace_id: str | None = None,
    consensus_metrics: dict[str, Any] | None = None,
    verdict_source: str = "ai",
    experience_memory: dict[str, Any] | None = None,
    node_trace: list[NodeTraceEntry] | None = None,
    debate_skip_reason: str = "",
    latency_breakdown: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> DecisionCommit:
    """Build a DecisionCommit with a generated hash.

    Callers should prefer passing a single ``observability`` :class:`CommitObservability`
    instance — the 8 per-field kwargs are kept for backwards compatibility with
    existing call sites but will be removed in a future refactor.
    """
    # Consolidate observability: bundle takes priority when both are supplied, but
    # individual kwargs override bundle defaults when bundle field is None/empty.
    obs = observability or CommitObservability()
    trace_id = trace_id if trace_id is not None else obs.trace_id
    consensus_metrics = consensus_metrics if consensus_metrics is not None else obs.consensus_metrics
    verdict_source = verdict_source if verdict_source != "ai" else obs.verdict_source
    experience_memory = experience_memory if experience_memory is not None else obs.experience_memory
    node_trace = node_trace if node_trace is not None else obs.node_trace
    debate_skip_reason = debate_skip_reason or obs.debate_skip_reason
    latency_breakdown = latency_breakdown if latency_breakdown is not None else obs.latency_breakdown
    token_usage = token_usage if token_usage is not None else obs.token_usage

    now = datetime.now(UTC)
    verdict_action = verdict.action if verdict else "none"
    h = generate_hash(
        {
            "pair": pair,
            "summary": snapshot_summary,
            "parent": parent_hash,
            "ts": now.isoformat(),
            "verdict": verdict_action,
        }
    )
    from cryptotrader.models import ConsensusMetrics

    cm: ConsensusMetrics | None = None
    if consensus_metrics is not None and isinstance(consensus_metrics, dict):
        cm = ConsensusMetrics(
            strength=consensus_metrics.get("strength", 0.0),
            mean_score=consensus_metrics.get("mean_score", 0.0),
            dispersion=consensus_metrics.get("dispersion", 0.0),
            skip_threshold=consensus_metrics.get("skip_threshold", 0.5),
            confusion_threshold=consensus_metrics.get("confusion_threshold", 0.05),
        )
    elif isinstance(consensus_metrics, ConsensusMetrics):
        cm = consensus_metrics

    return DecisionCommit(
        hash=h,
        parent_hash=parent_hash,
        timestamp=now,
        pair=pair,
        snapshot_summary=snapshot_summary,
        analyses=analyses,
        debate_rounds=debate_rounds,
        challenges=challenges or [],
        divergence=divergence,
        verdict=verdict,
        risk_gate=risk_gate,
        order=order,
        fill_price=fill_price,
        slippage=slippage,
        portfolio_after=portfolio_after or {},
        trace_id=trace_id,
        consensus_metrics=cm,
        verdict_source=verdict_source,  # type: ignore[arg-type]
        experience_memory=experience_memory or {},
        node_trace=node_trace or [],
        debate_skip_reason=debate_skip_reason,
        latency_breakdown=latency_breakdown or {},
        token_usage=token_usage or {},
    )
