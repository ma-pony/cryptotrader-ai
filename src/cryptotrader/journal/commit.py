"""Build decision commits for the journal."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from cryptotrader.models import (
    AgentAnalysis,
    DecisionCommit,
    GateResult,
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
    trace_id: str | None = None,
) -> DecisionCommit:
    """Build a DecisionCommit with a generated hash."""
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
    )
