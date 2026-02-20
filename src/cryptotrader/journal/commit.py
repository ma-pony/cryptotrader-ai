"""Build decision commits for the journal."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from cryptotrader.models import (
    AgentAnalysis,
    DecisionCommit,
    GateResult,
    Order,
    TradeVerdict,
)


def generate_hash(data: dict) -> str:
    """SHA256 of json-serialized data, return first 8 chars."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


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
) -> DecisionCommit:
    """Build a DecisionCommit with a generated hash."""
    h = generate_hash({"pair": pair, "summary": snapshot_summary, "parent": parent_hash})
    return DecisionCommit(
        hash=h,
        parent_hash=parent_hash,
        timestamp=datetime.utcnow(),
        pair=pair,
        snapshot_summary=snapshot_summary,
        analyses=analyses,
        debate_rounds=debate_rounds,
        divergence=divergence,
        verdict=verdict,
        risk_gate=risk_gate,
        order=order,
    )
