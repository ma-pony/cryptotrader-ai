"""Journal commit nodes — record trades and rejections."""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)


def _to_agent_analyses(raw_analyses: dict, pair: str) -> dict:
    """Convert raw analysis dicts from state to AgentAnalysis objects."""
    from cryptotrader.models import AgentAnalysis

    analyses = {}
    for k, v in raw_analyses.items():
        if isinstance(v, dict):
            analyses[k] = AgentAnalysis(
                agent_id=k,
                pair=pair,
                direction=v.get("direction", "neutral"),
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
                key_factors=v.get("key_factors", []),
                risk_flags=v.get("risk_flags", []),
            )
        else:
            analyses[k] = v
    return analyses


async def journal_trade(state: ArenaState) -> dict:
    """Journal a successful trade."""
    from cryptotrader.journal.commit import build_commit
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import GateResult, Order, TradeVerdict

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)

    analyses = _to_agent_analyses(state["data"].get("analyses", {}), state["metadata"]["pair"])

    raw_verdict = state["data"].get("verdict")
    verdict = None
    if raw_verdict and isinstance(raw_verdict, dict):
        verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

    raw_gate = state["data"].get("risk_gate")
    risk_gate = None
    if raw_gate and isinstance(raw_gate, dict):
        risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

    raw_order = state["data"].get("order")
    order = None
    if raw_order and isinstance(raw_order, dict):
        order = Order(
            pair=raw_order.get("pair", ""),
            side=raw_order.get("side", "buy"),
            amount=raw_order.get("amount", 0),
            price=raw_order.get("price", 0),
        )

    parent_hash = state["data"].get("journal_hash")
    commit = build_commit(
        pair=state["metadata"]["pair"],
        snapshot_summary=state["data"].get("snapshot_summary", {}),
        analyses=analyses,
        debate_rounds=state.get("debate_round", 0),
        divergence=(state.get("divergence_scores") or [0.0])[-1],
        verdict=verdict,
        risk_gate=risk_gate,
        order=order,
        parent_hash=parent_hash,
    )
    await store.commit(commit)
    return {"data": {"journal_hash": commit.hash}}


async def journal_rejection(state: ArenaState) -> dict:
    """Journal a risk-gate rejection."""
    from cryptotrader.journal.commit import build_commit
    from cryptotrader.journal.store import JournalStore
    from cryptotrader.models import GateResult, TradeVerdict
    from cryptotrader.nodes.verdict import _get_notifier

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)

    analyses = _to_agent_analyses(state["data"].get("analyses", {}), state["metadata"]["pair"])

    raw_verdict = state["data"].get("verdict")
    verdict = None
    if raw_verdict and isinstance(raw_verdict, dict):
        verdict = TradeVerdict(**{k: v for k, v in raw_verdict.items() if k in TradeVerdict.__dataclass_fields__})

    raw_gate = state["data"].get("risk_gate")
    risk_gate = None
    if raw_gate and isinstance(raw_gate, dict):
        risk_gate = GateResult(**{k: v for k, v in raw_gate.items() if k in GateResult.__dataclass_fields__})

    parent_hash = state["data"].get("journal_hash")
    commit = build_commit(
        pair=state["metadata"]["pair"],
        snapshot_summary=state["data"].get("snapshot_summary", {}),
        analyses=analyses,
        debate_rounds=state.get("debate_round", 0),
        divergence=(state.get("divergence_scores") or [0.0])[-1],
        verdict=verdict,
        risk_gate=risk_gate,
        order=None,
        parent_hash=parent_hash,
    )
    await store.commit(commit)

    # Fire-and-forget rejection notification
    try:
        notifier = _get_notifier(state)
        raw_gate = state["data"].get("risk_gate", {})
        await notifier.notify(
            "rejection",
            {
                "pair": state["metadata"]["pair"],
                "rejected_by": raw_gate.get("rejected_by"),
                "reason": raw_gate.get("reason"),
            },
        )
    except Exception:
        logger.debug("Rejection notification failed", exc_info=True)

    return {"data": {"journal_hash": commit.hash}}
