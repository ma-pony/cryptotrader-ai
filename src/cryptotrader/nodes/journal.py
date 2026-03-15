"""Journal commit nodes — record trades and rejections."""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

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
                data_points=v.get("data_points", {}),
                data_sufficiency=v.get("data_sufficiency", "medium"),
            )
        else:
            analyses[k] = v
    return analyses


async def _get_portfolio_snapshot(state: ArenaState) -> dict:
    """Get current portfolio state for journal commit."""
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        db_url = state["metadata"].get("database_url")
        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        return {
            "total_value": portfolio.get("total_value", 0),
            "positions": portfolio.get("positions", {}),
        }
    except Exception:
        logger.debug("Portfolio snapshot for journal failed", exc_info=True)
        return {}


@node_logger()
async def journal_trade(state: ArenaState) -> dict:
    """Journal a successful trade."""
    try:
        from cryptotrader.journal.commit import build_commit
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import GateResult, Order, TradeVerdict
        from cryptotrader.tracing import get_trace_id

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
            order = Order(**{k: v for k, v in raw_order.items() if k in Order.__dataclass_fields__})

        fill_price = state["data"].get("fill_price")
        slippage = state["data"].get("slippage")

        # Build portfolio_after snapshot
        portfolio_after = await _get_portfolio_snapshot(state)

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
            fill_price=fill_price,
            slippage=slippage,
            portfolio_after=portfolio_after,
            trace_id=get_trace_id(),
            consensus_metrics=state["data"].get("consensus_metrics"),
            verdict_source=state["data"].get("verdict", {}).get("verdict_source", "ai"),
            experience_memory=state["data"].get("experience_memory", {}),
            node_trace=state["data"].get("node_trace", []),
            debate_skip_reason=state["data"].get("debate_skip_reason", ""),
        )
        await store.commit(commit)
        logger.info(
            "Journal trade committed: hash=%s pair=%s action=%s",
            commit.hash,
            state["metadata"]["pair"],
            raw_verdict.get("action") if raw_verdict else "unknown",
        )
        return {"data": {"journal_hash": commit.hash}}
    except Exception:
        logger.error(
            "CRITICAL: journal commit failed after live trade for %s",
            state["metadata"].get("pair"),
            exc_info=True,
        )
        return {"data": {"journal_hash": None}}


@node_logger()
async def journal_rejection(state: ArenaState) -> dict:
    """Journal a risk-gate rejection."""
    try:
        from cryptotrader.journal.commit import build_commit
        from cryptotrader.journal.store import JournalStore
        from cryptotrader.models import GateResult, TradeVerdict
        from cryptotrader.nodes.verdict import _get_notifier
        from cryptotrader.tracing import get_trace_id

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
            trace_id=get_trace_id(),
            consensus_metrics=state["data"].get("consensus_metrics"),
            verdict_source=state["data"].get("verdict", {}).get("verdict_source", "ai"),
            experience_memory=state["data"].get("experience_memory", {}),
            node_trace=state["data"].get("node_trace", []),
            debate_skip_reason=state["data"].get("debate_skip_reason", ""),
        )
        await store.commit(commit)
        logger.info(
            "Journal rejection committed: hash=%s pair=%s rejected_by=%s",
            commit.hash,
            state["metadata"]["pair"],
            raw_gate.get("rejected_by") if raw_gate else "unknown",
        )

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
    except Exception:
        logger.error(
            "CRITICAL: journal commit failed for rejection of %s",
            state["metadata"].get("pair"),
            exc_info=True,
        )
        return {"data": {"journal_hash": None}}
