"""LangGraph main orchestration — build_trading_graph() per ARCHITECTURE.md section 3.2.

Node functions live in cryptotrader.nodes.* for maintainability.
This file defines ArenaState and graph builder functions.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

# ── Re-export node functions for backward compatibility ──
# These are used by graph_supervisor.py, tests, and external callers.
from cryptotrader.nodes.agents import (
    chain_analyze,
    macro_analyze,
    news_analyze,
    tech_analyze,
)
from cryptotrader.nodes.data import collect_snapshot, update_past_pnl, verbal_reinforcement
from cryptotrader.nodes.debate import (
    bull_bear_debate,
    check_stability,
    convergence_router,
    debate_round,
    judge_verdict,
)
from cryptotrader.nodes.execution import check_stop_loss, place_order
from cryptotrader.nodes.journal import journal_rejection, journal_trade
from cryptotrader.nodes.verdict import _risk_gate_cache, make_verdict, risk_check, risk_router
from cryptotrader.state import ArenaState, merge_dicts

__all__ = [
    "ArenaState",
    "_risk_gate_cache",
    "build_debate_graph",
    "build_lite_graph",
    "build_trading_graph",
    "bull_bear_debate",
    "chain_analyze",
    "check_stability",
    "check_stop_loss",
    "collect_snapshot",
    "convergence_router",
    "debate_round",
    "journal_rejection",
    "journal_trade",
    "judge_verdict",
    "macro_analyze",
    "make_verdict",
    "merge_dicts",
    "news_analyze",
    "place_order",
    "risk_check",
    "risk_router",
    "tech_analyze",
    "update_past_pnl",
    "verbal_reinforcement",
]


# ── Graph builders ──


def build_trading_graph(config: dict | None = None) -> Any:
    return _build_full_graph(config)


def build_lite_graph(config: dict | None = None) -> Any:
    """Lightweight graph for backtesting: skip debate, go straight to verdict."""
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("verdict", make_verdict)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    graph.add_edge("tech_agent", "verdict")
    graph.add_edge("chain_agent", "verdict")
    graph.add_edge("news_agent", "verdict")
    graph.add_edge("macro_agent", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile()


def build_debate_graph(config: dict | None = None) -> Any:
    """Lite graph + bull/bear adversarial debate before verdict."""
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("debate", bull_bear_debate)
    graph.add_node("verdict", judge_verdict)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    graph.add_edge("tech_agent", "debate")
    graph.add_edge("chain_agent", "debate")
    graph.add_edge("news_agent", "debate")
    graph.add_edge("macro_agent", "debate")
    graph.add_edge("debate", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile()


def _stop_loss_router(state: ArenaState) -> str:
    """Route to exit if stop-loss was triggered, otherwise continue normal flow."""
    if state.get("data", {}).get("stop_loss_triggered"):
        return "exit_position"
    return "continue"


def _build_full_graph(config: dict | None = None) -> Any:
    """Full pipeline: agents → 2 fixed debate rounds → AI verdict → risk gate → execute.

    Phase 4C: Fixed 2 debate rounds instead of convergence-seeking loop.
    Agents are encouraged to maintain disagreement when data supports it.
    """
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("update_pnl", update_past_pnl)
    graph.add_node("stop_loss_check", check_stop_loss)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("debate_round_1", debate_round)
    graph.add_node("debate_round_2", debate_round)
    graph.add_node("verdict", make_verdict)
    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_trade", journal_trade)
    graph.add_node("record_rejection", journal_rejection)

    graph.add_edge(START, "collect_data")
    # After data collection: update PnL for past trades + check stop-loss
    graph.add_edge("collect_data", "update_pnl")
    graph.add_edge("update_pnl", "stop_loss_check")
    # Stop-loss router: if triggered, skip analysis and go straight to risk gate
    graph.add_conditional_edges(
        "stop_loss_check",
        _stop_loss_router,
        {
            "continue": "inject_experience",
            "exit_position": "risk_gate",
        },
    )
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    # Fan-in to first debate round
    graph.add_edge("tech_agent", "debate_round_1")
    graph.add_edge("chain_agent", "debate_round_1")
    graph.add_edge("news_agent", "debate_round_1")
    graph.add_edge("macro_agent", "debate_round_1")
    # Fixed 2 rounds — no convergence-seeking
    graph.add_edge("debate_round_1", "debate_round_2")
    graph.add_edge("debate_round_2", "verdict")
    graph.add_edge("verdict", "risk_gate")
    graph.add_conditional_edges(
        "risk_gate",
        risk_router,
        {
            "approved": "execute",
            "rejected": "record_rejection",
        },
    )
    graph.add_edge("execute", "record_trade")
    graph.add_edge("record_trade", END)
    graph.add_edge("record_rejection", END)

    return graph.compile()


def build_supervisor_graph_v2() -> Any:
    """Build graph using LangChain official supervisor pattern."""
    from cryptotrader.graph_supervisor import build_supervisor_graph

    return build_supervisor_graph()
