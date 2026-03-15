"""Supervisor pattern graph builder following LangChain official docs.

EXPERIMENTAL -- not enabled in the main trading path.
This module implements an alternative supervisor-pattern graph (build_supervisor_graph)
as a research prototype.  The production pipeline uses build_trading_graph() from
graph.py, which runs agents in parallel and applies the debate gate/verdict/risk flow.
This module is kept for comparison and future experimentation only.

This implements the supervisor pattern from:
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from cryptotrader.agents.langchain_agents import create_supervisor_agent
from cryptotrader.graph import ArenaState


async def supervisor_analyze(state: ArenaState) -> dict:
    """Supervisor coordinates specialized analysts to form trading decision."""
    snapshot = state["data"]["snapshot"]
    model = state["metadata"].get("verdict_model", "")

    # Build context for supervisor
    context = f"""Analyze {snapshot.pair} for trading decision.

Market Data:
- Price: ${snapshot.market.ticker.get("last", 0):,.2f}
- Volatility: {snapshot.market.volatility:.4f}
- Volume: {snapshot.market.ticker.get("baseVolume", 0):,.0f}
- Funding Rate: {snapshot.market.funding_rate:.4f}%
- Orderbook Imbalance: {snapshot.market.orderbook_imbalance:.2f}

On-chain Data:
- Open Interest: {snapshot.onchain.open_interest:,.0f}
- Liquidations 24h: {snapshot.onchain.liquidations_24h}

Macro Data:
- Fear & Greed Index: {snapshot.macro.fear_greed_index}
- BTC Dominance: {snapshot.macro.btc_dominance:.2f}%

Coordinate your analysts to determine: long, short, or hold?
"""

    supervisor = create_supervisor_agent(model)
    result = await supervisor.ainvoke({"messages": [{"role": "user", "content": context}]})

    # Extract verdict from supervisor's final message
    final_message = result["messages"][-1].content

    from cryptotrader.debate.verdict import _extract_json

    try:
        verdict = _extract_json(final_message)
    except (ValueError, Exception):
        verdict = {
            "action": "hold",
            "confidence": 0.5,
            "reasoning": final_message,
            "position_scale": 0.5,
        }

    # Ensure all required fields for downstream nodes
    verdict.setdefault("action", "hold")
    verdict.setdefault("confidence", 0.5)
    verdict.setdefault("position_scale", 0.5)
    verdict.setdefault("reasoning", "")
    verdict.setdefault("divergence", 0.0)
    verdict.setdefault("thesis", "")
    verdict.setdefault("invalidation", "")

    return {"data": {"verdict": verdict}}


def build_supervisor_graph() -> Any:
    """Build trading graph using supervisor pattern.

    This is an alternative to build_trading_graph() that uses LangChain's
    official supervisor pattern instead of fan-out parallel execution.

    Trade-offs:
    - Slower (serial execution) but more coordinated
    - Uses progressive disclosure (loads skills/experience on-demand)
    - Better for complex analysis, worse for high-frequency backtesting
    """
    from cryptotrader.graph import collect_snapshot

    graph = StateGraph(ArenaState)

    # 1. Collect market data
    graph.add_node("collect_snapshot", collect_snapshot)

    # 2. Supervisor coordinates analysts
    graph.add_node("supervisor_analyze", supervisor_analyze)

    # 3. Risk gate and execution (reuse existing nodes)
    from cryptotrader.graph import journal_rejection, journal_trade, place_order, risk_check, risk_router

    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_trade", journal_trade)
    graph.add_node("record_rejection", journal_rejection)

    # Flow
    graph.add_edge(START, "collect_snapshot")
    graph.add_edge("collect_snapshot", "supervisor_analyze")
    graph.add_edge("supervisor_analyze", "risk_gate")
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
