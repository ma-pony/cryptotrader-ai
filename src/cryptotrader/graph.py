"""LangGraph main orchestration — build_trading_graph() per ARCHITECTURE.md section 3.2."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import operator


def merge_dicts(a: dict, b: dict) -> dict:
    result = {**a}
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


class ArenaState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
    debate_round: int
    max_debate_rounds: int
    divergence_scores: list[float]


# ── Node functions ──

async def collect_snapshot(state: ArenaState) -> dict:
    # Skip collection if snapshot already provided (e.g. backtest)
    if state.get("data", {}).get("snapshot"):
        snapshot = state["data"]["snapshot"]
        summary = {
            "pair": snapshot.pair,
            "price": snapshot.market.ticker.get("last", 0),
            "funding_rate": snapshot.market.funding_rate,
            "volatility": snapshot.market.volatility,
            "orderbook_imbalance": snapshot.market.orderbook_imbalance,
        }
        return {"data": {"snapshot_summary": summary}}

    from cryptotrader.data.snapshot import SnapshotAggregator

    pair = state["metadata"]["pair"]
    exchange_id = state["metadata"].get("exchange_id", "binance")
    timeframe = state["metadata"].get("timeframe", "1h")
    limit = state["metadata"].get("ohlcv_limit", 100)

    agg = SnapshotAggregator()
    snapshot = await agg.collect(pair, exchange_id, timeframe, limit)
    summary = {
        "pair": pair,
        "price": snapshot.market.ticker.get("last", 0),
        "funding_rate": snapshot.market.funding_rate,
        "volatility": snapshot.market.volatility,
        "orderbook_imbalance": snapshot.market.orderbook_imbalance,
    }
    return {"data": {"snapshot": snapshot, "snapshot_summary": summary}}


async def verbal_reinforcement(state: ArenaState) -> dict:
    from cryptotrader.learning.verbal import get_experience
    from cryptotrader.journal.store import JournalStore

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    summary = state["data"].get("snapshot_summary", {})
    experience = await get_experience(store, summary)
    return {"data": {"experience": experience}}


async def _run_agent(agent_type: str, state: ArenaState) -> dict:
    from cryptotrader.agents.tech import TechAgent
    from cryptotrader.agents.chain import ChainAgent
    from cryptotrader.agents.news import NewsAgent
    from cryptotrader.agents.macro import MacroAgent

    agents = {
        "tech_agent": lambda m: TechAgent(model=m),
        "chain_agent": lambda m: ChainAgent(model=m),
        "news_agent": lambda m: NewsAgent(model=m),
        "macro_agent": lambda m: MacroAgent(model=m),
    }
    # Per-agent model: metadata.models.tech_agent, fallback to analysis_model
    models_cfg = state["metadata"].get("models", {})
    model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", "gpt-4o-mini"))
    agent = agents[agent_type](model)
    snapshot = state["data"]["snapshot"]
    experience = state["data"].get("experience", "")
    analysis = await agent.analyze(snapshot, experience)
    result = {
        "direction": analysis.direction,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "key_factors": analysis.key_factors,
        "risk_flags": analysis.risk_flags,
    }
    # Pass through all extra fields from data_points (regime, strength, crowding, etc.)
    result.update(analysis.data_points)
    return {"data": {"analyses": {agent_type: result}}}


async def tech_analyze(state: ArenaState) -> dict:
    return await _run_agent("tech_agent", state)

async def chain_analyze(state: ArenaState) -> dict:
    return await _run_agent("chain_agent", state)

async def news_analyze(state: ArenaState) -> dict:
    return await _run_agent("news_agent", state)

async def macro_analyze(state: ArenaState) -> dict:
    return await _run_agent("macro_agent", state)


async def debate_round(state: ArenaState) -> dict:
    from cryptotrader.debate.challenge import build_challenge_prompt
    import litellm

    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    updated: dict[str, Any] = {}

    for agent_id, analysis in analyses.items():
        others = {k: v for k, v in analyses.items() if k != agent_id}
        prompt = build_challenge_prompt(agent_id, state["metadata"]["pair"], analysis, others)
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            updated[agent_id] = {
                "direction": data.get("direction", analysis["direction"]),
                "confidence": float(data.get("confidence", analysis["confidence"])),
                "reasoning": data.get("reasoning", analysis["reasoning"]),
                "key_factors": data.get("key_factors", analysis.get("key_factors", [])),
                "risk_flags": data.get("risk_flags", analysis.get("risk_flags", [])),
            }
        except Exception:
            updated[agent_id] = analysis

    return {
        "data": {"analyses": updated},
        "debate_round": state["debate_round"] + 1,
    }


async def check_stability(state: ArenaState) -> dict:
    from cryptotrader.debate.convergence import compute_divergence

    analyses = state["data"].get("analyses", {})
    divergence = compute_divergence(analyses)
    scores = list(state.get("divergence_scores") or [])
    scores.append(divergence)
    return {"divergence_scores": scores}


def convergence_router(state: ArenaState) -> str:
    from cryptotrader.debate.convergence import check_convergence

    scores = state.get("divergence_scores") or []
    if state["debate_round"] >= state["max_debate_rounds"]:
        return "converged"
    if len(scores) >= 2 and check_convergence(scores[:-1], scores[-1]):
        return "converged"
    return "continue"


async def make_verdict(state: ArenaState) -> dict:
    from cryptotrader.debate.verdict import make_verdict_llm, make_verdict_weighted

    analyses = state["data"].get("analyses", {})
    use_llm_verdict = state["metadata"].get("llm_verdict", True)

    if use_llm_verdict:
        model = state["metadata"].get("verdict_model",
                    state["metadata"].get("debate_model", "gpt-4o-mini"))
        verdict = await make_verdict_llm(analyses, model=model)
    else:
        scores = state.get("divergence_scores") or [0.0]
        threshold = state["metadata"].get("divergence_hold_threshold", 0.7)
        verdict = make_verdict_weighted(analyses, scores[-1], threshold)

    return {"data": {"verdict": {
        "action": verdict.action,
        "confidence": verdict.confidence,
        "position_scale": verdict.position_scale,
        "divergence": verdict.divergence,
        "reasoning": verdict.reasoning,
    }}}


async def risk_check(state: ArenaState) -> dict:
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.config import load_config
    from cryptotrader.models import TradeVerdict

    config = load_config()
    redis_state = RedisStateManager(state["metadata"].get("redis_url"))
    gate = RiskGate(config.risk, redis_state)

    vd = state["data"]["verdict"]
    verdict = TradeVerdict(**vd)
    portfolio = state["data"].get("portfolio", {
        "total_value": 10000,
        "positions": {},
        "daily_pnl": 0.0,
        "drawdown": 0.0,
        "returns_60d": [],
        "recent_prices": [],
        "funding_rate": state["data"].get("snapshot_summary", {}).get("funding_rate", 0),
        "api_latency_ms": 100,
    })
    result = await gate.check(verdict, portfolio)
    return {"data": {"risk_gate": {
        "passed": result.passed,
        "rejected_by": result.rejected_by,
        "reason": result.reason,
    }}}


def risk_router(state: ArenaState) -> str:
    rg = state["data"].get("risk_gate", {})
    return "approved" if rg.get("passed", False) else "rejected"


async def place_order(state: ArenaState) -> dict:
    from cryptotrader.execution.simulator import PaperExchange
    from cryptotrader.execution.exchange import LiveExchange
    from cryptotrader.models import Order

    verdict = state["data"]["verdict"]
    if verdict["action"] == "hold":
        return {"data": {"order": None}}

    pair = state["metadata"]["pair"]
    price = state["data"].get("snapshot_summary", {}).get("price", 0)
    scale = verdict.get("position_scale", 1.0)
    total = state["data"].get("portfolio", {}).get("total_value", 10000)
    amount = (total * 0.1 * scale) / max(price, 1)

    order = Order(
        pair=pair,
        side="buy" if verdict["action"] == "long" else "sell",
        amount=amount,
        price=price,
    )

    engine = state["metadata"].get("engine", "paper")
    if engine == "paper":
        exchange = PaperExchange()
    else:
        cfg = state["metadata"].get("exchange_config", {})
        exchange = LiveExchange(
            cfg.get("exchange_id", "binance"),
            cfg.get("api_key", ""),
            cfg.get("secret", ""),
        )

    result = await exchange.place_order(order)
    return {"data": {"order": {
        "pair": order.pair,
        "side": order.side,
        "amount": order.amount,
        "price": result.get("price", order.price),
        "status": "filled",
    }}}


async def journal_rejection(state: ArenaState) -> dict:
    from cryptotrader.journal.commit import build_commit, generate_hash
    from cryptotrader.journal.store import JournalStore

    db_url = state["metadata"].get("database_url")
    store = JournalStore(db_url)
    commit = build_commit(
        pair=state["metadata"]["pair"],
        snapshot_summary=state["data"].get("snapshot_summary", {}),
        analyses=state["data"].get("analyses", {}),
        debate_rounds=state.get("debate_round", 0),
        divergence=state.get("divergence_scores", [0.0])[-1],
        verdict=state["data"].get("verdict", {}),
        risk_gate=state["data"].get("risk_gate", {}),
        order=None,
        parent_hash=None,
    )
    await store.commit(commit)
    return {"data": {"journal_hash": commit.hash}}


# ── Graph builder ──

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


# ── Bull/Bear debate nodes ──

async def bull_bear_debate(state: ArenaState) -> dict:
    from cryptotrader.debate.researchers import run_debate
    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    rounds = state["metadata"].get("debate_rounds", 2)
    debate = await run_debate(analyses, rounds=rounds, model=model)
    return {"data": {"debate": debate}}


async def judge_verdict(state: ArenaState) -> dict:
    from cryptotrader.debate.researchers import judge_debate
    debate = state["data"]["debate"]
    pair = state["metadata"]["pair"]
    model = state["metadata"].get("verdict_model",
                state["metadata"].get("debate_model", "gpt-4o-mini"))
    result = await judge_debate(debate, pair, model=model)
    return {"data": {"verdict": {
        "action": result["action"],
        "confidence": result["confidence"],
        "position_scale": result["confidence"],
        "divergence": 0.0,
        "reasoning": result["reasoning"],
    }}}


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


def _build_full_graph(config: dict | None = None) -> Any:
    graph = StateGraph(ArenaState)

    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)
    graph.add_node("cross_challenge", debate_round)
    graph.add_node("check_convergence", check_stability)
    graph.add_node("verdict", make_verdict)
    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_rejection", journal_rejection)

    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")
    graph.add_edge("tech_agent", "cross_challenge")
    graph.add_edge("chain_agent", "cross_challenge")
    graph.add_edge("news_agent", "cross_challenge")
    graph.add_edge("macro_agent", "cross_challenge")
    graph.add_edge("cross_challenge", "check_convergence")
    graph.add_conditional_edges("check_convergence", convergence_router, {
        "converged": "verdict",
        "continue": "cross_challenge",
    })
    graph.add_edge("verdict", "risk_gate")
    graph.add_conditional_edges("risk_gate", risk_router, {
        "approved": "execute",
        "rejected": "record_rejection",
    })
    graph.add_edge("execute", END)
    graph.add_edge("record_rejection", END)

    return graph.compile()
