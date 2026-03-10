"""Agent analysis nodes — fan-out to 4 specialized agents."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)


async def _run_agent(agent_type: str, state: ArenaState) -> dict:
    from cryptotrader.agents.chain import ChainAgent
    from cryptotrader.agents.macro import MacroAgent
    from cryptotrader.agents.news import NewsAgent
    from cryptotrader.agents.tech import TechAgent

    backtest_mode = state["metadata"].get("backtest_mode", False)
    agents: dict[str, Any] = {
        "tech_agent": lambda m: TechAgent(model=m),
        "chain_agent": lambda m: ChainAgent(model=m, backtest_mode=backtest_mode),
        "news_agent": lambda m: NewsAgent(model=m, backtest_mode=backtest_mode),
        "macro_agent": lambda m: MacroAgent(model=m),
    }
    models_cfg = state["metadata"].get("models", {})
    model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", ""))
    agent = agents[agent_type](model)
    snapshot = state["data"]["snapshot"]
    experience = state["data"].get("experience", "")
    # Inject per-agent bias correction (each agent only sees their own)
    agent_corrections = state["data"].get("agent_corrections", {})
    correction = agent_corrections.get(agent_type, "")
    if correction:
        experience = f"{experience}\n\n{correction}" if experience else correction
    # Inject per-agent reflection memo (strategy memo from prior self-reflection)
    agent_reflections = state["data"].get("agent_reflections", {})
    reflection = agent_reflections.get(agent_type, "")
    if reflection:
        block = f"Strategy memo (your own prior self-reflection):\n{reflection}"
        experience = f"{experience}\n\n{block}" if experience else block
    logger.info("Running %s for %s (model=%s)", agent_type, snapshot.pair, model or "default")
    analysis = await agent.analyze(snapshot, experience)
    logger.info(
        "%s result: direction=%s confidence=%.2f mock=%s sufficiency=%s",
        agent_type,
        analysis.direction,
        analysis.confidence,
        analysis.is_mock,
        analysis.data_sufficiency,
    )
    result = {
        "direction": analysis.direction,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "key_factors": analysis.key_factors,
        "risk_flags": analysis.risk_flags,
        "is_mock": analysis.is_mock,
        "data_sufficiency": analysis.data_sufficiency,
    }
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
