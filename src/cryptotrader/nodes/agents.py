"""Agent analysis nodes — fan-out to 4 specialized agents."""

from __future__ import annotations

from typing import Any

from cryptotrader.state import ArenaState


async def _run_agent(agent_type: str, state: ArenaState) -> dict:
    from cryptotrader.agents.chain import ChainAgent
    from cryptotrader.agents.macro import MacroAgent
    from cryptotrader.agents.news import NewsAgent
    from cryptotrader.agents.tech import TechAgent

    agents: dict[str, Any] = {
        "tech_agent": lambda m: TechAgent(model=m),
        "chain_agent": lambda m: ChainAgent(model=m),
        "news_agent": lambda m: NewsAgent(model=m),
        "macro_agent": lambda m: MacroAgent(model=m),
    }
    models_cfg = state["metadata"].get("models", {})
    model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", ""))
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
