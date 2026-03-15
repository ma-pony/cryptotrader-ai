"""Agent analysis nodes — fan-out to 4 specialized agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)

# Default token budget for experience context (chars / 4 ≈ tokens)
_DEFAULT_TOKEN_BUDGET = 2000

# Degraded analysis result returned when an agent times out or fails
_MOCK_ANALYSIS_RESULT: dict[str, Any] = {
    "direction": "neutral",
    "confidence": 0.0,
    "reasoning": "Agent analysis unavailable (timeout or error)",
    "key_factors": [],
    "risk_flags": [],
    "is_mock": True,
    "data_sufficiency": "insufficient",
}


async def _run_agent(agent_type: str, state: ArenaState) -> dict:
    from cryptotrader.agents.chain import ChainAgent
    from cryptotrader.agents.macro import MacroAgent
    from cryptotrader.agents.news import NewsAgent
    from cryptotrader.agents.tech import TechAgent
    from cryptotrader.config import load_config
    from cryptotrader.learning.context import gather_packets, select_packets, structure_experience

    # Snapshot hash reuse: active only in scheduler continuous-cycle scenarios.
    # Conditions: current hash exists AND matches prev hash AND cached entry exists for this agent.
    current_hash = state["data"].get("snapshot_hash")
    prev_hash = state["data"].get("prev_snapshot_hash")
    prev_analyses = state["data"].get("prev_analyses", {})
    if current_hash is not None and prev_hash is not None and current_hash == prev_hash and agent_type in prev_analyses:
        logger.info(
            "snapshot_hash match -- reusing cached result for %s (LLM call skipped)",
            agent_type,
        )
        return {"data": {"analyses": {agent_type: prev_analyses[agent_type]}}}

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

    # Build GSSC experience context
    experience = _build_experience(state, agent_type)
    if not experience:
        # Fallback: use GSSC pipeline from structured data
        memory = state["data"].get("experience_memory", {}).get(agent_type)
        cases = state["data"].get("historical_cases", [])
        agent_corrections = state["data"].get("agent_corrections", {})
        correction = agent_corrections.get(agent_type, "")
        regime_tags = state["data"].get("regime_tags", [])

        packets = gather_packets(memory, cases, correction)
        selected = select_packets(packets, regime_tags, _DEFAULT_TOKEN_BUDGET)
        experience = structure_experience(selected)

    timeout_seconds = load_config().models.timeout_seconds
    logger.info("Running %s for %s (model=%s)", agent_type, snapshot.pair, model or "default")
    try:
        analysis = await asyncio.wait_for(
            agent.analyze(snapshot, experience),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "LLM timeout for %s after %ss — degrading to mock result",
            agent_type,
            timeout_seconds,
        )
        return {"data": {"analyses": {agent_type: dict(_MOCK_ANALYSIS_RESULT)}}}
    except Exception:
        logger.warning(
            "LLM call failed for %s — degrading to mock result",
            agent_type,
            exc_info=True,
        )
        return {"data": {"analyses": {agent_type: dict(_MOCK_ANALYSIS_RESULT)}}}

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

    # Update prev_snapshot_hash and prev_analyses for the next cycle's hash reuse.
    # Merge with existing prev_analyses to avoid evicting other agents' cached results.
    output: dict[str, Any] = {"analyses": {agent_type: result}}
    if current_hash is not None:
        merged_prev = dict(prev_analyses)
        merged_prev[agent_type] = result
        output["prev_snapshot_hash"] = current_hash
        output["prev_analyses"] = merged_prev
    return {"data": output}


def _build_experience(state: ArenaState, agent_type: str) -> str:
    """Build experience string from legacy state fields (backward compat).

    Returns empty string if new GSSC fields are available.
    """
    # If new structured fields are present, return empty to trigger GSSC path
    if "experience_memory" in state.get("data", {}):
        return ""

    # Legacy path: experience is a pre-formatted string
    experience = state["data"].get("experience", "")
    agent_corrections = state["data"].get("agent_corrections", {})
    correction = agent_corrections.get(agent_type, "")
    if correction:
        experience = f"{experience}\n\n{correction}" if experience else correction
    agent_reflections = state["data"].get("agent_reflections", {})
    reflection = agent_reflections.get(agent_type, "")
    if reflection:
        block = f"Strategy memo (your own prior self-reflection):\n{reflection}"
        experience = f"{experience}\n\n{block}" if experience else block
    return experience


@node_logger()
async def tech_analyze(state: ArenaState) -> dict:
    return await _run_agent("tech_agent", state)


@node_logger()
async def chain_analyze(state: ArenaState) -> dict:
    return await _run_agent("chain_agent", state)


@node_logger()
async def news_analyze(state: ArenaState) -> dict:
    return await _run_agent("news_agent", state)


@node_logger()
async def macro_analyze(state: ArenaState) -> dict:
    return await _run_agent("macro_agent", state)
