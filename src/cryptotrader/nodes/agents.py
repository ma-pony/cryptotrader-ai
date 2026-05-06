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
    regime_tags = state["data"].get("regime_tags", [])
    cfg = load_config()

    agent_cfg = cfg.agents.get(agent_type)
    if agent_cfg is not None and agent_cfg.model:
        model = agent_cfg.model
    else:
        models_cfg = state["metadata"].get("models", {})
        model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", ""))

    try:
        agent = cfg.agents.build(
            agent_type,
            backtest_mode=backtest_mode,
            regime_tags=regime_tags,
            model_override=model,
        )
    except KeyError:
        from cryptotrader.agents.chain import ChainAgent
        from cryptotrader.agents.macro import MacroAgent
        from cryptotrader.agents.news import NewsAgent
        from cryptotrader.agents.tech import TechAgent

        agents_fallback: dict[str, Any] = {
            "tech_agent": lambda m: TechAgent(model=m),
            "chain_agent": lambda m: ChainAgent(model=m, backtest_mode=backtest_mode),
            "news_agent": lambda m: NewsAgent(model=m, backtest_mode=backtest_mode),
            "macro_agent": lambda m: MacroAgent(model=m),
        }
        agent = agents_fallback[agent_type](model)
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

    # Inject live steering instructions (T024)
    experience = await _inject_steering(state, agent_type, experience)

    # Publish agent_thinking event
    from cryptotrader.chat.runtime_registry import get_event_bus

    event_bus = get_event_bus((state.get("metadata") or {}).get("session_id"))
    if event_bus is not None:
        await event_bus.publish("agent_thinking", {"agent_id": agent_type})

    agent_timeout = agent_cfg.timeout_seconds if agent_cfg is not None else 0
    timeout_seconds = agent_timeout if agent_timeout > 0 else cfg.models.timeout_seconds
    logger.info("Running %s for %s (model=%s)", agent_type, snapshot.pair, model or "default")
    steered = (state.get("metadata") or {}).get(f"_steered_{agent_type}", False)
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
        mock = dict(_MOCK_ANALYSIS_RESULT)
        await _publish_agent_done(event_bus, agent_type, mock, steered, state)
        return {"data": {"analyses": {agent_type: mock}}}
    except Exception:
        logger.warning(
            "LLM call failed for %s — degrading to mock result",
            agent_type,
            exc_info=True,
        )
        mock = dict(_MOCK_ANALYSIS_RESULT)
        await _publish_agent_done(event_bus, agent_type, mock, steered, state)
        return {"data": {"analyses": {agent_type: mock}}}

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
        "steered": steered,
    }
    result.update(analysis.data_points)
    await _publish_agent_done(event_bus, agent_type, result, steered, state)

    # Update prev_snapshot_hash and prev_analyses for the next cycle's hash reuse.
    # Merge with existing prev_analyses to avoid evicting other agents' cached results.
    output: dict[str, Any] = {"analyses": {agent_type: result}}
    if current_hash is not None:
        merged_prev = dict(prev_analyses)
        merged_prev[agent_type] = result
        output["prev_snapshot_hash"] = current_hash
        output["prev_analyses"] = merged_prev
    return {"data": output}


async def _inject_steering(
    state: ArenaState,
    agent_type: str,
    experience: str,
) -> str:
    """Read and apply live steering instructions from Redis queue."""
    metadata = state.get("metadata") or {}
    session_id = metadata.get("session_id")
    state_mgr = metadata.get("redis_state_manager")
    if not session_id or state_mgr is None:
        return experience

    steer_key = f"steering:{session_id}:{agent_type}"
    try:
        instructions = await state_mgr.buffer_range(steer_key, 0, -1)
        if instructions:
            await state_mgr.buffer_delete(steer_key)
            joined = "\n".join(instructions)
            if metadata is not None:
                metadata[f"_steered_{agent_type}"] = True
            suffix = f"\n\n[用户实时引导]\n{joined}"
            return f"{experience}{suffix}" if experience else suffix
    except Exception:
        logger.warning("Failed to read steering queue for %s", agent_type, exc_info=True)
    return experience


async def _publish_agent_done(
    event_bus: Any,
    agent_type: str,
    result: dict[str, Any],
    steered: bool,
    state: ArenaState,
) -> None:
    """Publish agent_analysis event and update completed_agents on the task."""
    if event_bus is not None:
        await event_bus.publish(
            "agent_analysis",
            {
                "agent_id": agent_type,
                "direction": result.get("direction", "neutral"),
                "confidence": result.get("confidence", 0),
                "steered": steered,
            },
        )

    metadata = state.get("metadata") or {}
    session_id = metadata.get("session_id")
    if session_id:
        try:
            from cryptotrader.chat.task_manager import BackgroundTaskManager

            mgr = BackgroundTaskManager.get_instance()
            task = mgr.get(session_id)
            if task is not None:
                task.completed_agents.append(agent_type)
        except Exception:
            pass


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
