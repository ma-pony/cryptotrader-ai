"""Agent analysis nodes — fan-out to 4 specialized agents."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from cryptotrader.agents.prompt_builder import (
    PromptBuilder,
)
from cryptotrader.learning.evolution.provider import EvolvingMemoryProvider
from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider
from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)

# Module-level singletons: one PromptBuilder per agent_id, shared across cycles.
_memory_provider: EvolvingMemoryProvider | None = None
_skill_provider: EvolvingSkillProvider | None = None  # spec 019: replaced DefaultSkillProvider
_prompt_builders: dict[str, PromptBuilder] = {}


def _get_or_build_pb(agent_id: str, model: str) -> PromptBuilder:
    """Return cached PromptBuilder for agent_id, building it on first access.

    agent_id follows the full registry name (e.g. "tech_agent"); the config file
    uses the short name without the "_agent" suffix (e.g. "config/agents/tech.md").
    Paths are resolved relative to the repo root (parent of src/), so the function
    works correctly regardless of the process cwd.
    """
    global _memory_provider, _skill_provider
    # Resolve repo root as the directory three levels above this file:
    # nodes/agents.py → nodes/ → cryptotrader/ → src/ → <repo_root>/
    _repo_root = Path(__file__).parent.parent.parent.parent
    if _memory_provider is None:
        _memory_provider = EvolvingMemoryProvider(memory_root=_repo_root / "agent_memory")
        # spec 019: EvolvingSkillProvider replaces DefaultSkillProvider (FR-W12)
        _skill_provider = EvolvingSkillProvider(skill_root=_repo_root / "agent_skills")
        # spec 019: wire load_skill_tool to use EvolvingSkillProvider (FR-W14)
        import cryptotrader.agents.skills.tool as _skill_tool_mod

        _skill_tool_mod.load_skill_tool = _skill_tool_mod._make_load_skill_tool(provider=_skill_provider)
    if agent_id not in _prompt_builders:
        # Strip "_agent" suffix: "tech_agent" → "tech", "chain_agent" → "chain"
        short_id = agent_id.removesuffix("_agent")
        _prompt_builders[agent_id] = PromptBuilder(
            agent_id=short_id,
            config_dir=_repo_root / "config" / "agents",
            memory_provider=_memory_provider,
            skill_provider=_skill_provider,
            model=model,
        )
    return _prompt_builders[agent_id]


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
    cfg = load_config()

    agent_cfg = cfg.agents.get(agent_type)
    if agent_cfg is not None and agent_cfg.model:
        model = agent_cfg.model
    else:
        models_cfg = state["metadata"].get("models", {})
        model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", ""))

    pb = _get_or_build_pb(agent_type, model)
    try:
        agent = cfg.agents.build(
            agent_type,
            prompt_builder=pb,
            backtest_mode=backtest_mode,
            model_override=model,
        )
    except KeyError:
        from cryptotrader.agents.chain import ChainAgent
        from cryptotrader.agents.macro import MacroAgent
        from cryptotrader.agents.news import NewsAgent
        from cryptotrader.agents.tech import TechAgent

        agents_fallback: dict[str, Any] = {
            "tech_agent": lambda m: TechAgent(prompt_builder=_get_or_build_pb(agent_type, m), model=m),
            "chain_agent": lambda m: ChainAgent(
                prompt_builder=_get_or_build_pb(agent_type, m), model=m, backtest_mode=backtest_mode
            ),
            "news_agent": lambda m: NewsAgent(
                prompt_builder=_get_or_build_pb(agent_type, m), model=m, backtest_mode=backtest_mode
            ),
            "macro_agent": lambda m: MacroAgent(prompt_builder=_get_or_build_pb(agent_type, m), model=m),
        }
        agent = agents_fallback[agent_type](model)
    snapshot = state["data"]["snapshot"]

    # Build experience context from legacy state fields (steering/corrections only)
    # Skills are now injected via PromptBuilder (spec 017b) inside agent.analyze()
    experience = _build_experience(state, agent_type)

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
        # Emit the standard result-line even on degraded paths so downstream
        # audits / dashboards see a consistent per-agent record per cycle.
        logger.info(
            "%s result: direction=%s confidence=%.2f mock=%s sufficiency=%s (degraded=timeout)",
            agent_type,
            mock.get("direction"),
            float(mock.get("confidence", 0.0)),
            True,
            mock.get("data_sufficiency", "low"),
        )
        await _publish_agent_done(event_bus, agent_type, mock, steered, state)
        return {"data": {"analyses": {agent_type: mock}}}
    except Exception:
        logger.warning(
            "LLM call failed for %s — degrading to mock result",
            agent_type,
            exc_info=True,
        )
        mock = dict(_MOCK_ANALYSIS_RESULT)
        logger.info(
            "%s result: direction=%s confidence=%.2f mock=%s sufficiency=%s (degraded=exception)",
            agent_type,
            mock.get("direction"),
            float(mock.get("confidence", 0.0)),
            True,
            mock.get("data_sufficiency", "low"),
        )
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
    """Build experience string from state fields.

    2026-05-13 final state: returns "" unconditionally. All legacy
    injection paths have been removed:
      - bias-correction (`agent_corrections`)
      - verbal-reinforcement (`historical_cases` / `experience`)
      - self-reflection memo (`agent_reflections`)
    Each shared the same flaw — feeding small-sample LLM-generated
    priors back into prompts re-introduces the directional anchoring
    the round-3 minimal skills were designed to eliminate.

    Skills injection (the proper feedback channel) is handled by
    PromptBuilder (spec 017b) inside agent.analyze(); the evolution
    daemon writes distilled patterns into the SKILL.md
    `AUTO-DISTILLED-PATTERNS` section, which PromptBuilder loads.
    """
    # state and agent_type kept in signature because spec 017b's
    # PromptBuilder.build(...) currently takes an `experience` kwarg.
    # Once that surface is collapsed too, this helper can disappear.
    del state, agent_type  # explicitly unused
    return ""


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
