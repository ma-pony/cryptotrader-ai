"""Debate round nodes — cross-challenge and convergence."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import create_llm, extract_content
from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

_DEBATE_ROLES = {
    "tech_agent": "technical analysis",
    "chain_agent": "on-chain and derivatives analysis",
    "news_agent": "news and sentiment analysis",
    "macro_agent": "macroeconomic analysis",
}

DEBATE_SYSTEM = """You are a {role} specialist in a multi-agent trading debate.

Rules:
- Base arguments ONLY on data. Cite specific numbers for every claim.
- HOLD your position when your data supports it — do NOT converge toward majority opinion.
- Only change your view if another agent presented a specific data point you hadn't considered.
- If you change, state exactly which data point changed your mind.
- Look for cross-domain contradictions (e.g., bullish technicals + bearish on-chain = important signal).

Output JSON: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reasoning": "...",
"key_factors": [...], "risk_flags": [...], "new_findings": "cross-domain insight from other agents' data"}}"""


async def _debate_one_agent(
    agent_id: str,
    analysis: dict,
    others: dict[str, dict],
    pair: str,
    model: str,
) -> tuple[str, dict]:
    """Single agent's debate response — extracted for parallel execution."""
    from cryptotrader.debate.challenge import build_challenge_prompt
    from cryptotrader.debate.verdict import _extract_json

    prompt = build_challenge_prompt(agent_id, pair, analysis, others)
    role_label = _DEBATE_ROLES.get(agent_id, agent_id)
    system = DEBATE_SYSTEM.format(role=role_label)
    try:
        llm = create_llm(model=model, temperature=0.3, json_mode=True)
        lc_msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]
        resp = await llm.ainvoke(lc_msgs)
        text = extract_content(resp)
        data = _extract_json(text)
        merged = dict(analysis)
        merged.update(
            {
                "direction": data.get("direction", analysis["direction"]),
                "confidence": float(data.get("confidence", analysis["confidence"])),
                "reasoning": data.get("reasoning", analysis["reasoning"]),
                "key_factors": data.get("key_factors", analysis.get("key_factors", [])),
                "risk_flags": data.get("risk_flags", analysis.get("risk_flags", [])),
                "new_findings": data.get("new_findings", ""),
            }
        )
        return agent_id, merged
    except Exception as e:
        logger.warning("Debate round LLM call failed for %s: %s", agent_id, e)
        return agent_id, analysis


async def debate_round(state: ArenaState) -> dict:
    """One round of cross-challenge debate between agents (parallel)."""
    from cryptotrader.config import load_config as _load_config

    analyses = state["data"].get("analyses", {})
    _dcfg = _load_config()
    _default_debate_model = _dcfg.models.debate or _dcfg.models.fallback
    model = state["metadata"].get("debate_model", _default_debate_model)

    tasks = []
    agent_ids = list(analyses.keys())
    for agent_id in agent_ids:
        analysis = analyses[agent_id]
        others = {k: v for k, v in analyses.items() if k != agent_id}
        tasks.append(_debate_one_agent(agent_id, analysis, others, state["metadata"]["pair"], model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    updated: dict[str, Any] = {}
    for i, result in enumerate(results):
        aid = agent_ids[i]
        if isinstance(result, Exception):
            logger.warning("Debate gather failed for %s: %s", aid, result)
            updated[aid] = analyses[aid]
        else:
            updated[result[0]] = result[1]

    return {
        "data": {"analyses": updated},
        "debate_round": state["debate_round"] + 1,
    }


async def debate_gate(state: ArenaState) -> dict:
    """Compute consensus metrics; set debate_skipped flag."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.convergence import compute_consensus_strength, compute_divergence

    analyses = state["data"].get("analyses", {})
    strength, mean_score = compute_consensus_strength(analyses)
    dispersion = compute_divergence(analyses)
    config = _load_config().debate

    skip = False
    reason = ""
    if not config.skip_debate:
        pass
    elif strength > config.consensus_skip_threshold:
        skip = True
        reason = f"strong consensus (strength={strength:.3f})"
    elif abs(mean_score) < config.confusion_skip_threshold and dispersion < config.confusion_max_dispersion:
        # Low mean + low dispersion = shared confusion (all agents uncertain)
        # Low mean + high dispersion = disagreement (debate needed)
        skip = True
        reason = f"shared confusion (|mean|={abs(mean_score):.3f}, dispersion={dispersion:.3f})"

    if skip:
        logger.info("Debate SKIPPED: %s", reason)

    return {"data": {"debate_skipped": skip, "debate_skip_reason": reason}}


def debate_gate_router(state: ArenaState) -> str:
    """Route to 'debate' or 'skip' based on debate gate result."""
    if state["data"].get("debate_skipped"):
        return "skip"
    return "debate"


async def check_stability(state: ArenaState) -> dict:
    """Compute divergence score after a debate round."""
    from cryptotrader.debate.convergence import compute_divergence

    analyses = state["data"].get("analyses", {})
    divergence = compute_divergence(analyses)
    scores = list(state.get("divergence_scores") or [])
    scores.append(divergence)
    logger.info("Divergence after round %d: %.4f (history: %s)", state["debate_round"], divergence, scores)
    return {"divergence_scores": scores}


def convergence_router(state: ArenaState) -> str:
    """Route to 'converged' or 'continue' based on divergence trend."""
    from cryptotrader.debate.convergence import check_convergence

    scores = state.get("divergence_scores") or []
    if state["debate_round"] >= state["max_debate_rounds"]:
        logger.info("Debate max rounds reached (%d) — converging", state["max_debate_rounds"])
        return "converged"
    threshold = state["metadata"].get("convergence_threshold", 0.1)
    if len(scores) >= 2 and check_convergence(scores[:-1], scores[-1], threshold=threshold):
        logger.info("Debate converged at round %d (threshold=%.2f)", state["debate_round"], threshold)
        return "converged"
    logger.info("Debate continuing to round %d", state["debate_round"] + 1)
    return "continue"


async def bull_bear_debate(state: ArenaState) -> dict:
    """Run bull/bear adversarial debate."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.researchers import run_debate

    analyses = state["data"].get("analyses", {})
    _bbcfg = _load_config()
    _default_bb_model = _bbcfg.models.debate or _bbcfg.models.fallback
    model = state["metadata"].get("debate_model", _default_bb_model)
    rounds = state["metadata"].get("debate_rounds", 2)
    debate = await run_debate(analyses, rounds=rounds, model=model)
    return {"data": {"debate": debate}}


async def judge_verdict(state: ArenaState) -> dict:
    """Judge evaluates bull/bear debate and issues verdict."""
    from cryptotrader.config import load_config as _load_config
    from cryptotrader.debate.researchers import judge_debate

    debate = state["data"]["debate"]
    pair = state["metadata"]["pair"]
    _jcfg = _load_config()
    _default_judge_model = _jcfg.models.verdict or _jcfg.models.fallback
    model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", _default_judge_model))
    result = await judge_debate(debate, pair, model=model)
    return {
        "data": {
            "verdict": {
                "action": result["action"],
                "confidence": result["confidence"],
                "position_scale": result.get("position_scale", result["confidence"]),
                "divergence": 0.0,
                "reasoning": result["reasoning"],
            }
        }
    }
