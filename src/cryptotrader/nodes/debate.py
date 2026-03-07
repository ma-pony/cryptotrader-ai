"""Debate round nodes — cross-challenge and convergence."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState  # noqa: TCH001

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


async def debate_round(state: ArenaState) -> dict:
    """One round of cross-challenge debate between agents."""
    from cryptotrader.debate.challenge import build_challenge_prompt
    from cryptotrader.debate.verdict import _extract_json

    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    updated: dict[str, Any] = {}

    for agent_id, analysis in analyses.items():
        others = {k: v for k, v in analyses.items() if k != agent_id}
        prompt = build_challenge_prompt(agent_id, state["metadata"]["pair"], analysis, others)
        role_label = _DEBATE_ROLES.get(agent_id, agent_id)
        system = DEBATE_SYSTEM.format(role=role_label)
        try:
            from cryptotrader.agents.base import acompletion_with_fallback

            resp = await acompletion_with_fallback(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                timeout=60,
            )
            text = resp.choices[0].message.content
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
            updated[agent_id] = merged
        except Exception as e:
            logger.warning("Debate round LLM call failed for %s: %s", agent_id, e)
            updated[agent_id] = analysis

    return {
        "data": {"analyses": updated},
        "debate_round": state["debate_round"] + 1,
    }


async def check_stability(state: ArenaState) -> dict:
    """Compute divergence score after a debate round."""
    from cryptotrader.debate.convergence import compute_divergence

    analyses = state["data"].get("analyses", {})
    divergence = compute_divergence(analyses)
    scores = list(state.get("divergence_scores") or [])
    scores.append(divergence)
    return {"divergence_scores": scores}


def convergence_router(state: ArenaState) -> str:
    """Route to 'converged' or 'continue' based on divergence trend."""
    from cryptotrader.debate.convergence import check_convergence

    scores = state.get("divergence_scores") or []
    if state["debate_round"] >= state["max_debate_rounds"]:
        return "converged"
    threshold = state["metadata"].get("convergence_threshold", 0.1)
    if len(scores) >= 2 and check_convergence(scores[:-1], scores[-1], threshold=threshold):
        return "converged"
    return "continue"


async def bull_bear_debate(state: ArenaState) -> dict:
    """Run bull/bear adversarial debate."""
    from cryptotrader.debate.researchers import run_debate

    analyses = state["data"].get("analyses", {})
    model = state["metadata"].get("debate_model", "gpt-4o-mini")
    rounds = state["metadata"].get("debate_rounds", 2)
    debate = await run_debate(analyses, rounds=rounds, model=model)
    return {"data": {"debate": debate}}


async def judge_verdict(state: ArenaState) -> dict:
    """Judge evaluates bull/bear debate and issues verdict."""
    from cryptotrader.debate.researchers import judge_debate

    debate = state["data"]["debate"]
    pair = state["metadata"]["pair"]
    model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", "gpt-4o-mini"))
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
