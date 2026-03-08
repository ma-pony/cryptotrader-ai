"""AI-driven verdict: evaluates argument quality, not just weighted averages.

Phase 4B: All verdicts go through AI evaluation that sees full agent outputs
(reasoning, key_factors, risk_flags, data_points) plus risk constraints.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import FUNDING_RATE_HIGH, FUNDING_RATE_LOW, create_llm, extract_content
from cryptotrader.models import TradeVerdict


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response using balanced-brace extraction."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced braces in response: {text[:200]}")


logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"long", "short", "hold"}


def _normalize_action(raw: str) -> str:
    """Map LLM action strings to valid actions."""
    raw = raw.strip().lower()
    if raw in _VALID_ACTIONS:
        return raw
    if raw in ("buy", "bullish"):
        return "long"
    if raw in ("sell", "bearish"):
        return "short"
    return "hold"


VERDICT_PROMPT = """You are the chief decision-maker for a crypto trading system. Four specialist agents have
analyzed the market. You must evaluate the QUALITY of their arguments — not just count votes.

Your job:
1. Read each agent's full analysis: direction, confidence, reasoning, key_factors, risk_flags, and data_points.
2. Assess which agents have the strongest evidence (specific data points, not vague claims).
3. Identify contradictions between agents and determine which side has better support.
4. Factor in the risk constraints below — you CANNOT exceed these limits.
5. Output a final trading decision.

Evaluation criteria (in order of importance):
- Evidence quality: Does the agent cite specific numbers, or just make vague claims?
- Risk flag coverage: Did any agent flag risks that others missed? Unaddressed risks should lower confidence.
- Contradiction resolution: When agents disagree, which side has more concrete evidence?
- Confidence calibration: An agent claiming 0.9 confidence with weak evidence should be trusted LESS than one
  claiming 0.5 with strong data points.

Decision rules:
- "hold" is valid when evidence is genuinely mixed or when risk constraints make trading inadvisable.
- Do NOT default to hold out of caution — if evidence clearly favors a direction, act on it.
- position_scale should reflect your conviction: 0.3-0.5 for moderate, 0.6-0.8 for strong, 0.9+ for exceptional.
- thesis: one sentence summarizing WHY you're taking this trade (or why holding).
- invalidation: what specific condition would prove this thesis wrong
  (e.g., "BTC drops below 60k", "funding rate flips negative").

Output ONLY JSON:
{
  "action": "long|short|hold",
  "confidence": 0.0-1.0,
  "position_scale": 0.0-1.0,
  "reasoning": "2-3 sentences explaining your evaluation of agent arguments",
  "thesis": "one sentence trade thesis",
  "invalidation": "specific condition that would invalidate thesis"
}"""


def _format_funding_rate(fr: float) -> str:
    """Format funding rate with crowd signal label."""
    label = ""
    if fr > FUNDING_RATE_HIGH:
        label = " (ELEVATED — crowded long)"
    elif fr < FUNDING_RATE_LOW:
        label = " (NEGATIVE — crowded short)"
    return f"Current funding rate: {fr:.6f}{label}"


def _format_constraints(constraints: dict) -> str:
    """Format risk constraints into a readable block for the verdict prompt."""
    if not constraints:
        return "No risk constraints available."
    parts = []

    # Simple formatted fields
    _fields = [
        ("max_position_pct", "Max position size: {:.0%} of portfolio"),
        ("remaining_exposure_pct", "Remaining exposure capacity: {:.0%}"),
        ("drawdown_current", "Current drawdown: {:.1%}"),
        ("max_drawdown_pct", "Max drawdown limit: {:.0%}"),
        ("volatility", "Current volatility: {:.4f}"),
    ]
    for key, template in _fields:
        if key in constraints:
            parts.append(template.format(constraints[key]))

    # Daily loss with exhaustion warning
    if "daily_loss_remaining_pct" in constraints:
        v = constraints["daily_loss_remaining_pct"]
        suffix = " (EXHAUSTED — no new trades)" if v <= 0 else ""
        parts.append(f"Daily loss budget remaining: {v:.1%}{suffix}")

    if constraints.get("cooldown_pairs"):
        parts.append(f"Pairs on cooldown: {', '.join(constraints['cooldown_pairs'])}")
    if constraints.get("circuit_breaker_active"):
        parts.append("CIRCUIT BREAKER ACTIVE — all trading halted until manual reset")
    if "funding_rate" in constraints:
        parts.append(_format_funding_rate(constraints["funding_rate"]))

    return "\n".join(parts) if parts else "No specific constraints."


async def make_verdict_ai(
    analyses: dict[str, dict],
    constraints: dict | None = None,
    calibration: str = "",
    model: str = "gpt-4o",
) -> TradeVerdict:
    """AI-driven verdict that evaluates argument quality with risk constraints and calibration."""
    # Format full agent reports (all fields, not just direction+confidence)
    agent_reports = "\n\n".join(
        f"── {aid.upper()} ──\n{json.dumps(a, indent=2, default=str)}" for aid, a in analyses.items()
    )

    constraint_block = _format_constraints(constraints or {})
    calibration_block = f"\n\n{calibration}" if calibration else ""

    user_msg = f"""RISK CONSTRAINTS (hard limits — you cannot exceed these):
{constraint_block}{calibration_block}

AGENT ANALYSES:
{agent_reports}"""

    try:
        llm = create_llm(model=model, temperature=0.1, timeout=120, json_mode=True)
        messages = [SystemMessage(content=VERDICT_PROMPT), HumanMessage(content=user_msg)]
        resp = await llm.ainvoke(messages)
        text = extract_content(resp)
        data = _extract_json(text)

        action = _normalize_action(data.get("action", "hold"))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))

        # If circuit breaker is active or daily loss exhausted, force hold
        if constraints and constraints.get("circuit_breaker_active"):
            action = "hold"
            confidence = 0.0

        return TradeVerdict(
            action=action,
            confidence=confidence,
            position_scale=max(0.0, min(1.0, float(data.get("position_scale", confidence)))),
            reasoning=data.get("reasoning", ""),
            thesis=data.get("thesis", ""),
            invalidation=data.get("invalidation", ""),
        )
    except Exception:
        logger.exception("Verdict AI call failed, falling back to conservative hold")
        return TradeVerdict(action="hold", confidence=0.1, reasoning="Verdict AI call failed — defaulting to hold")


# Legacy aliases for backward compatibility in tests / backtest
def make_verdict_rules(analyses: dict[str, dict]) -> TradeVerdict:
    """Lightweight weighted-average verdict for backtesting (no LLM call)."""
    _dir_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    score = sum(
        _dir_map.get(a.get("direction", "neutral"), 0.0) * float(a.get("confidence", 0.0)) for a in analyses.values()
    )
    action = "long" if score > 0.1 else "short" if score < -0.1 else "hold"
    if action != "hold":
        target = "bullish" if action == "long" else "bearish"
        agreeing = [float(a["confidence"]) for a in analyses.values() if a.get("direction") == target]
        confidence = sum(agreeing) / len(agreeing) if agreeing else 0.0
    else:
        confidence = 0.0
    return TradeVerdict(
        action=action,
        confidence=confidence,
        position_scale=max(0.0, confidence),
        reasoning=f"Weighted score={score:.3f}",
    )


async def make_verdict_llm(
    analyses: dict[str, dict],
    model: str = "",
    constraints: dict | None = None,
    calibration: str = "",
) -> TradeVerdict:
    """Main entry point — routes to AI verdict."""
    return await make_verdict_ai(analyses, constraints=constraints, calibration=calibration, model=model)


def make_verdict_weighted(
    analyses: dict[str, dict],
    divergence: float = 0.0,
    divergence_threshold: float = 0.7,
) -> TradeVerdict:
    """Legacy fallback: weighted average verdict for backtest/no-LLM mode."""
    _dir_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    if divergence > divergence_threshold:
        return TradeVerdict(action="hold", divergence=divergence, reasoning="High divergence")
    score = sum(a["confidence"] * _dir_map.get(a["direction"], 0.0) for a in analyses.values())
    action = "long" if score > 0 else "short" if score < 0 else "hold"
    if action != "hold":
        target = "bullish" if action == "long" else "bearish"
        agreeing = [a["confidence"] for a in analyses.values() if a["direction"] == target]
        confidence = sum(agreeing) / len(agreeing) if agreeing else 0.0
    else:
        confidence = 0.0
    return TradeVerdict(
        action=action,
        confidence=confidence,
        position_scale=max(0.0, 1.0 - divergence),
        divergence=divergence,
        reasoning=f"Weighted score={score:.3f}",
    )
