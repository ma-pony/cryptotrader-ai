"""Hybrid verdict: deterministic rules first, LLM only for ambiguous cases."""

from __future__ import annotations

import json
import logging

import litellm

from cryptotrader.models import TradeVerdict

logger = logging.getLogger(__name__)

_DIR_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def _extract_field(analysis: dict, field: str, default=None):
    """Extract structured field from agent analysis (may be in data_points or top-level)."""
    if field in analysis:
        return analysis[field]
    return analysis.get("data_points", {}).get(field, default)


def make_verdict_rules(analyses: dict[str, dict]) -> TradeVerdict | None:
    """Deterministic rules engine. Returns None if no rule matches (ambiguous)."""
    tech = analyses.get("tech_agent", {})
    chain = analyses.get("chain_agent", {})
    macro = analyses.get("macro_agent", {})

    # Extract structured fields
    regime = _extract_field(tech, "regime", "unknown")
    tech_dir = tech.get("direction", "neutral")
    tech_conf = float(tech.get("confidence", 0.0))

    chain_dq = _extract_field(chain, "data_quality", "none")
    crowding = _extract_field(chain, "crowding", "no_data")
    funding_signal = _extract_field(chain, "funding_signal", "neutral")

    macro_env = _extract_field(macro, "environment", "neutral")
    macro_dir = macro.get("direction", "neutral")

    # Rule 1: True ranging (regime=ranging AND direction=neutral) → HOLD
    if regime == "ranging" and tech_dir == "neutral":
        return TradeVerdict(action="hold", confidence=0.3, reasoning=f"Ranging + neutral direction")

    # Rule 2: Determine action from tech direction (regime is secondary)
    if tech_dir == "bearish" and tech_conf >= 0.55:
        action = "short"
        base_conf = min(tech_conf + 0.05, 0.85)  # Slight boost for rule-based
    elif tech_dir == "bullish" and tech_conf >= 0.55:
        action = "long"
        base_conf = min(tech_conf + 0.05, 0.85)
    elif tech_dir == "neutral":
        return TradeVerdict(action="hold", confidence=0.3, reasoning="Tech neutral")
    else:
        return None  # Low confidence, let LLM decide

    # Ranging regime penalty (has direction but weak trend)
    if regime == "ranging":
        base_conf -= 0.1

    # Chain adjustment (only with real data)
    if chain_dq == "full":
        same_dir_crowded = (action == "short" and crowding == "shorts_crowded") or \
                           (action == "long" and crowding == "longs_crowded")
        if same_dir_crowded:
            base_conf -= 0.15

    if (action == "long" and funding_signal == "contrarian_bearish") or \
       (action == "short" and funding_signal == "contrarian_bullish"):
        base_conf -= 0.1

    # Macro adjustment
    if action == "long" and (macro_env == "risk_off" or macro_dir == "bearish"):
        base_conf -= 0.1
    elif action == "short" and (macro_env == "risk_on" or macro_dir == "bullish"):
        base_conf -= 0.1

    base_conf = max(0.1, min(0.9, base_conf))

    if base_conf < 0.35:
        return TradeVerdict(action="hold", confidence=base_conf, reasoning="Low confidence after adjustments")

    return TradeVerdict(
        action=action, confidence=base_conf, position_scale=base_conf,
        reasoning=f"Rule: regime={regime}, tech={tech_dir}@{tech_conf:.0%}, chain={crowding}, macro={macro_env}",
    )


TIEBREAK_PROMPT = """Four agents analyzed a crypto market. Their analyses are below.
Tech and other signals are ambiguous. Make a final call.
Respond ONLY with JSON: {"action": "long|short|hold", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


async def make_verdict_llm(
    analyses: dict[str, dict],
    model: str = "gpt-4o-mini",
) -> TradeVerdict:
    """Hybrid: rules first, LLM only for tiebreaking."""
    # Try deterministic rules first
    rule_verdict = make_verdict_rules(analyses)
    if rule_verdict is not None:
        return rule_verdict

    # Ambiguous case: use LLM
    agent_reports = "\n\n".join(
        f"[{aid}]\n{json.dumps(a, indent=2, default=str)}" for aid, a in analyses.items()
    )
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": TIEBREAK_PROMPT},
                {"role": "user", "content": agent_reports},
            ],
            temperature=0.1, max_tokens=256,
        )
        text = resp.choices[0].message.content
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return TradeVerdict(
            action=data.get("action", "hold"),
            confidence=float(data.get("confidence", 0.0)),
            position_scale=float(data.get("confidence", 0.0)),
            reasoning=f"LLM tiebreak: {data.get('reasoning', '')}",
        )
    except Exception:
        logger.exception("Verdict LLM tiebreak failed")
        return TradeVerdict(action="hold", confidence=0.2, reasoning="LLM tiebreak failed")


def make_verdict_weighted(
    analyses: dict[str, dict],
    divergence: float = 0.0,
    divergence_threshold: float = 0.7,
) -> TradeVerdict:
    """Legacy fallback: weighted average verdict."""
    if divergence > divergence_threshold:
        return TradeVerdict(action="hold", divergence=divergence, reasoning="High divergence")
    score = sum(a["confidence"] * _DIR_MAP.get(a["direction"], 0.0) for a in analyses.values())
    action = "long" if score > 0 else "short" if score < 0 else "hold"
    if action != "hold":
        target = "bullish" if action == "long" else "bearish"
        agreeing = [a["confidence"] for a in analyses.values() if a["direction"] == target]
        confidence = sum(agreeing) / len(agreeing) if agreeing else 0.0
    else:
        confidence = 0.0
    return TradeVerdict(action=action, confidence=confidence, position_scale=max(0.0, 1.0 - divergence),
                        divergence=divergence, reasoning=f"Weighted score={score:.3f}")
