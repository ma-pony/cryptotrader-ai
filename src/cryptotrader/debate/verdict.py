"""Hybrid verdict: deterministic rules first, LLM only for ambiguous cases."""

from __future__ import annotations

import json
import logging

import litellm

from cryptotrader.models import TradeVerdict

logger = logging.getLogger(__name__)

_DIR_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
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


def make_verdict_rules(analyses: dict[str, dict]) -> TradeVerdict | None:
    """Simple weighted average. Returns None when the signal is ambiguous (abs(score) < 0.1)."""
    score = 0.0
    for a in analyses.values():
        d = _DIR_MAP.get(a.get("direction", "neutral"), 0.0)
        c = float(a.get("confidence", 0.0))
        score += d * c

    # Ambiguous — let LLM tiebreak decide
    if abs(score) < 0.1:
        return None

    action = "long" if score > 0 else "short" if score < 0 else "hold"
    if action != "hold":
        target = "bullish" if action == "long" else "bearish"
        agreeing = [float(a["confidence"]) for a in analyses.values() if a.get("direction") == target]
        confidence = sum(agreeing) / len(agreeing) if agreeing else 0.0
    else:
        confidence = 0.0

    return TradeVerdict(
        action=action, confidence=confidence, position_scale=max(0.0, confidence),
        reasoning=f"Weighted score={score:.3f}",
    )


TIEBREAK_PROMPT = """Four specialist agents analyzed a crypto market. Their analyses are below, but the weighted signal is ambiguous.

Your job: break the tie by evaluating argument quality across all four agents.

Consider:
- Which agents cited the most specific, verifiable data points?
- Are any agents flagging risks that others missed?
- Do the higher-confidence agents have stronger evidence, or just more assertive language?

Rules:
- Do NOT default to hold. The agents disagreed — you must pick the side with better evidence.
- Confidence should be moderate (0.4-0.6) since this is a tiebreak on ambiguous data.
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
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in verdict response: {text[:100]}")
        data = json.loads(text[start:end + 1])
        return TradeVerdict(
            action=_normalize_action(data.get("action", "hold")),
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
