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

_VALID_ACTIONS = {"long", "short", "hold", "close"}


def _normalize_action(raw: str) -> str:
    """Map LLM action strings to valid actions."""
    raw = raw.strip().lower()
    if raw in _VALID_ACTIONS:
        return raw
    if raw in ("buy", "bullish"):
        return "long"
    if raw in ("sell", "bearish"):
        return "short"
    if raw in ("exit", "flatten", "close_position"):
        return "close"
    return "hold"


VERDICT_PROMPT = """You are an aggressive crypto portfolio manager. Four specialist agents analyzed the market.
Your edge comes from DECISIVE action and PROPER SIZING — not from being cautious.

CORE PHILOSOPHY: In crypto, the cost of missing a move is higher than the cost of a small wrong bet.
Flat positions earn nothing. Every moment flat in a trending market is an opportunity cost.

ACTIONS:
- "long": Open/add to long position. Set position_scale to match your conviction.
- "short": Open/add to short position. Set position_scale to match your conviction.
- "close": Exit current position. Only when thesis is clearly invalidated.
- "hold": Keep current state. Use SPARINGLY — this is your LEAST useful action.
  "hold" while flat = guaranteed zero return. Prefer a small position over flat.

DECISION PROCESS:
1. Check POSITION STATE — are you FLAT, LONG, or SHORT?
2. Read PRICE CONTEXT — what is the dominant trend? Trends persist more than they reverse.
3. Evaluate agents — weight by data_sufficiency. "low" sufficiency = discount heavily.
4. ACT — if any directional signal exists, trade it with appropriate size.

WHEN FLAT — BE BIASED TOWARD ENTERING:
- 1 agent with strong thesis + confirming price trend → ENTER (scale 0.4-0.6).
- 2+ agents agree on direction → ENTER decisively (scale 0.6-0.8).
- 3+ agents agree + strong trend → ENTER aggressively (scale 0.8-1.0).
- Stay flat ONLY when: agents are perfectly split 2v2 AND price is range-bound AND no clear trend.
- A clear price trend (>3% over 7d) is itself a signal — trade in the trend direction even with mixed agents.

WHEN IN POSITION:
- Thesis intact + trend continues → HOLD or ADD (increase position_scale).
- Agents flip direction + price confirms reversal → CLOSE immediately, then consider reversing.
- Small unrealized loss (<5%) in a confirmed trend → HOLD. This is normal noise.
- BTC has 5-10% swings in strong trends. Do NOT panic-close on every dip.

POSITION SIZING (position_scale) — THIS IS CRITICAL:
- position_scale = fraction of capital to deploy. YOU control risk through sizing.
- MATCH scale to confidence: confidence 0.65 → scale ~0.65. Do NOT default to 0.4.
- 0.3-0.5: weak signal, exploratory. Only 1 agent agrees, mixed price action.
- 0.5-0.7: moderate signal. 2 agents agree, visible trend.
- 0.7-0.9: strong signal. 3+ agents agree, confirmed trend, good data.
- 0.9-1.0: overwhelming. All agents aligned, extreme trend, high-conviction setup.
- NEVER use 0.4 as a default. Think about WHY you chose that scale.
- For "hold" action, set position_scale to 0.0 (it is ignored anyway).

ANTI-PATTERNS TO AVOID:
- Do NOT hold flat through a trending market. Missing a 10% move costs more than a 2% wrong entry.
- Do NOT always use scale=0.4. If your confidence is 0.7, your scale should be 0.7.
- Do NOT require all 4 agents to agree. 2-3 is enough for a strong trade.
- Do NOT close winning positions just because one agent turns neutral.

Output ONLY JSON:
{
  "action": "long|short|hold|close",
  "confidence": 0.0-1.0,
  "position_scale": 0.0-1.0,
  "reasoning": "2-3 sentences explaining your decision",
  "thesis": "one sentence trade thesis (or exit rationale if closing)",
  "invalidation": "specific condition that would invalidate this decision"
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


def _format_position_context(position_context: dict | None) -> str:
    """Format current position state for verdict prompt."""
    if not position_context:
        return "Current position: FLAT (no open position)"
    side = position_context.get("side", "flat")
    if side == "flat":
        last_ctx = position_context.get("last_action_context", "")
        if last_ctx:
            return f"Current position: FLAT (no open position)\n  Previous verdict: {last_ctx}"
        return "Current position: FLAT (no open position)"
    entry = position_context.get("entry_price", 0)
    current = position_context.get("current_price", 0)
    days = position_context.get("days_held", 0)
    if entry > 0 and current > 0:
        pnl_pct = (current - entry) / entry if side == "long" else (entry - current) / entry
    else:
        pnl_pct = 0.0
    pnl_label = f"{pnl_pct:+.1%}"
    parts = [
        f"Current position: {side.upper()}",
        f"  Entry price: ${entry:,.2f} | Current price: ${current:,.2f}",
        f"  Unrealized P&L: {pnl_label}",
        f"  Days held: {days}",
    ]
    last_ctx = position_context.get("last_action_context", "")
    if last_ctx:
        parts.append(f"  Previous verdict: {last_ctx}")
    return "\n".join(parts)


def _format_trend_context(trend_context: dict | None) -> str:
    """Format price trend summary for verdict prompt."""
    if not trend_context:
        return "Price context: unavailable"
    parts = ["Price context:"]
    for period in ("7d", "14d", "30d"):
        val = trend_context.get(f"change_{period}")
        if val is not None:
            parts.append(f"  {period} change: {val:+.1%}")
    high = trend_context.get("high_30d")
    low = trend_context.get("low_30d")
    if high and low:
        parts.append(f"  30d range: ${low:,.0f} — ${high:,.0f}")
        current = trend_context.get("current_price", 0)
        if current and high > low:
            pct_in_range = (current - low) / (high - low)
            parts.append(f"  Current at {pct_in_range:.0%} of 30d range")
    return "\n".join(parts)


async def make_verdict_ai(
    analyses: dict[str, dict],
    constraints: dict | None = None,
    calibration: str = "",
    model: str = "",
    position_context: dict | None = None,
    trend_context: dict | None = None,
) -> TradeVerdict:
    """AI-driven verdict that evaluates argument quality with full context."""
    # Format full agent reports (all fields, not just direction+confidence)
    agent_reports = "\n\n".join(
        f"── {aid.upper()} ──\n{json.dumps(a, indent=2, default=str)}" for aid, a in analyses.items()
    )

    position_block = _format_position_context(position_context)
    trend_block = _format_trend_context(trend_context)
    constraint_block = _format_constraints(constraints or {})
    calibration_block = f"\n\n{calibration}\n" if calibration else ""

    user_msg = f"""POSITION STATE:
{position_block}

PRICE CONTEXT:
{trend_block}

RISK CONSTRAINTS (hard limits — you cannot exceed these):
{constraint_block}
{calibration_block}
AGENT ANALYSES:
{agent_reports}"""

    try:
        llm = create_llm(model=model, temperature=0.1, json_mode=True)
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

        verdict = TradeVerdict(
            action=action,
            confidence=confidence,
            position_scale=max(0.0, min(1.0, float(data.get("position_scale", confidence)))),
            reasoning=data.get("reasoning", ""),
            thesis=data.get("thesis", ""),
            invalidation=data.get("invalidation", ""),
        )
        logger.info(
            "AI verdict: action=%s confidence=%.2f scale=%.2f thesis=%s",
            verdict.action,
            verdict.confidence,
            verdict.position_scale,
            (verdict.thesis or "")[:100],
        )
        return verdict
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
    position_context: dict | None = None,
    trend_context: dict | None = None,
) -> TradeVerdict:
    """Main entry point — routes to AI verdict."""
    return await make_verdict_ai(
        analyses,
        constraints=constraints,
        calibration=calibration,
        model=model,
        position_context=position_context,
        trend_context=trend_context,
    )


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
