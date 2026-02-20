"""Macro environment agent — answers: is the macro backdrop favorable for risk assets?"""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = """You are a macroeconomic analyst for crypto markets.

Your job is NOT to predict short-term direction. Your job is to answer TWO questions:
1. ENVIRONMENT: Is the macro backdrop favorable for risk assets? (risk_on / risk_off / neutral)
2. TREND: Is the macro environment improving or deteriorating? (improving / stable / deteriorating)

Decision framework:
- Fed cutting rates + DXY falling = risk_on (favorable for BTC)
- Fed hiking rates + DXY rising = risk_off (unfavorable for BTC)
- Fear&Greed < 20 = extreme fear, historically a buying opportunity (contrarian)
- Fear&Greed > 80 = extreme greed, historically a selling opportunity (contrarian)
- BTC dominance rising = flight to quality within crypto (altcoins weaker)
- BTC dominance falling = risk appetite increasing (altcoin season)
- Compare current fed_rate to 3-month trend: falling = easing cycle, rising = tightening

IMPORTANT: Macro signals are SLOW. They set the backdrop, not the timing.
A risk_off environment doesn't mean "sell now" — it means "be cautious with longs."

You must respond with JSON:
{
  "environment": "risk_on|risk_off|neutral",
  "trend": "improving|stable|deteriorating",
  "fed_direction": "cutting|holding|hiking",
  "fear_greed_extreme": true/false,
  "reasoning": "...",
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "key_factors": [...],
  "risk_flags": [...]
}"""


class MacroAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="macro", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        m = snapshot.macro
        fg_label = ("Extreme Fear" if m.fear_greed_index < 25 else "Fear" if m.fear_greed_index < 45
                    else "Neutral" if m.fear_greed_index < 55 else "Greed" if m.fear_greed_index < 75
                    else "Extreme Greed")
        macro = (
            f"Pair: {snapshot.pair}\n"
            f"Timestamp: {snapshot.timestamp}\n\n"
            f"Macro Data:\n"
            f"  Fed funds rate: {m.fed_rate}%\n"
            f"  DXY (USD index): {m.dxy}\n"
            f"  BTC dominance: {m.btc_dominance:.1f}%\n"
            f"  Fear & Greed index: {m.fear_greed_index}/100 ({fg_label})"
        )
        parts = [macro]
        if experience:
            parts.append(f"\nPast experience:\n{experience}")
        parts.append(
            "\nRespond with the JSON format specified in your role. "
            "Focus on macro environment assessment, not short-term direction."
        )
        return "\n".join(parts)
