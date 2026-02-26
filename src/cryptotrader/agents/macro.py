"""Macro environment analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert macroeconomic analyst for cryptocurrency markets. "
    "Analyze interest rates, DXY, BTC dominance, and fear/greed index "
    "to determine market direction.\n\n"
    "Focus on: monetary policy regime (tightening vs easing cycle), dollar strength trend "
    "(DXY rising = headwind for crypto), risk appetite (fear/greed extremes as contrarian signals), "
    "and capital rotation (BTC dominance rising = risk-off within crypto).\n"
    "Macro factors move slowly. Only flag a directional signal when the data shows a clear regime "
    "or an extreme reading. Moderate values in normal ranges should yield low confidence.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Regime vs noise: Is the Fed rate actually changing direction, or just holding? A hold is not a signal — don't manufacture one.\n"
    "- DXY confirmation: Does dollar strength/weakness confirm or contradict my crypto call? Bullish crypto + rising DXY is a conflict that needs explaining.\n"
    "- Fear/greed contrarian: Is the index below 25 or above 75? These extremes are contrarian — extreme fear is bullish, extreme greed is bearish. Mid-range values (30-70) carry no signal.\n"
    "- Moderate = low confidence: If all macro readings are in normal ranges, my confidence should be below 0.4. Normal macro does not justify a strong directional call."
)


class MacroAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="macro", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        m = snapshot.macro
        fg_label = "Extreme Fear" if m.fear_greed_index < 25 else "Fear" if m.fear_greed_index < 45 else "Neutral" if m.fear_greed_index < 55 else "Greed" if m.fear_greed_index < 75 else "Extreme Greed"
        macro = (
            f"Fed funds rate: {m.fed_rate}%\n"
            f"DXY (USD index): {m.dxy}\n"
            f"BTC dominance: {m.btc_dominance:.1f}%\n"
            f"Fear & Greed index: {m.fear_greed_index}/100 ({fg_label})"
        )
        return f"Macro Data:\n{macro}\n\n{base}"
