"""Macro environment analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert macroeconomic analyst for cryptocurrency markets. "
    "Analyze interest rates, DXY, BTC dominance, and fear/greed index "
    "to determine market direction."
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
