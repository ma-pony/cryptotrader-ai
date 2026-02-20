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
        macro = (
            f"Fed rate: {snapshot.macro.fed_rate}\n"
            f"DXY: {snapshot.macro.dxy}\n"
            f"BTC dominance: {snapshot.macro.btc_dominance}\n"
            f"Fear & Greed index: {snapshot.macro.fear_greed_index}"
        )
        return f"Macro Data:\n{macro}\n\n{base}"
