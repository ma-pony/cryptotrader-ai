"""Macro environment analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import BaseAgent

if TYPE_CHECKING:
    from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert macroeconomic analyst for cryptocurrency markets. "
    "Analyze interest rates, DXY, BTC dominance, fear/greed index, ETF fund flows, VIX, S&P 500, "
    "stablecoin supply, and BTC hashrate to determine market direction.\n\n"
    "Focus on: monetary policy regime (tightening vs easing cycle), dollar strength trend "
    "(DXY rising = headwind for crypto), risk appetite (fear/greed extremes as contrarian signals, "
    "VIX spikes = risk-off environment), equity market correlation (S&P 500 trend), "
    "capital rotation (BTC dominance rising = risk-off within crypto), institutional flows "
    "(ETF net inflows = institutional buying pressure, outflows = selling pressure), "
    "liquidity (stablecoin supply growth = dry powder for buying), and network health "
    "(hashrate = mining confidence and network security).\n"
    "Macro factors move slowly. Only flag a directional signal when the data shows a clear regime "
    "or an extreme reading. Moderate values in normal ranges should yield low confidence.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Regime vs noise: Is the Fed rate actually changing direction, or just holding? A hold is not a signal — "
    "don't manufacture one.\n"
    "- DXY confirmation: Does dollar strength/weakness confirm or contradict my crypto call? Bullish crypto + rising "
    "DXY is a conflict that needs explaining.\n"
    "- Fear/greed contrarian: Is the index below 25 or above 75? These extremes are contrarian — extreme fear is "
    "bullish, extreme greed is bearish. Mid-range values (30-70) carry no signal.\n"
    "- ETF flows: Large daily inflows (>$200M) are bullish institutional signal. Large outflows (>$200M) are bearish. "
    "Consecutive days of inflows/outflows carry more weight than a single day. Compare daily flow to cumulative AUM.\n"
    "- Moderate = low confidence: If all macro readings are in normal ranges, my confidence should be below 0.4. "
    "Normal macro does not justify a strong directional call."
)


class MacroAgent(BaseAgent):
    def __init__(self, model: str = "") -> None:
        super().__init__(agent_id="macro", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        m = snapshot.macro
        fg_label = (
            "Extreme Fear"
            if m.fear_greed_index < 25
            else "Fear"
            if m.fear_greed_index < 45
            else "Neutral"
            if m.fear_greed_index < 55
            else "Greed"
            if m.fear_greed_index < 75
            else "Extreme Greed"
        )
        parts = [
            f"Fed funds rate: {m.fed_rate}%",
            f"DXY (USD index): {m.dxy}",
            f"BTC dominance: {m.btc_dominance:.1f}%",
            f"Fear & Greed index: {m.fear_greed_index}/100 ({fg_label})",
        ]
        if m.etf_daily_net_inflow != 0 or m.etf_total_net_assets != 0:
            flow = m.etf_daily_net_inflow
            flow_label = "INFLOW" if flow > 0 else "OUTFLOW" if flow < 0 else "FLAT"
            parts.append(f"BTC ETF daily net flow: ${flow / 1e6:+,.1f}M ({flow_label})")
            parts.append(f"BTC ETF total AUM: ${m.etf_total_net_assets / 1e9:,.1f}B")
            parts.append(f"BTC ETF cumulative net inflow: ${m.etf_cum_net_inflow / 1e9:,.1f}B")
        if m.vix > 0:
            vix_label = "LOW" if m.vix < 15 else "NORMAL" if m.vix < 25 else "HIGH" if m.vix < 35 else "EXTREME"
            parts.append(f"VIX (volatility index): {m.vix:.1f} ({vix_label})")
        if m.sp500 > 0:
            parts.append(f"S&P 500: {m.sp500:,.1f}")
        if m.stablecoin_total_supply > 0:
            parts.append(f"Total stablecoin supply: ${m.stablecoin_total_supply / 1e9:,.1f}B")
        if m.btc_hashrate > 0:
            parts.append(f"BTC hashrate: {m.btc_hashrate / 1e9:,.1f} GH/s")
        macro = "\n".join(parts)
        return f"Macro Data:\n{macro}\n\n{base}"
