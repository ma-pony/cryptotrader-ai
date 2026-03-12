"""Macro environment analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import BaseAgent

if TYPE_CHECKING:
    from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert macroeconomic analyst for cryptocurrency markets. "
    "Analyze interest rates, DXY, BTC dominance, fear/greed index, ETF fund flows, VIX, S&P 500, "
    "stablecoin supply, BTC hashrate, yield curve, M2 money supply, and CPI to determine market direction.\n\n"
    "Focus on: monetary policy regime (tightening vs easing cycle), dollar strength trend "
    "(DXY rising = headwind for crypto), risk appetite (fear/greed extremes as contrarian signals, "
    "VIX spikes = risk-off environment), equity market correlation (S&P 500 trend), "
    "capital rotation (BTC dominance rising = risk-off within crypto), institutional flows "
    "(ETF net inflows = institutional buying pressure, outflows = selling pressure), "
    "liquidity (stablecoin supply growth = dry powder for buying, M2 expansion = more money sloshing), "
    "network health (hashrate = mining confidence and network security), "
    "yield curve shape (positive = normal economy, inverted = recession risk ahead), "
    "and inflation regime (CPI trend signals whether Fed is likely to ease or tighten).\n"
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
    "- Yield curve: Inverted curve (negative T10Y2Y) historically precedes recessions — risk-off signal for crypto. "
    "Curve normalizing (moving from negative toward positive) can signal macro recovery.\n"
    "- Moderate = low confidence: If all macro readings are in normal ranges, my confidence should be below 0.4. "
    "Normal macro does not justify a strong directional call."
)


def _format_fear_greed_trend(history: list[int]) -> str:
    """Format 7-day Fear & Greed history as a trend string with direction label."""
    if not history or len(history) < 2:
        return ""
    # API returns newest-first; reverse to get chronological order
    chron = list(reversed(history))
    trend_str = "→".join(str(v) for v in chron)
    delta = chron[-1] - chron[0]
    if delta > 10:
        direction = "recovering"
    elif delta < -10:
        direction = "deteriorating"
    else:
        direction = "stable"
    return f"7d trend: {trend_str} ({direction})"


def _format_etf_top_flows(top_flows: list[dict]) -> str:
    """Format per-ticker ETF flows as a compact string."""
    if not top_flows:
        return ""
    parts = []
    for item in top_flows:
        ticker = item.get("ticker", "?")
        flow = item.get("dailyNetInflow", 0)
        sign = "+" if flow >= 0 else ""
        parts.append(f"{ticker} {sign}${flow / 1e6:,.0f}M")
    return ", ".join(parts)


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
        ]
        _append_fear_greed(parts, m, fg_label)
        _append_etf_data(parts, m)
        _append_market_indicators(parts, m)
        _append_macro_fundamentals(parts, m)
        macro = "\n".join(parts)
        return f"Macro Data:\n{macro}\n\n{base}"


def _append_fear_greed(parts: list[str], m, fg_label: str) -> None:
    """Append Fear & Greed index and optional 7-day trend."""
    fg_line = f"Fear & Greed index: {m.fear_greed_index}/100 ({fg_label})"
    trend = _format_fear_greed_trend(m.fear_greed_history)
    if trend:
        fg_line += f" | {trend}"
    parts.append(fg_line)


def _append_etf_data(parts: list[str], m) -> None:
    """Append BTC ETF flow data including per-ticker breakdown."""
    if m.etf_daily_net_inflow != 0 or m.etf_total_net_assets != 0:
        flow = m.etf_daily_net_inflow
        flow_label = "INFLOW" if flow > 0 else "OUTFLOW" if flow < 0 else "FLAT"
        parts.append(f"BTC ETF daily net flow: ${flow / 1e6:+,.1f}M ({flow_label})")
        parts.append(f"BTC ETF total AUM: ${m.etf_total_net_assets / 1e9:,.1f}B")
        parts.append(f"BTC ETF cumulative net inflow: ${m.etf_cum_net_inflow / 1e9:,.1f}B")
        ticker_flows = _format_etf_top_flows(m.etf_top_flows)
        if ticker_flows:
            parts.append(f"BTC ETF top flows: {ticker_flows}")


def _append_market_indicators(parts: list[str], m) -> None:
    """Append VIX and S&P 500 data."""
    if m.vix > 0:
        vix_label = "LOW" if m.vix < 15 else "NORMAL" if m.vix < 25 else "HIGH" if m.vix < 35 else "EXTREME"
        parts.append(f"VIX (volatility index): {m.vix:.1f} ({vix_label})")
    if m.sp500 > 0:
        parts.append(f"S&P 500: {m.sp500:,.1f}")


def _append_macro_fundamentals(parts: list[str], m) -> None:
    """Append stablecoin supply, hashrate, yield curve, M2, and CPI."""
    if m.stablecoin_total_supply > 0:
        parts.append(f"Total stablecoin supply: ${m.stablecoin_total_supply / 1e9:,.1f}B")
    if m.btc_hashrate > 0:
        parts.append(f"BTC hashrate: {m.btc_hashrate / 1e9:,.1f} GH/s")
    if m.yield_curve != 0:
        yc_label = "NORMAL (positive slope)" if m.yield_curve > 0 else "INVERTED (recession warning)"
        parts.append(f"T10Y2Y yield curve spread: {m.yield_curve:+.2f}% ({yc_label})")
    if m.m2_supply > 0:
        parts.append(f"M2 money supply: ${m.m2_supply / 1e3:,.1f}T")
    if m.cpi > 0:
        parts.append(f"CPI (inflation index): {m.cpi:.1f}")
