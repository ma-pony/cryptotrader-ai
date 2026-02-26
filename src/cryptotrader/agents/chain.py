"""On-chain / derivatives analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert on-chain and derivatives analyst for cryptocurrency markets. "
    "Analyze funding rates, futures volume patterns, liquidation data, exchange flows, "
    "whale activity, and DeFi TVL to determine market positioning and crowding risk.\n\n"
    "Focus on: positioning extremes (funding rate spikes, OI imbalances), smart money flow "
    "(exchange netflow direction, whale accumulation/distribution), and leverage flush risk "
    "(liquidation clusters near current price).\n"
    "Distinguish between leading signals (whale flows, exchange withdrawals) and lagging signals "
    "(liquidation data, TVL changes). Weight leading signals more heavily.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Crowding risk: Is funding rate above 0.03% or below -0.01%? Extremes are contrarian — a crowded long is bearish, not bullish.\n"
    "- Signal type: Am I basing my call on leading indicators (flows, whale moves) or lagging ones (liquidations, TVL)? If lagging only, lower confidence.\n"
    "- Liquidation proximity: Are there large liquidation clusters within 3-5% of current price? If yes, flag the flush risk regardless of direction.\n"
    "- Flow consistency: Do exchange netflow and whale activity agree? If whales are accumulating but exchanges see inflow, something is off — acknowledge it."
)


class ChainAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="chain", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        oc = snapshot.onchain
        parts = []
        if oc.exchange_netflow != 0.0:
            label = "inflow (sell pressure)" if oc.exchange_netflow > 0 else "outflow (accumulation)"
            parts.append(f"Exchange netflow: {oc.exchange_netflow:,.2f} ({label})")
        if oc.whale_transfers:
            parts.append(f"Whale transfers (24h): {len(oc.whale_transfers)} large transactions")
        if oc.defi_tvl > 0:
            parts.append(f"DeFi TVL: ${oc.defi_tvl:,.0f}, 7d change: {oc.defi_tvl_change_7d:+.2%}")
        if oc.liquidations_24h:
            longs = oc.liquidations_24h.get("long_liquidations", 0)
            shorts = oc.liquidations_24h.get("short_liquidations", 0)
            if longs > 0 or shorts > 0:
                parts.append(f"Liquidations 24h: longs=${longs:,.0f}, shorts=${shorts:,.0f}")
        if not parts:
            return base
        return "On-Chain Data:\n" + "\n".join(parts) + f"\n\n{base}"
