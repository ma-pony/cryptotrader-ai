"""On-chain / derivatives analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert on-chain and derivatives analyst for cryptocurrency markets. "
    "Analyze funding rates, open interest, exchange flows, and liquidation data "
    "to determine market direction."
)


class ChainAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="chain", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        chain_data = (
            f"Funding rate: {snapshot.market.funding_rate}\n"
            f"Open interest: {snapshot.onchain.open_interest}\n"
            f"Exchange netflow: {snapshot.onchain.exchange_netflow}\n"
            f"Liquidations 24h: {snapshot.onchain.liquidations_24h}\n"
            f"Whale transfers: {len(snapshot.onchain.whale_transfers)}"
        )
        return f"On-Chain / Derivatives Data:\n{chain_data}\n\n{base}"
