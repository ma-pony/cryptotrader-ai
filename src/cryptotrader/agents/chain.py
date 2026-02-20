"""On-chain / derivatives agent — answers: is positioning crowded? extreme levels?"""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = """You are a derivatives and on-chain positioning analyst.

Your job is to assess positioning crowding and extreme levels from derivatives data.

Decision framework:
- Funding rate > 0.03% (8h) = longs crowded → contrarian bearish
- Funding rate < -0.01% = shorts crowded → contrarian bullish  
- Funding rate between -0.01% and 0.03% = balanced, no signal
- Extreme funding (|rate| > 0.05%) = high probability of mean reversion

CRITICAL: If open_interest=0 and liquidations are empty, those fields have NO DATA.
Do NOT interpret zero values as "low activity". Say "no_data" explicitly.
Only the funding rate is reliable when other fields are zero.

Respond with JSON:
{
  "crowding": "longs_crowded|shorts_crowded|balanced|no_data",
  "extreme": true/false,
  "funding_signal": "contrarian_bullish|contrarian_bearish|neutral",
  "data_quality": "full|funding_only|none",
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "key_factors": [...],
  "risk_flags": [...]
}"""


class ChainAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="chain", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        chain_data = (
            f"Pair: {snapshot.pair}\n"
            f"Timestamp: {snapshot.timestamp}\n\n"
            f"Derivatives Data:\n"
            f"  Funding rate (8h): {snapshot.market.funding_rate}\n"
            f"  Open interest: {snapshot.onchain.open_interest}\n"
            f"  Exchange netflow: {snapshot.onchain.exchange_netflow}\n"
            f"  Liquidations 24h: {snapshot.onchain.liquidations_24h}\n"
            f"  Whale transfers: {len(snapshot.onchain.whale_transfers)}"
        )
        parts = [chain_data]
        if experience:
            parts.append(f"\nPast experience:\n{experience}")
        parts.append(
            "\nRespond with the JSON format specified in your role. "
            "Focus on positioning analysis, not direction prediction."
        )
        return "\n".join(parts)
