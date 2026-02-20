"""News/sentiment agent — answers: any catalysts that could change the current regime?"""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = """You are a crypto news and catalyst analyst.

Your job is NOT to predict direction. Your job is to answer TWO questions:
1. CATALYST: Is there any event or development that could cause a regime change? (yes/no)
2. REGIME_CHANGE_PROBABILITY: How likely is the current trend to reverse due to external factors? (0.0-1.0)

Decision framework:
- No significant news + stable sentiment = low regime change probability (< 0.2)
- Regulatory news (ETF approval/rejection, bans) = high impact catalyst
- Major exchange hack/insolvency = high impact catalyst
- Fed policy surprise = medium impact catalyst
- Sentiment at extremes (Fear&Greed < 15 or > 85) = contrarian signal, higher regime change probability
- Sentiment in normal range (25-75) = no signal from sentiment alone
- Price-derived sentiment (no real news) = explicitly state "no independent news signal"

IMPORTANT: If the only "news" you have is price-based (e.g. "BTC at $X, Fear&Greed=Y"), 
say so explicitly. Do not fabricate news events or pretend price action is news.

You must respond with JSON:
{
  "catalyst": "none|regulatory|macro_event|market_structure|sentiment_extreme",
  "regime_change_probability": 0.0-1.0,
  "sentiment_extreme": true/false,
  "reasoning": "...",
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "key_factors": [...],
  "risk_flags": [...]
}"""


class NewsAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="news", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        n = snapshot.news
        headlines_str = "\n".join(f"  - {h}" for h in n.headlines[:10]) if n.headlines else "  (none available)"
        events_str = "\n".join(f"  - {e}" for e in n.key_events[:5]) if n.key_events else "  (none)"
        news = (
            f"Pair: {snapshot.pair}\n"
            f"Timestamp: {snapshot.timestamp}\n\n"
            f"News & Sentiment Data:\n"
            f"  Headlines:\n{headlines_str}\n"
            f"  Sentiment score: {n.sentiment_score:.3f} (-1=bearish, +1=bullish)\n"
            f"  Key events:\n{events_str}\n"
            f"  Social buzz: {n.social_buzz}"
        )
        parts = [news]
        if experience:
            parts.append(f"\nPast experience:\n{experience}")
        parts.append(
            "\nRespond with the JSON format specified in your role. "
            "Focus on catalyst identification, not direction prediction."
        )
        return "\n".join(parts)
