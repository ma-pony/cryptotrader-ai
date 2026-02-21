"""News sentiment analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert crypto news and sentiment analyst. "
    "Analyze headlines, social sentiment, and key events to determine market direction."
)


class NewsAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(agent_id="news", role_description=ROLE, model=model)

    def _build_prompt(self, snapshot: DataSnapshot, experience: str) -> str:
        base = super()._build_prompt(snapshot, experience)
        n = snapshot.news
        headlines_str = "\n".join(f"  - {h}" for h in n.headlines[:10]) if n.headlines else "  (none available)"
        events_str = "\n".join(f"  - {e}" for e in n.key_events[:5]) if n.key_events else "  (none)"
        news = (
            f"Headlines:\n{headlines_str}\n"
            f"Sentiment score: {n.sentiment_score:.3f} (-1=bearish, +1=bullish)\n"
            f"Key events:\n{events_str}\n"
            f"Social buzz: {n.social_buzz}"
        )
        return f"News & Sentiment Data:\n{news}\n\n{base}"
