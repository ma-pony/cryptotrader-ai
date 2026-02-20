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
        news = (
            f"Headlines: {snapshot.news.headlines}\n"
            f"Sentiment score: {snapshot.news.sentiment_score}\n"
            f"Key events: {snapshot.news.key_events}\n"
            f"Social buzz: {snapshot.news.social_buzz}"
        )
        return f"News & Sentiment Data:\n{news}\n\n{base}"
