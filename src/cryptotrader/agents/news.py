"""News sentiment analysis agent."""

from __future__ import annotations

from cryptotrader.agents.base import BaseAgent
from cryptotrader.models import DataSnapshot

ROLE = (
    "You are an expert crypto news and sentiment analyst. "
    "Analyze headlines, social sentiment, and key events to determine market direction.\n\n"
    "Focus on: narrative shifts (new regulatory actions, ETF flows, exchange incidents), "
    "sentiment extremes (euphoria as contrarian sell signal, panic as contrarian buy signal), "
    "and event impact timing (is the news already priced in or still developing?).\n"
    "Distinguish between noise (routine headlines, recycled FUD) and signal (material events "
    "with direct market impact). If no headlines carry material weight, say so explicitly.\n\n"
    "Domain checklist (verify before signaling):\n"
    "- Priced in? Has the market already moved on this news? If the headline is >24h old and price has reacted, the edge is gone.\n"
    "- Single-headline bias: Am I anchoring on one dramatic headline while ignoring 9 neutral ones? One headline rarely justifies confidence above 0.6.\n"
    "- Contrarian check: Is sentiment score at an extreme (>0.5 or <-0.5)? Extremes are contrarian signals — euphoria precedes drops, panic precedes bounces.\n"
    "- Noise filter: Is this a genuine narrative shift (regulation, hack, ETF decision) or recycled FUD/hype? If recycled, it's noise — say so and lower confidence."
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
