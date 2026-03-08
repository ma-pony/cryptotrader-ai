"""Trading analysis skills - progressive disclosure pattern per LangChain docs."""

from typing import TypedDict


class Skill(TypedDict):
    """A skill that can be progressively disclosed to agents."""

    name: str
    description: str  # Lightweight description shown in system prompt
    content: str  # Full content loaded on-demand via tool call


TRADING_SKILLS: list[Skill] = [
    {
        "name": "funding_rate_analysis",
        "description": "Perpetual funding rate interpretation and position bias signals",
        "content": """# Funding Rate Analysis

Funding rates indicate position bias in perpetual futures markets.

## Signal Interpretation
- **>0.1%**: Extreme long bias, potential long squeeze risk
- **0.01-0.1%**: Healthy long bias, bullish sentiment
- **-0.01 to 0.01%**: Neutral, balanced positioning
- **<-0.01%**: Short bias, potential short squeeze risk

## Trading Implications
- Sustained high positive funding (>0.05% for 3+ days) often precedes corrections
- Negative funding during uptrends can signal capitulation, potential reversal
- Funding rate divergence from price action is a key contrarian signal

## Example
If BTC price rising but funding rate negative → shorts getting squeezed, strong bullish signal
""",
    },
    {
        "name": "btc_dominance_analysis",
        "description": "BTC dominance interpretation and altcoin market implications",
        "content": """# BTC Dominance Analysis

BTC dominance measures Bitcoin's market cap as % of total crypto market cap.

## Market Phases
- **>60%**: BTC-led rally, altcoins underperform (flight to safety)
- **50-60%**: Balanced market, BTC and alts move together
- **40-50%**: Early altseason, capital rotating to large caps
- **<40%**: Full altseason, speculative capital in alts

## Key Signals
- **Rising dominance + rising BTC price**: BTC-led bull market
- **Falling dominance + rising BTC price**: Altseason incoming (capital rotation)
- **Rising dominance + falling BTC price**: Risk-off, capital fleeing to BTC
- **Falling dominance + falling BTC price**: Bear market, alts bleeding harder

## Trading Implications
For BTC trading: High dominance (>60%) suggests BTC is the safe haven
For altcoin trading: Falling dominance during BTC rally is best entry signal
""",
    },
    {
        "name": "liquidation_cascade_analysis",
        "description": "Liquidation volume analysis and cascade risk assessment",
        "content": """# Liquidation Cascade Analysis

Large liquidation events can trigger cascading price moves.

## Volume Ratio Signals
- **>3x average**: Extreme liquidation event, potential reversal
- **2-3x average**: Significant liquidations, watch for follow-through
- **1-2x average**: Normal volatility
- **<1x average**: Low volatility, range-bound

## Cascade Risk
High liquidation volume + high funding rate = cascade risk
- Long liquidations → price drops → more longs liquidated → cascade down
- Short liquidations → price rises → more shorts liquidated → cascade up

## Trading Implications
After major liquidation event (>3x), expect:
- Short-term reversal (liquidations exhausted)
- Reduced volatility (leverage flushed out)
- Potential trend continuation after consolidation
""",
    },
    {
        "name": "fear_greed_interpretation",
        "description": "Fear & Greed Index interpretation and contrarian signals",
        "content": """# Fear & Greed Index Analysis

Sentiment indicator from 0 (extreme fear) to 100 (extreme greed).

## Ranges
- **0-25**: Extreme Fear - potential buying opportunity (contrarian)
- **25-45**: Fear - cautious sentiment
- **45-55**: Neutral
- **55-75**: Greed - market optimism
- **75-100**: Extreme Greed - potential top, take profits (contrarian)

## Contrarian Strategy
- Extreme Fear (<20) + technical support = strong buy signal
- Extreme Greed (>80) + technical resistance = strong sell signal
- Sustained extreme readings (5+ days) increase reversal probability

## Limitations
- Lagging indicator, reflects recent price action
- Works best at extremes (<25 or >75)
- Combine with technical analysis for confirmation
""",
    },
]


def get_skill_descriptions() -> str:
    """Get lightweight skill descriptions for system prompt."""
    return "\n".join(f"- {s['name']}: {s['description']}" for s in TRADING_SKILLS)


def load_skill_content(skill_name: str) -> str:
    """Load full skill content by name."""
    skill = next((s for s in TRADING_SKILLS if s["name"] == skill_name), None)
    if not skill:
        return f"Skill '{skill_name}' not found. Available: {', '.join(s['name'] for s in TRADING_SKILLS)}"
    return skill["content"]
