"""Test supervisor pattern implementation."""

import asyncio
from datetime import UTC, datetime

import pandas as pd

from cryptotrader.graph import build_supervisor_graph_v2
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData


async def test_supervisor_flow():
    """Test the supervisor pattern graph with mock data."""
    print("Building supervisor graph...")
    graph = build_supervisor_graph_v2()
    print("✓ Graph built successfully")

    # Create mock snapshot
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now(UTC).timestamp() * 1000] * 10,
            "open": [50000] * 10,
            "high": [51000] * 10,
            "low": [49000] * 10,
            "close": [50500] * 10,
            "volume": [1000] * 10,
        }
    )

    snapshot = DataSnapshot(
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        market=MarketData(
            pair="BTC/USDT",
            ohlcv=df,
            ticker={"last": 50500, "baseVolume": 1000},
            funding_rate=0.01,
            orderbook_imbalance=0.1,
            volatility=0.02,
        ),
        onchain=OnchainData(open_interest=1000000, liquidations_24h={"volume_ratio": 1.5, "futures_volume": 500000}),
        news=NewsSentiment(sentiment_score=0.6, key_events=["BTC holding support"], headlines=["BTC at $50,500"]),
        macro=MacroData(fear_greed_index=55, btc_dominance=60.0, fed_rate=5.0, dxy=103.0),
    )

    print("\nRunning supervisor analysis...")
    print("Note: This requires OpenAI API key to be set")

    initial_state = {
        "messages": [],
        "data": {"snapshot": snapshot},
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "verdict_model": "gpt-4o-mini",
        },
        "debate_round": 0,
        "max_debate_rounds": 0,
        "divergence_scores": [],
    }

    try:
        result = await graph.ainvoke(initial_state)
        print("\n✓ Graph execution completed")

        verdict = result.get("data", {}).get("verdict", {})
        print("\nVerdict:")
        print(f"  Action: {verdict.get('action', 'N/A')}")
        print(f"  Confidence: {verdict.get('confidence', 0):.2%}")
        print(f"  Reasoning: {verdict.get('reasoning', 'N/A')[:100]}...")

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        print("This is expected if OpenAI API key is not configured")


if __name__ == "__main__":
    asyncio.run(test_supervisor_flow())
