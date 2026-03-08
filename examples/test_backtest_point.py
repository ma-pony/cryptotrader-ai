"""Quick test of supervisor pattern on single timepoint."""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData


async def test_single_point():
    """Test supervisor on single data point."""
    print("Creating mock snapshot...")

    # Mock OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now(UTC).timestamp() * 1000] * 100,
            "open": [50000 + i * 10 for i in range(100)],
            "high": [50100 + i * 10 for i in range(100)],
            "low": [49900 + i * 10 for i in range(100)],
            "close": [50050 + i * 10 for i in range(100)],
            "volume": [1000] * 100,
        }
    )

    snapshot = DataSnapshot(
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        market=MarketData(
            pair="BTC/USDT",
            ohlcv=df,
            ticker={"last": 51050, "baseVolume": 1000},
            funding_rate=0.015,
            orderbook_imbalance=0.15,
            volatility=0.025,
        ),
        onchain=OnchainData(open_interest=1200000, liquidations_24h={"volume_ratio": 2.1, "futures_volume": 600000}),
        news=NewsSentiment(
            sentiment_score=0.65, key_events=["BTC breaks resistance"], headlines=["BTC rallies to $51k"]
        ),
        macro=MacroData(fear_greed_index=62, btc_dominance=58.5, fed_rate=5.0, dxy=102.5),
    )

    print("✓ Snapshot created")
    print(f"  Price: ${snapshot.market.ticker['last']:,.0f}")
    print(f"  Funding: {snapshot.market.funding_rate:.3%}")
    print(f"  Fear&Greed: {snapshot.macro.fear_greed_index}")

    print("\nNote: Full test requires OpenAI API key")
    print("Supervisor would coordinate: tech_agent, chain_agent, macro_agent")
    print("Expected: Progressive disclosure of skills as needed")


if __name__ == "__main__":
    asyncio.run(test_single_point())
