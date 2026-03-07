"""Demo script for skills integration."""

import asyncio
import os

from cryptotrader.data.binance_audit import BinanceAudit
from cryptotrader.data.binance_sentiment import BinanceSentiment


async def demo_token_security():
    """Demo token security audit."""
    print("\n=== Token Security Audit ===")
    audit = BinanceAudit()

    # Check USDT on BSC (should be safe)
    result = await audit.audit_token(
        symbol="USDT", contract_address="0x55d398326f99059ff775485246999027b3197955", chain="BSC"
    )

    print("Token: USDT")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Is Honeypot: {result['is_honeypot']}")
    print(f"Is Scam: {result['is_scam']}")
    print(f"Buy Tax: {result['buy_tax']}%")
    print(f"Sell Tax: {result['sell_tax']}%")
    if result["issues"]:
        print(f"Issues: {', '.join(result['issues'])}")


async def demo_okx_market():
    """Demo OKX market data."""
    print("\n=== OKX Market Data ===")

    # Check if credentials are configured
    if not all([os.getenv("OKX_API_KEY"), os.getenv("OKX_SECRET_KEY"), os.getenv("OKX_PASSPHRASE")]):
        print("OKX credentials not configured (skipped)")
        print("Set OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE to enable")
        return

    from cryptotrader.data.okx_market import OKXMarket

    okx = OKXMarket()

    # Get native OKB price on XLayer
    price = await okx.get_price("196", "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    print(f"OKB Price: ${price}" if price else "Price unavailable")

    # Get 24h candles
    candles = await okx.get_candles("196", "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "1H", 24)
    print(f"24h Candles: {len(candles)} data points")


async def demo_sentiment():
    """Demo Binance sentiment data."""
    print("\n=== Binance Sentiment ===")
    sentiment = BinanceSentiment()

    # Get social hype on BSC
    social = await sentiment.get_social_hype("56", limit=3)
    print(f"Social Hype Tokens: {len(social)}")

    # Get smart money signals on Solana
    signals = await sentiment.get_smart_money_signals("CT_501", page=1)
    print(f"Smart Money Signals: {len(signals)}")


async def main():
    """Run all demos."""
    await demo_token_security()
    await demo_okx_market()
    await demo_sentiment()
    print("\n✓ Skills integration demo complete")


if __name__ == "__main__":
    asyncio.run(main())
