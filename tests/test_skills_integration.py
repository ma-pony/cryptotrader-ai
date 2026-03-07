"""Tests for skills integration."""

import os

import pytest

from cryptotrader.data.binance_audit import BinanceAudit
from cryptotrader.data.binance_sentiment import BinanceSentiment


@pytest.mark.asyncio
async def test_binance_audit():
    """Test Binance token audit."""
    audit = BinanceAudit()
    # Test with USDT on BSC (known safe token)
    result = await audit.audit_token(
        symbol="USDT", contract_address="0x55d398326f99059ff775485246999027b3197955", chain="BSC"
    )
    assert "risk_level" in result
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


@pytest.mark.asyncio
async def test_okx_market():
    """Test OKX market data (skipped if no credentials)."""
    if not all([os.getenv("OKX_API_KEY"), os.getenv("OKX_SECRET_KEY"), os.getenv("OKX_PASSPHRASE")]):
        pytest.skip("OKX credentials not configured")

    from cryptotrader.data.okx_market import OKXMarket

    okx = OKXMarket()
    # Test with native token on XLayer
    price = await okx.get_price("196", "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    assert price is None or isinstance(price, float)


@pytest.mark.asyncio
async def test_binance_sentiment():
    """Test Binance sentiment data."""
    sentiment = BinanceSentiment()
    social = await sentiment.get_social_hype(chain_id="56", limit=5)
    assert isinstance(social, list)
