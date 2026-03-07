"""End-to-end integration test for skills."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.config import RiskConfig
from cryptotrader.models import TradeVerdict
from cryptotrader.risk.gate import RiskGate
from cryptotrader.risk.state import RedisStateManager


@pytest.mark.asyncio
async def test_risk_gate_with_token_security():
    """Test that RiskGate includes token security check."""
    # Setup
    config = RiskConfig()
    redis_state = MagicMock(spec=RedisStateManager)
    redis_state.get_daily_loss = AsyncMock(return_value=0.0)
    redis_state.get_last_trade_time = AsyncMock(return_value=None)
    redis_state.increment_rate_limit = AsyncMock(return_value=1)

    gate = RiskGate(config, redis_state)

    # Verify 12 checks (was 11 before)
    assert len(gate._checks) == 12

    # Verify token security check is present
    check_names = [c.name for c in gate._checks]
    assert "token_security" in check_names


@pytest.mark.asyncio
async def test_token_security_check_integration():
    """Test token security check in isolation."""
    from cryptotrader.models import CheckResult
    from cryptotrader.risk.checks.token_security import TokenSecurityCheck

    check = TokenSecurityCheck()

    # Test with verdict without contract address (should pass)
    verdict = TradeVerdict(action="long", confidence=0.8, reasoning="Test")

    result = await check.evaluate(verdict, {})
    assert isinstance(result, CheckResult)
    assert result.passed is True


@pytest.mark.asyncio
async def test_enhanced_data_provider():
    """Test enhanced data provider integration."""
    from cryptotrader.data.enhanced import EnhancedDataProvider

    provider = EnhancedDataProvider()

    # Test price data (may fail due to API, that's ok)
    price_data = await provider.get_price_data("BTC/USDT")
    assert "okx_available" in price_data

    # Test sentiment data
    sentiment_data = await provider.get_sentiment_data("BTC/USDT")
    assert "sentiment_available" in sentiment_data


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        await test_risk_gate_with_token_security()
        await test_token_security_check_integration()
        await test_enhanced_data_provider()

    asyncio.run(run_tests())
