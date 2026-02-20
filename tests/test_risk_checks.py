"""Tests for risk checks with edge cases."""

import pytest
from cryptotrader.models import TradeVerdict, CheckResult
from cryptotrader.config import (
    PositionConfig, LossConfig, VolatilityConfig,
    CooldownConfig, ExchangeCheckConfig, RateLimitConfig,
)
from cryptotrader.risk.checks.position import MaxPositionSize, MaxTotalExposure
from cryptotrader.risk.checks.loss import DailyLossLimit, DrawdownLimit
from cryptotrader.risk.checks.cvar import CVaRCheck
from cryptotrader.risk.checks.volatility import VolatilityGate, FundingRateGate
from cryptotrader.risk.checks.exchange import ExchangeHealthCheck


@pytest.fixture
def verdict():
    return TradeVerdict(action="long", confidence=0.7, position_scale=0.05)


@pytest.fixture
def portfolio():
    return {
        "total_value": 10000,
        "positions": {"BTC/USDT": 2000},
        "daily_pnl": -100,
        "drawdown": 0.05,
        "returns_60d": [-0.02, -0.03, 0.01, -0.04, -0.05, 0.02, -0.06, -0.01, 0.03, -0.07],
        "recent_prices": [100, 99, 98, 97, 96],
        "funding_rate": 0.0005,
        "api_latency_ms": 500,
    }


# ── Position checks ──

@pytest.mark.asyncio
async def test_max_position_pass(verdict, portfolio):
    c = MaxPositionSize(PositionConfig(max_single_pct=0.10))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_max_position_fail(portfolio):
    v = TradeVerdict(action="long", position_scale=1.5)
    c = MaxPositionSize(PositionConfig(max_single_pct=0.10))
    r = await c.evaluate(v, portfolio)
    assert not r.passed

@pytest.mark.asyncio
async def test_max_position_zero_portfolio(verdict):
    c = MaxPositionSize(PositionConfig())
    r = await c.evaluate(verdict, {"total_value": 0})
    assert not r.passed

@pytest.mark.asyncio
async def test_total_exposure_pass(verdict, portfolio):
    c = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.50))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_total_exposure_fail(verdict):
    c = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.30))
    r = await c.evaluate(verdict, {"total_value": 10000, "positions": {"A": 4000}})
    assert not r.passed


# ── Loss checks ──

@pytest.mark.asyncio
async def test_daily_loss_pass(verdict, portfolio):
    c = DailyLossLimit(LossConfig(max_daily_loss_pct=0.03))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_daily_loss_fail(verdict):
    c = DailyLossLimit(LossConfig(max_daily_loss_pct=0.03))
    r = await c.evaluate(verdict, {"total_value": 10000, "daily_pnl": -400})
    assert not r.passed

@pytest.mark.asyncio
async def test_circuit_breaker_stays_active(verdict):
    c = DailyLossLimit(LossConfig(max_daily_loss_pct=0.03))
    await c.evaluate(verdict, {"total_value": 10000, "daily_pnl": -400})
    r = await c.evaluate(verdict, {"total_value": 10000, "daily_pnl": 0})
    assert not r.passed  # breaker still active

@pytest.mark.asyncio
async def test_drawdown_pass(verdict, portfolio):
    c = DrawdownLimit(LossConfig(max_drawdown_pct=0.10))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_drawdown_fail(verdict):
    c = DrawdownLimit(LossConfig(max_drawdown_pct=0.10))
    r = await c.evaluate(verdict, {"drawdown": 0.15})
    assert not r.passed


# ── CVaR ──

@pytest.mark.asyncio
async def test_cvar_pass(verdict, portfolio):
    c = CVaRCheck(LossConfig(max_cvar_95=0.10))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_cvar_fail(verdict):
    c = CVaRCheck(LossConfig(max_cvar_95=0.01))
    r = await c.evaluate(verdict, {"returns_60d": [-0.05, -0.06, -0.07, -0.08, -0.02, 0.01]})
    assert not r.passed

@pytest.mark.asyncio
async def test_cvar_insufficient_data(verdict):
    c = CVaRCheck(LossConfig())
    r = await c.evaluate(verdict, {"returns_60d": [0.01]})
    assert r.passed


# ── Volatility ──

@pytest.mark.asyncio
async def test_volatility_pass(verdict, portfolio):
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_volatility_flash_crash(verdict):
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    r = await c.evaluate(verdict, {"recent_prices": [100, 99, 98, 97, 93]})
    assert not r.passed

@pytest.mark.asyncio
async def test_funding_rate_pass(verdict, portfolio):
    c = FundingRateGate(VolatilityConfig(funding_rate_threshold=0.001))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_funding_rate_fail(verdict):
    c = FundingRateGate(VolatilityConfig(funding_rate_threshold=0.001))
    r = await c.evaluate(verdict, {"funding_rate": 0.002})
    assert not r.passed


# ── Exchange health ──

@pytest.mark.asyncio
async def test_exchange_health_pass(verdict, portfolio):
    c = ExchangeHealthCheck(ExchangeCheckConfig(max_api_latency_ms=2000))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed

@pytest.mark.asyncio
async def test_exchange_health_fail(verdict):
    c = ExchangeHealthCheck(ExchangeCheckConfig(max_api_latency_ms=2000))
    r = await c.evaluate(verdict, {"api_latency_ms": 3000})
    assert not r.passed
