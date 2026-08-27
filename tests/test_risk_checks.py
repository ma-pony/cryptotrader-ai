"""各硬风控检查器对目标仓位的业务契约。"""

from __future__ import annotations

import pytest

from cryptotrader.config import (
    CooldownConfig,
    ExchangeCheckConfig,
    LossConfig,
    PositionConfig,
    RateLimitConfig,
    VolatilityConfig,
)
from cryptotrader.decision.models import TargetPosition
from cryptotrader.risk.checks.available_margin import AvailableMargin
from cryptotrader.risk.checks.cooldown import CooldownCheck
from cryptotrader.risk.checks.cvar import CVaRCheck
from cryptotrader.risk.checks.exchange import ExchangeHealthCheck
from cryptotrader.risk.checks.loss import DailyLossLimit, DrawdownLimit
from cryptotrader.risk.checks.position import MaxPositionSize, MaxTotalExposure
from cryptotrader.risk.checks.rate_limit import RateLimitCheck
from cryptotrader.risk.checks.volatility import FundingRateGate, VolatilityGate
from cryptotrader.risk.state import RedisStateManager
from tests.factories.signal_fusion import position, risk_request


@pytest.fixture
def risk_input():
    return risk_request(target=TargetPosition("long", 0.5))


@pytest.fixture
def portfolio():
    return {
        "total_value": 10_000.0,
        "cash": 8_000.0,
        "free_cash": 8_000.0,
        "positions": {"ETH/USDT:USDT": {"amount": 1.0, "avg_price": 2_000.0}},
        "daily_pnl": -100.0,
        "drawdown": 0.05,
        "returns_60d": [
            -0.02,
            -0.03,
            0.01,
            -0.04,
            -0.05,
            0.02,
            -0.06,
            -0.01,
            0.03,
            -0.07,
            -0.01,
            0.02,
            -0.03,
            0.01,
            -0.02,
            0.04,
            -0.01,
            0.03,
            -0.05,
            0.02,
            -0.03,
            0.01,
            -0.02,
            0.01,
            -0.04,
        ],
        "recent_prices": [100, 99, 98, 97, 96],
        "funding_rate": 0.0005,
        "api_latency_ms": 500,
    }


@pytest.mark.asyncio
async def test_max_position_accepts_valid_target_ratio(risk_input, portfolio):
    result = await MaxPositionSize(PositionConfig(max_single_pct=0.1)).evaluate(risk_input, portfolio)

    assert result.passed is True


@pytest.mark.asyncio
async def test_total_exposure_passes_when_projected_target_fits(risk_input, portfolio):
    check = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.5, max_margin_used_pct=1.0))

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is True
    assert result.size_ratio_cap is None


@pytest.mark.asyncio
async def test_total_exposure_rejects_when_no_notional_budget(risk_input):
    check = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.3, max_margin_used_pct=1.0))
    portfolio = {"total_value": 10_000.0, "positions": {"ETH/USDT": 4_000.0}}

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is False
    assert "No remaining notional budget" in result.reason


@pytest.mark.asyncio
async def test_total_exposure_proposes_absolute_target_cap_without_mutating_request():
    check = MaxTotalExposure(PositionConfig(max_single_pct=0.5, max_total_exposure_pct=0.5, max_margin_used_pct=1.0))
    risk_input = risk_request(target=TargetPosition("long", 0.9))
    portfolio = {"total_value": 10_000.0, "positions": {"ETH/USDT": 2_000.0}}

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is True
    assert result.size_ratio_cap == pytest.approx(0.6)
    assert risk_input.target == TargetPosition("long", 0.9)


@pytest.mark.asyncio
async def test_total_exposure_replaces_current_pair_instead_of_double_counting():
    check = MaxTotalExposure(PositionConfig(max_single_pct=0.5, max_total_exposure_pct=0.6, max_margin_used_pct=1.0))
    risk_input = risk_request(
        current=position("long", amount=0.04, size_ratio=0.4),
        target=TargetPosition("long", 0.5),
    )
    portfolio = {
        "total_value": 10_000.0,
        "positions": {
            "BTC/USDT:USDT": {"amount": 0.04, "avg_price": 100_000.0},
            "ETH/USDT:USDT": {"amount": 1.0, "avg_price": 2_000.0},
        },
    }

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is True
    assert result.size_ratio_cap is None


@pytest.mark.asyncio
async def test_total_exposure_rejects_when_margin_budget_is_full(risk_input):
    check = MaxTotalExposure(
        PositionConfig(max_single_pct=0.5, max_total_exposure_pct=2.0, max_margin_used_pct=0.4),
        leverage=2,
    )
    portfolio = {
        "total_value": 10_000.0,
        "positions": {"ETH/USDT:USDT": {"amount": 4.0, "avg_price": 2_000.0}},
    }

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is False
    assert "No remaining margin budget" in result.reason


@pytest.mark.asyncio
async def test_available_margin_caps_only_incremental_same_direction_exposure():
    check = AvailableMargin(PositionConfig(max_single_pct=0.5), leverage=2, safety_buffer=1.0)
    risk_input = risk_request(
        current=position("long", amount=1.0, size_ratio=0.2),
        target=TargetPosition("long", 0.8),
    )
    portfolio = {"total_value": 10_000.0, "free_cash": 750.0}

    result = await check.evaluate(risk_input, portfolio)

    assert result.passed is True
    assert result.size_ratio_cap == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_available_margin_rejects_when_no_margin_is_available(risk_input):
    check = AvailableMargin(PositionConfig(max_single_pct=0.5), leverage=2)

    result = await check.evaluate(risk_input, {"total_value": 10_000.0, "free_cash": 0.0})

    assert result.passed is False
    assert "No free USDT" in result.reason


@pytest.mark.asyncio
async def test_daily_loss_limit_trips_and_stays_active(risk_input):
    check = DailyLossLimit(LossConfig(max_daily_loss_pct=0.03))

    first = await check.evaluate(risk_input, {"total_value": 10_000.0, "daily_pnl": -400.0})
    second = await check.evaluate(risk_input, {"total_value": 10_000.0, "daily_pnl": 0.0})

    assert first.passed is False
    assert second.passed is False
    assert "Circuit breaker" in second.reason


@pytest.mark.asyncio
async def test_daily_loss_limit_allows_unknown_daily_pnl(risk_input):
    check = DailyLossLimit(LossConfig(max_daily_loss_pct=0.03))

    result = await check.evaluate(risk_input, {"total_value": 10_000.0, "daily_pnl": None})

    assert result.passed is True
    assert result.reason == "daily_pnl unknown"


@pytest.mark.asyncio
async def test_drawdown_limit_rejects_excess_drawdown(risk_input):
    result = await DrawdownLimit(LossConfig(max_drawdown_pct=0.1)).evaluate(risk_input, {"drawdown": 0.15})

    assert result.passed is False


@pytest.mark.asyncio
async def test_cvar_rejects_excess_tail_loss(risk_input):
    returns = [-0.09, -0.08, -0.07, -0.06, -0.05] * 5
    result = await CVaRCheck(LossConfig(max_cvar_95=0.01)).evaluate(risk_input, {"returns_60d": returns})

    assert result.passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "prices", "passed", "message"),
    [
        (TargetPosition("long", 0.5), [100, 99, 98, 97, 93], False, "blocking long"),
        (TargetPosition("short", 0.5), [100, 99, 98, 97, 93], True, ""),
        (TargetPosition("short", 0.5), [93, 94, 95, 97, 100], False, "blocking short"),
        (TargetPosition("long", 0.5), [93, 94, 95, 97, 100], True, ""),
    ],
)
async def test_volatility_gate_is_directional(target, prices, passed, message):
    check = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))

    result = await check.evaluate(risk_request(target=target), {"recent_prices": prices})

    assert result.passed is passed
    assert message in result.reason


@pytest.mark.asyncio
async def test_funding_rate_gate_rejects_extreme_rate(risk_input):
    result = await FundingRateGate(VolatilityConfig(funding_rate_threshold=0.001)).evaluate(
        risk_input,
        {"funding_rate": 0.002},
    )

    assert result.passed is False


@pytest.mark.asyncio
async def test_exchange_health_rejects_latency_and_trade_cooldown(risk_input):
    check = ExchangeHealthCheck(ExchangeCheckConfig(max_api_latency_ms=2_000))

    latency = await check.evaluate(risk_input, {"api_latency_ms": 3_000})
    unavailable = await check.evaluate(risk_input, {"trade_unavailable_remaining_s": 45.0})

    assert latency.passed is False
    assert unavailable.passed is False
    assert "45s" in unavailable.reason


@pytest.mark.asyncio
async def test_cooldown_check_uses_request_pair(risk_input):
    redis = RedisStateManager(None)
    await redis.set_cooldown("BTC/USDT:USDT", 5)
    check = CooldownCheck(CooldownConfig(same_pair_minutes=5, post_loss_minutes=10), redis)

    result = await check.evaluate(risk_input, {})

    assert result.passed is False
    assert "BTC/USDT:USDT" in result.reason


@pytest.mark.asyncio
async def test_rate_limit_rejects_hourly_limit(risk_input):
    redis = RedisStateManager(None)
    for _ in range(2):
        await redis.incr_trade_count()
    check = RateLimitCheck(RateLimitConfig(max_trades_per_hour=2, max_trades_per_day=50), redis)

    result = await check.evaluate(risk_input, {})

    assert result.passed is False
    assert "Hourly" in result.reason
