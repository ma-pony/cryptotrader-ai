"""Tests for risk checks with edge cases."""

import pytest

from cryptotrader.config import (
    CooldownConfig,
    ExchangeCheckConfig,
    LossConfig,
    PositionConfig,
    RateLimitConfig,
    VolatilityConfig,
)
from cryptotrader.models import TradeVerdict
from cryptotrader.risk.checks.cooldown import CooldownCheck
from cryptotrader.risk.checks.cvar import CVaRCheck
from cryptotrader.risk.checks.exchange import ExchangeHealthCheck
from cryptotrader.risk.checks.loss import DailyLossLimit, DrawdownLimit
from cryptotrader.risk.checks.position import MaxPositionSize, MaxTotalExposure
from cryptotrader.risk.checks.rate_limit import RateLimitCheck
from cryptotrader.risk.checks.volatility import FundingRateGate, VolatilityGate
from cryptotrader.risk.state import RedisStateManager


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


# ── Position checks ──


@pytest.mark.asyncio
async def test_max_position_pass(verdict, portfolio):
    c = MaxPositionSize(PositionConfig(max_single_pct=0.10))
    r = await c.evaluate(verdict, portfolio)
    assert r.passed


@pytest.mark.asyncio
async def test_max_position_always_passes(portfolio):
    """MaxPositionSize always passes — scale is clamped to [0,1] by TradeVerdict,
    so target can never exceed max_pct. Execution layer handles delta."""
    portfolio_with_pair = {**portfolio, "pair": "BTC/USDT"}
    v = TradeVerdict(action="long", position_scale=1.0)
    c = MaxPositionSize(PositionConfig(max_single_pct=0.10))
    r = await c.evaluate(v, portfolio_with_pair)
    assert r.passed


@pytest.mark.asyncio
async def test_max_position_zero_portfolio(verdict):
    """Cold start: zero portfolio should allow first trade (not reject)."""
    c = MaxPositionSize(PositionConfig())
    r = await c.evaluate(verdict, {"total_value": 0})
    assert r.passed


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
    assert "No remaining exposure budget" in r.reason


@pytest.mark.asyncio
async def test_total_exposure_hold_always_passes():
    """Hold action should pass even with high exposure (no new position added)."""
    hold = TradeVerdict(action="hold", confidence=0.7, position_scale=0.0)
    c = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.30))
    r = await c.evaluate(hold, {"total_value": 10000, "positions": {"A": 9000}})
    assert r.passed


@pytest.mark.asyncio
async def test_total_exposure_close_always_passes():
    """Close action should pass even with high exposure (reducing position)."""
    close = TradeVerdict(action="close", confidence=0.7, position_scale=0.0)
    c = MaxTotalExposure(PositionConfig(max_total_exposure_pct=0.30))
    r = await c.evaluate(close, {"total_value": 10000, "positions": {"A": 9000}})
    assert r.passed


@pytest.mark.asyncio
async def test_total_exposure_projected_clamps():
    """PROD-I3: Over-budget trade returns a scale_adjustment proposal (no in-place mutation)."""
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.90)
    # max_single_pct=0.50, scale=0.90 → projected_new=0.45
    # existing=0.20, projected_total=0.65 > max=0.50
    # remaining=0.30 → proposed scale_adjustment=0.30/0.50=0.60
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.50, max_total_exposure_pct=0.50))
    r = await c.evaluate(v, {"total_value": 10000, "positions": {"A": 2000}})
    assert r.passed
    # Verdict MUST NOT be mutated — the check only proposes via CheckResult.
    assert v.position_scale == pytest.approx(0.90)
    assert r.scale_adjustment == pytest.approx(0.60)
    assert "Scale clamped" in r.reason


@pytest.mark.asyncio
async def test_total_exposure_projected_passes():
    """New trade within budget should pass."""
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.10)
    # max_single_pct=0.10, scale=0.10 → projected_new=0.01
    # existing=0.20, projected_total=0.21 < max=0.50
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.10, max_total_exposure_pct=0.50))
    r = await c.evaluate(v, {"total_value": 10000, "positions": {"A": 2000}})
    assert r.passed


@pytest.mark.asyncio
async def test_total_exposure_skips_unparseable_position_value(verdict):
    """Non-numeric position values must not crash the risk gate."""
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.10, max_total_exposure_pct=0.50))
    # Mixed shapes: dict with None fields, bare string, bare None, normal float.
    portfolio = {
        "total_value": 10000,
        "positions": {
            "BAD_DICT": {"amount": None, "avg_price": None},
            "BAD_STR": "pending",
            "BAD_NONE": None,
            "GOOD": 1000,
        },
    }
    r = await c.evaluate(verdict, portfolio)
    # Only "GOOD" (1000) and "BAD_DICT" (0*0=0) should be summed → 10% existing.
    # verdict scale=0.05, max_single_pct=0.10 → projected_new=0.5%; total < max.
    assert r.passed


@pytest.mark.asyncio
async def test_total_exposure_at_limit_rejects_with_no_budget():
    """TEST-I1: when remaining ≤ 0.01 the check rejects, doesn't propose tiny scale."""
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.50)
    # existing=4995/10000=49.95%, max=50% → remaining=0.05% (<= 0.01 threshold).
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.50, max_total_exposure_pct=0.50))
    r = await c.evaluate(v, {"total_value": 10000, "positions": {"A": 4995}})
    assert not r.passed
    assert r.scale_adjustment is None
    assert "No remaining exposure budget" in r.reason


@pytest.mark.asyncio
async def test_total_exposure_just_above_threshold_clamps():
    """TEST-I1: remaining slightly above 0.01 → scale clamping kicks in (boundary above)."""
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.50)
    # existing=4880/10000=48.8%, max=50% → remaining=1.2% (> 1% threshold).
    # proposed = 1.2% / 50% = 0.024 → still very small but legal.
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.50, max_total_exposure_pct=0.50))
    r = await c.evaluate(v, {"total_value": 10000, "positions": {"A": 4880}})
    assert r.passed
    assert r.scale_adjustment is not None
    assert 0 < r.scale_adjustment < 0.05


@pytest.mark.asyncio
async def test_total_exposure_dict_positions_summed_correctly():
    """TEST-M1: dict-format positions (production shape) accumulate exposure correctly."""
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.50)
    # Two dict positions: 0.1 BTC @ $30k = $3000 + 1 ETH @ $2000 = $2000 → $5000 / $10000 = 50%.
    # max=50% → remaining=0 → reject.
    c = MaxTotalExposure(PositionConfig(max_single_pct=0.50, max_total_exposure_pct=0.50))
    portfolio = {
        "total_value": 10000,
        "positions": {
            "BTC/USDT": {"amount": 0.1, "avg_price": 30000},
            "ETH/USDT": {"amount": 1, "avg_price": 2000},
        },
    }
    r = await c.evaluate(v, portfolio)
    assert not r.passed
    assert "50.00%" in r.reason


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
    bad_returns = [
        -0.05,
        -0.06,
        -0.07,
        -0.08,
        -0.02,
        0.01,
        -0.09,
        -0.04,
        -0.06,
        -0.03,
        -0.07,
        -0.05,
        -0.08,
        -0.02,
        -0.06,
        0.01,
        -0.04,
        -0.09,
        -0.03,
        -0.07,
        -0.05,
        -0.08,
        -0.06,
        -0.04,
        -0.09,
    ]
    r = await c.evaluate(verdict, {"returns_60d": bad_returns})
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
async def test_volatility_flash_crash_blocks_long(verdict):
    """Flash crash blocks long (catching falling knife)."""
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    r = await c.evaluate(verdict, {"recent_prices": [100, 99, 98, 97, 93]})
    assert not r.passed
    assert "blocking long" in r.reason


@pytest.mark.asyncio
async def test_volatility_flash_crash_allows_short():
    """Flash crash allows short (going with the trend)."""
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    short_verdict = TradeVerdict(action="short", confidence=0.7, position_scale=0.05)
    r = await c.evaluate(short_verdict, {"recent_prices": [100, 99, 98, 97, 93]})
    assert r.passed


@pytest.mark.asyncio
async def test_volatility_flash_crash_allows_close():
    """Flash crash allows close (risk-reducing action)."""
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    close_verdict = TradeVerdict(action="close", confidence=0.7, position_scale=0.0)
    r = await c.evaluate(close_verdict, {"recent_prices": [100, 99, 98, 97, 93]})
    assert r.passed


@pytest.mark.asyncio
async def test_volatility_spike_blocks_short():
    """Rapid spike blocks short (shorting into a spike)."""
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    short_verdict = TradeVerdict(action="short", confidence=0.7, position_scale=0.05)
    r = await c.evaluate(short_verdict, {"recent_prices": [93, 94, 95, 97, 100]})
    assert not r.passed
    assert "blocking short" in r.reason


@pytest.mark.asyncio
async def test_volatility_spike_allows_long():
    """Rapid spike allows long (going with the trend)."""
    c = VolatilityGate(VolatilityConfig(flash_crash_threshold=0.05))
    long_verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    r = await c.evaluate(long_verdict, {"recent_prices": [93, 94, 95, 97, 100]})
    assert r.passed


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


# ── Cooldown checks (in-memory fallback) ──


@pytest.mark.asyncio
async def test_cooldown_pass_no_active(verdict):
    """No cooldown set — should pass."""
    rsm = RedisStateManager(None)  # No Redis, uses memory fallback
    c = CooldownCheck(CooldownConfig(same_pair_minutes=5, post_loss_minutes=10), rsm)
    r = await c.evaluate(verdict, {"pair": "BTC/USDT"})
    assert r.passed


@pytest.mark.asyncio
async def test_cooldown_fail_pair_active(verdict):
    """Per-pair cooldown active — should block."""
    rsm = RedisStateManager(None)
    await rsm.set_cooldown("BTC/USDT", 5)
    c = CooldownCheck(CooldownConfig(same_pair_minutes=5, post_loss_minutes=10), rsm)
    r = await c.evaluate(verdict, {"pair": "BTC/USDT"})
    assert not r.passed
    assert "Cooldown active" in r.reason


@pytest.mark.asyncio
async def test_cooldown_fail_post_loss(verdict):
    """Post-loss cooldown active — should block."""
    rsm = RedisStateManager(None)
    await rsm.set_post_loss_cooldown(10)
    c = CooldownCheck(CooldownConfig(same_pair_minutes=5, post_loss_minutes=10), rsm)
    r = await c.evaluate(verdict, {"pair": "ETH/USDT"})
    assert not r.passed
    assert "Post-loss" in r.reason


# ── Rate limit checks (in-memory fallback) ──


@pytest.mark.asyncio
async def test_rate_limit_pass(verdict):
    rsm = RedisStateManager(None)
    c = RateLimitCheck(RateLimitConfig(max_trades_per_hour=10, max_trades_per_day=50), rsm)
    r = await c.evaluate(verdict, {})
    assert r.passed


@pytest.mark.asyncio
async def test_rate_limit_fail_hourly(verdict):
    rsm = RedisStateManager(None)
    for _ in range(10):
        await rsm.incr_trade_count()
    c = RateLimitCheck(RateLimitConfig(max_trades_per_hour=10, max_trades_per_day=50), rsm)
    r = await c.evaluate(verdict, {})
    assert not r.passed
    assert "Hourly" in r.reason


# ── MaxPositionSize with dict-format positions ──


@pytest.mark.asyncio
async def test_max_position_dict_format(verdict):
    """MaxPositionSize always passes — delta logic is in execution layer."""
    c = MaxPositionSize(PositionConfig(max_single_pct=0.10))
    portfolio = {
        "total_value": 10000,
        "positions": {"BTC/USDT": {"amount": 0.02, "avg_price": 50000}},
        "pair": "BTC/USDT",
    }
    r = await c.evaluate(verdict, portfolio)
    assert r.passed
