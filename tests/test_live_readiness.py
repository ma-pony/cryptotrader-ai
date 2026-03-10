"""Tests for live trading readiness fixes (P0/P1/P2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.config import ExchangeCredentials, ExchangesConfig

# ── P0-1: Live credentials ──


def test_exchange_credentials_config():
    """ExchangesConfig stores and retrieves credentials by exchange id."""
    creds = ExchangeCredentials(api_key="key", secret="sec", passphrase="pass", sandbox=False)
    cfg = ExchangesConfig(_exchanges={"okx": creds})
    assert cfg.get("okx") is creds
    assert cfg.get("binance") is None


def test_live_credentials_empty_raises():
    """_get_exchange raises RuntimeError for live mode with empty credentials."""
    from cryptotrader.nodes.execution import _get_exchange, _live_exchanges

    # Clear cache to force credential lookup
    _live_exchanges.pop("binance", None)

    state = {
        "messages": [],
        "data": {},
        "metadata": {"engine": "live", "exchange_id": "binance"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    # Mock config with empty credentials
    mock_config = MagicMock()
    mock_config.exchanges.get.return_value = None

    with (
        patch("cryptotrader.config.load_config", return_value=mock_config),
        pytest.raises(RuntimeError, match="No credentials configured"),
    ):
        _get_exchange(state, "BTC/USDT")


def test_live_exchange_sandbox_keyword_only():
    """LiveExchange requires sandbox as keyword-only argument."""
    from cryptotrader.execution.exchange import LiveExchange

    # Should fail without sandbox keyword
    with pytest.raises(TypeError):
        LiveExchange("binance", "key", "secret", True)  # positional — should fail


# ── P0-2: Risk gate name match ──


def test_gate_redis_dependent_names():
    """The redis_dependent set must match actual check .name attributes."""
    from cryptotrader.risk.checks.cooldown import CooldownCheck
    from cryptotrader.risk.checks.loss import DailyLossLimit
    from cryptotrader.risk.checks.rate_limit import RateLimitCheck

    # These are the names that must match
    assert CooldownCheck.name == "cooldown"
    assert DailyLossLimit.name == "daily_loss_limit"
    assert RateLimitCheck.name == "rate_limit"


# ── P0-3: Post-loss cooldown activation ──


@pytest.mark.asyncio
async def test_post_loss_cooldown_called():
    """DailyLossLimit calls set_post_loss_cooldown when circuit breaker triggers."""
    from cryptotrader.config import LossConfig
    from cryptotrader.models import TradeVerdict
    from cryptotrader.risk.checks.loss import DailyLossLimit

    mock_redis = MagicMock()
    mock_redis.available = True
    mock_redis.is_circuit_breaker_active = AsyncMock(return_value=False)
    mock_redis.set_circuit_breaker = AsyncMock()
    mock_redis.set_post_loss_cooldown = AsyncMock()

    config = LossConfig(max_daily_loss_pct=0.03)
    check = DailyLossLimit(config, mock_redis, post_loss_minutes=60)

    verdict = TradeVerdict(action="long", confidence=0.8)
    portfolio = {"total_value": 10000, "daily_pnl": -500}  # 5% loss > 3% max

    result = await check.evaluate(verdict, portfolio)
    assert not result.passed
    mock_redis.set_circuit_breaker.assert_called_once()
    mock_redis.set_post_loss_cooldown.assert_called_once_with(60)


# ── P0-4: Circuit breaker TTL ──


@pytest.mark.asyncio
async def test_circuit_breaker_ttl():
    """set_circuit_breaker passes TTL to underlying set()."""
    from cryptotrader.risk.state import RedisStateManager

    rsm = RedisStateManager(None)  # memory fallback
    await rsm.set_circuit_breaker(ttl_seconds=3600)

    # Should be active
    assert await rsm.is_circuit_breaker_active()

    # Check that the TTL was passed (memory store records expire_ts)
    entry = rsm._mem._data.get("circuit_breaker:active")
    assert entry is not None
    assert entry[1] is not None  # expire_ts is set


# ── P0-5: Cold start protection ──


@pytest.mark.asyncio
async def test_cold_start_live_reject():
    """Live mode rejects when both local portfolio and exchange balance are 0."""
    from cryptotrader.nodes.verdict import _risk_gate_cache, risk_check

    _risk_gate_cache.clear()

    state = {
        "messages": [],
        "data": {
            "verdict": {
                "action": "long",
                "confidence": 0.7,
                "position_scale": 0.5,
                "divergence": 0.1,
                "reasoning": "test",
                "thesis": "test",
                "invalidation": "test",
            },
        },
        "metadata": {"pair": "BTC/USDT", "engine": "live"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_pm = MagicMock()
    mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 0, "positions": {}})
    mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
    mock_pm.get_drawdown = AsyncMock(return_value=0.0)
    mock_pm.get_returns = AsyncMock(return_value=[])

    with (
        patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        patch("cryptotrader.nodes.verdict._fetch_exchange_total", new_callable=AsyncMock, return_value=0.0),
    ):
        result = await risk_check(state)

    rg = result["data"]["risk_gate"]
    assert not rg["passed"]
    assert rg["rejected_by"] == "portfolio_unknown"


@pytest.mark.asyncio
async def test_cold_start_live_uses_exchange_balance():
    """Live mode uses exchange balance when local portfolio is empty."""
    from cryptotrader.nodes.verdict import _risk_gate_cache, risk_check

    _risk_gate_cache.clear()

    state = {
        "messages": [],
        "data": {
            "verdict": {
                "action": "long",
                "confidence": 0.7,
                "position_scale": 0.5,
                "divergence": 0.1,
                "reasoning": "test",
                "thesis": "test",
                "invalidation": "test",
            },
        },
        "metadata": {"pair": "BTC/USDT", "engine": "live"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    mock_pm = MagicMock()
    mock_pm.get_portfolio = AsyncMock(return_value={"total_value": 0, "positions": {}})
    mock_pm.get_daily_pnl = AsyncMock(return_value=0.0)
    mock_pm.get_drawdown = AsyncMock(return_value=0.0)
    mock_pm.get_returns = AsyncMock(return_value=[])

    with (
        patch("cryptotrader.portfolio.manager.PortfolioManager", return_value=mock_pm),
        # Exchange has 8000 USDT
        patch("cryptotrader.nodes.verdict._fetch_exchange_total", new_callable=AsyncMock, return_value=8000.0),
        patch("cryptotrader.nodes.verdict._measure_api_latency", new_callable=AsyncMock, return_value=100),
    ):
        result = await risk_check(state)

    rg = result["data"]["risk_gate"]
    # Should NOT be rejected — exchange has balance
    assert rg["rejected_by"] != "portfolio_unknown"


# ── P0-6: Retry fatal errors ──


@pytest.mark.asyncio
async def test_retry_fatal_error():
    """_retry raises immediately on fatal ccxt errors without retrying."""
    import sys
    from unittest.mock import MagicMock as SyncMock

    # Create a mock ccxt module with exception classes
    mock_ccxt = SyncMock()
    mock_ccxt.AuthenticationError = type("AuthenticationError", (Exception,), {})
    mock_ccxt.PermissionDenied = type("PermissionDenied", (Exception,), {})
    mock_ccxt.BadSymbol = type("BadSymbol", (Exception,), {})
    mock_ccxt.InsufficientFunds = type("InsufficientFunds", (Exception,), {})

    from cryptotrader.execution.exchange import LiveExchange

    # We can't easily construct a LiveExchange without ccxt, so test the concept
    # by directly testing that the fatal error types are not retried
    call_count = 0

    async def failing_fn():
        nonlocal call_count
        call_count += 1
        raise mock_ccxt.AuthenticationError("Bad API key")

    # Patch ccxt at module level for the _retry method
    old = sys.modules.get("ccxt")
    sys.modules["ccxt"] = mock_ccxt
    try:
        # Create a minimal instance to test _retry
        ex = object.__new__(LiveExchange)
        with pytest.raises(Exception, match="Bad API key"):
            await ex._retry(failing_fn, attempts=3)
        assert call_count == 1  # Should not retry
    finally:
        if old is not None:
            sys.modules["ccxt"] = old
        else:
            del sys.modules["ccxt"]


# ── P0-7: Redis unavailable → reject ──


@pytest.mark.asyncio
async def test_redis_unavailable_reject():
    """RiskGate rejects trade when Redis was configured but is unreachable."""
    from cryptotrader.config import RiskConfig
    from cryptotrader.models import TradeVerdict
    from cryptotrader.risk.gate import RiskGate

    mock_redis = MagicMock()
    mock_redis._redis = MagicMock()  # was configured
    mock_redis.ping = AsyncMock(return_value=False)  # but unreachable

    gate = RiskGate(RiskConfig(), mock_redis)
    verdict = TradeVerdict(action="long", confidence=0.8)
    portfolio = {"total_value": 10000, "positions": {}, "pair": "BTC/USDT"}

    result = await gate.check(verdict, portfolio)
    assert not result.passed
    assert result.rejected_by == "redis_unavailable"


# ── P1-1: Journal exception safety ──


@pytest.mark.asyncio
async def test_journal_trade_exception_safe():
    """journal_trade catches exceptions and returns None hash instead of crashing."""
    from cryptotrader.nodes.journal import journal_trade

    state = {
        "messages": [],
        "data": {
            "analyses": {"tech_agent": {"direction": "bullish", "confidence": 0.8}},
            "verdict": {"action": "long", "confidence": 0.8, "position_scale": 1.0},
            "risk_gate": {"passed": True},
            "order": {"pair": "BTC/USDT", "side": "buy", "amount": 0.02, "price": 50000},
        },
        "metadata": {"pair": "BTC/USDT", "engine": "live"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    with patch("cryptotrader.journal.store.JournalStore.commit", side_effect=Exception("DB connection lost")):
        result = await journal_trade(state)

    # Should not crash — returns None hash
    assert result["data"]["journal_hash"] is None


# ── P1-3: Drawdown triggers circuit breaker ──


@pytest.mark.asyncio
async def test_drawdown_circuit_breaker():
    """DrawdownLimit triggers circuit breaker via Redis when exceeded."""
    from cryptotrader.config import LossConfig
    from cryptotrader.models import TradeVerdict
    from cryptotrader.risk.checks.loss import DrawdownLimit

    mock_redis = MagicMock()
    mock_redis.available = True
    mock_redis.set_circuit_breaker = AsyncMock()

    config = LossConfig(max_drawdown_pct=0.10)
    check = DrawdownLimit(config, mock_redis)

    verdict = TradeVerdict(action="long", confidence=0.8)
    portfolio = {"drawdown": 0.15}  # 15% > 10% max

    result = await check.evaluate(verdict, portfolio)
    assert not result.passed
    mock_redis.set_circuit_breaker.assert_called_once()


# ── Config parsing ──


def test_config_parses_exchanges():
    """_build_config correctly parses [exchanges] section."""
    from cryptotrader.config import _build_config

    binance_creds = {
        "api_key": "test_key",  # pragma: allowlist secret
        "secret": "test_val",  # pragma: allowlist secret
        "sandbox": True,
    }
    okx_creds = {
        "api_key": "okx_key",  # pragma: allowlist secret
        "secret": "okx_val",  # pragma: allowlist secret
        "passphrase": "pp",
        "sandbox": False,
    }
    toml_data = {"exchanges": {"binance": binance_creds, "okx": okx_creds}}
    config = _build_config(toml_data)

    binance = config.exchanges.get("binance")
    assert binance is not None
    assert binance.api_key == "test_key"  # pragma: allowlist secret
    assert binance.sandbox is True

    okx = config.exchanges.get("okx")
    assert okx is not None
    assert okx.passphrase == "pp"
    assert okx.sandbox is False


# ── CVaR return merging ──


def test_merge_returns_prefers_pm():
    """_merge_returns uses portfolio returns when sufficient."""
    from cryptotrader.nodes.verdict import _merge_returns

    pm = [0.01] * 25
    ohlcv = [0.02] * 30
    result = _merge_returns(pm, ohlcv, min_count=20)
    assert result == pm  # sufficient — use pm only


def test_merge_returns_supplements():
    """_merge_returns supplements insufficient pm_returns with OHLCV."""
    from cryptotrader.nodes.verdict import _merge_returns

    pm = [0.01] * 10
    ohlcv = [0.02] * 30
    result = _merge_returns(pm, ohlcv, min_count=20)
    assert len(result) == 20
    assert result[:10] == [0.02] * 10  # padded from OHLCV
    assert result[10:] == [0.01] * 10  # original pm


def test_merge_returns_empty_pm():
    """_merge_returns falls back to OHLCV when pm is empty."""
    from cryptotrader.nodes.verdict import _merge_returns

    result = _merge_returns([], [0.02] * 15)
    assert result == [0.02] * 15
