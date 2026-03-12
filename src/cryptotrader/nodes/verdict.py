"""Verdict and risk gate nodes."""

from __future__ import annotations

import logging
from typing import Any

from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)


async def _gather_risk_constraints(state: ArenaState) -> dict:
    """Collect current risk constraints for injection into verdict prompt."""
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    constraints: dict[str, Any] = {
        "max_position_pct": config.risk.position.max_single_pct,
        "max_drawdown_pct": config.risk.loss.max_drawdown_pct,
    }

    # Portfolio state
    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        pm_data = await pm.get_portfolio()
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
        total = pm_data.get("total_value", 0)
        if total > 0:
            positions = pm_data.get("positions", {})
            current_exposure = sum(abs(p.get("amount", 0) * p.get("avg_price", 0)) for p in positions.values())
            max_exp = config.risk.position.max_total_exposure_pct
            constraints["remaining_exposure_pct"] = max(0.0, max_exp - current_exposure / total)
            daily_loss_budget = config.risk.loss.max_daily_loss_pct
            if daily_pnl < 0:
                constraints["daily_loss_remaining_pct"] = max(0.0, daily_loss_budget - abs(daily_pnl / total))
            else:
                constraints["daily_loss_remaining_pct"] = daily_loss_budget
            constraints["drawdown_current"] = drawdown
    except Exception:
        logger.warning("Failed to gather portfolio risk constraints", exc_info=True)

    # Redis-based state (cooldowns, circuit breaker)
    redis_url = state["metadata"].get("redis_url")
    redis_state = RedisStateManager(redis_url)
    try:
        constraints["circuit_breaker_active"] = await redis_state.is_circuit_breaker_active()
        pair = state["metadata"].get("pair", "")
        cooldown_val = await redis_state.get(f"cooldown:{pair}")
        if cooldown_val:
            constraints["cooldown_pairs"] = [pair]
    except Exception:
        logger.warning("Failed to gather Redis risk constraints", exc_info=True)

    # Market conditions from snapshot
    summary = state["data"].get("snapshot_summary", {})
    if summary.get("funding_rate") is not None:
        constraints["funding_rate"] = summary["funding_rate"]
    if summary.get("volatility") is not None:
        constraints["volatility"] = summary["volatility"]

    return constraints


async def make_verdict(state: ArenaState) -> dict:
    """Generate trading verdict via AI or weighted-average fallback."""
    from cryptotrader.debate.verdict import make_verdict_llm, make_verdict_weighted

    raw_analyses = state["data"].get("analyses", {})
    # Filter out mock analyses (LLM failures) — don't let fake data pollute AI verdict
    analyses = {
        k: v
        for k, v in raw_analyses.items()
        if not (v.get("is_mock") if isinstance(v, dict) else getattr(v, "is_mock", False))
    }
    # If all agents returned mock data (LLM outage), skip verdict entirely
    if not analyses:
        logger.warning("All agents returned mock analyses — forcing hold verdict")
        from cryptotrader.debate.verdict import TradeVerdict

        verdict = TradeVerdict(action="hold", confidence=0.0, reasoning="All agents failed — no real data")
        return {
            "data": {
                "verdict": {
                    "action": verdict.action,
                    "confidence": verdict.confidence,
                    "position_scale": 0.0,
                    "divergence": 0.0,
                    "reasoning": verdict.reasoning,
                    "thesis": "",
                    "invalidation": "",
                }
            }
        }
    use_llm_verdict = state["metadata"].get("llm_verdict", True)

    if use_llm_verdict:
        from cryptotrader.config import load_config as _load_config

        _cfg = _load_config()
        _default_model = _cfg.models.verdict or _cfg.models.fallback
        model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", _default_model))
        constraints = await _gather_risk_constraints(state)
        calibration = state["data"].get("verdict_calibration", "")
        position_context = state["data"].get("position_context")
        trend_context = state["data"].get("trend_context")
        verdict = await make_verdict_llm(
            analyses,
            model=model,
            constraints=constraints,
            calibration=calibration,
            position_context=position_context,
            trend_context=trend_context,
        )
    else:
        scores = state.get("divergence_scores") or [0.0]
        threshold = state["metadata"].get("divergence_hold_threshold", 0.7)
        verdict = make_verdict_weighted(analyses, scores[-1], threshold)

    logger.info(
        "Verdict: action=%s confidence=%.2f scale=%.2f divergence=%.2f | %s",
        verdict.action,
        verdict.confidence,
        verdict.position_scale,
        verdict.divergence,
        verdict.reasoning[:120],
    )
    return {
        "data": {
            "verdict": {
                "action": verdict.action,
                "confidence": verdict.confidence,
                "position_scale": verdict.position_scale,
                "divergence": verdict.divergence,
                "reasoning": verdict.reasoning,
                "thesis": verdict.thesis,
                "invalidation": verdict.invalidation,
            }
        }
    }


# Module-level cache for RiskGate to preserve circuit breaker state
_risk_gate_cache: dict[str, Any] = {}

# Lazy notifier
_notifier_instance: Any = None


def _get_notifier(state: ArenaState) -> Any:
    global _notifier_instance
    if _notifier_instance is None:
        from cryptotrader.config import load_config
        from cryptotrader.notifications import Notifier

        cfg = load_config().notifications
        _notifier_instance = Notifier(cfg.webhook_url, cfg.enabled, cfg.events, cfg.webhook_timeout)
    return _notifier_instance


def _merge_returns(pm_returns: list[float], ohlcv_returns: list[float], min_count: int = 20) -> list[float]:
    """Prefer portfolio returns; supplement with OHLCV returns if insufficient."""
    if len(pm_returns) >= min_count:
        return pm_returns
    if not pm_returns:
        return ohlcv_returns
    # Pad with OHLCV returns at the front
    needed = min_count - len(pm_returns)
    return ohlcv_returns[:needed] + pm_returns


async def _fetch_exchange_total(state: ArenaState) -> float:
    """Query exchange balance and return total USDT value (best-effort)."""
    from cryptotrader.config import load_config as _lc

    exchange_id = state["metadata"].get("exchange_id") or _lc().exchange_id
    try:
        exchange = _get_or_create_live_exchange(exchange_id)
        if exchange is None:
            return 0.0
        bal = await exchange.get_balance()
        return bal.get("USDT", 0.0)
    except Exception:
        logger.warning("Exchange balance query failed", exc_info=True)
        return 0.0


def _get_or_create_live_exchange(exchange_id: str):
    """Get cached live exchange, or create one from config if needed."""
    from cryptotrader.nodes.execution import _live_exchanges

    if exchange_id in _live_exchanges:
        return _live_exchanges[exchange_id]

    from cryptotrader.config import load_config
    from cryptotrader.execution.exchange import LiveExchange

    config = load_config()
    creds = config.exchanges.get(exchange_id)
    if creds is None or not creds.api_key or not creds.secret:
        return None

    live_exchange = LiveExchange(
        exchange_id,
        creds.api_key,
        creds.secret,
        sandbox=creds.sandbox,
        passphrase=creds.passphrase,
    )
    _live_exchanges[exchange_id] = live_exchange
    return live_exchange


def _extract_ohlcv_returns(state: ArenaState) -> tuple[list[float], list[float]]:
    """Extract recent prices and daily returns from snapshot OHLCV data."""
    snapshot = state["data"].get("snapshot")
    if not snapshot or not hasattr(snapshot, "market") or snapshot.market.ohlcv is None:
        return [], []
    closes = snapshot.market.ohlcv["close"].dropna().tolist()
    recent_prices = closes[-60:]
    if len(closes) < 2:
        return recent_prices, []
    bar_returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes)) if closes[i - 1] > 0]
    timeframe = state["metadata"].get("timeframe", "1h")
    _tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    bars_per_day = int(86_400_000 / _tf_ms.get(timeframe, 3_600_000))
    if bars_per_day > 1 and len(bar_returns) >= bars_per_day:
        returns_daily = []
        for j in range(0, len(bar_returns) - bars_per_day + 1, bars_per_day):
            chunk = bar_returns[j : j + bars_per_day]
            daily = 1.0
            for r in chunk:
                daily *= 1 + r
            returns_daily.append(daily - 1)
    else:
        returns_daily = bar_returns
    return recent_prices, returns_daily


async def _measure_api_latency(state: ArenaState) -> int:
    """Measure real exchange API latency for live mode; return ms."""
    engine = state["metadata"].get("engine", "paper")
    if engine != "live":
        return 100
    import time

    from cryptotrader.config import load_config as _lc2

    exchange_id = state["metadata"].get("exchange_id") or _lc2().exchange_id
    try:
        exchange = _get_or_create_live_exchange(exchange_id)
        if exchange is None:
            return 100
        t0 = time.monotonic()
        await exchange.get_balance()
        return int((time.monotonic() - t0) * 1000)
    except Exception:
        logger.warning("API latency check failed, using default", exc_info=True)
        return 100


async def risk_check(state: ArenaState) -> dict:
    """Run all risk gate checks on the verdict."""
    from cryptotrader.config import load_config
    from cryptotrader.models import TradeVerdict
    from cryptotrader.portfolio.manager import PortfolioManager
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    redis_url = state["metadata"].get("redis_url")
    cache_key = redis_url or "_default"
    if cache_key not in _risk_gate_cache:
        redis_state = RedisStateManager(redis_url)
        _risk_gate_cache[cache_key] = RiskGate(config.risk, redis_state)
    gate = _risk_gate_cache[cache_key]

    vd = state["data"]["verdict"]
    verdict = TradeVerdict(**vd)

    recent_prices, returns_daily = _extract_ohlcv_returns(state)

    # Load real portfolio state
    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        pm_data = await pm.get_portfolio()
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
        pm_returns = await pm.get_returns()
    except Exception:
        logger.debug("Portfolio data fetch failed, using defaults", exc_info=True)
        pm_data = {"total_value": 0, "positions": {}}
        daily_pnl = 0.0
        drawdown = 0.0
        pm_returns = []

    has_real_portfolio = pm_data.get("total_value", 0) > 0
    engine = state["metadata"].get("engine", "paper")

    # Live mode: if local portfolio is empty, query exchange for real balance
    if engine == "live" and not has_real_portfolio:
        exchange_total = await _fetch_exchange_total(state)
        if exchange_total > 0:
            pm_data["total_value"] = exchange_total
            has_real_portfolio = True
        else:
            return {
                "data": {
                    "risk_gate": {
                        "passed": False,
                        "rejected_by": "portfolio_unknown",
                        "reason": "Live mode: no portfolio data — local DB empty and exchange balance query failed.",
                    }
                }
            }

    api_latency_ms = await _measure_api_latency(state)

    portfolio = state["data"].get(
        "portfolio",
        {
            "total_value": pm_data["total_value"] if has_real_portfolio else config.backtest.initial_capital,
            "positions": pm_data.get("positions", {}),
            "daily_pnl": daily_pnl,
            "drawdown": drawdown,
            "returns_60d": _merge_returns(pm_returns, returns_daily),
            "recent_prices": recent_prices,
            "funding_rate": state["data"].get("snapshot_summary", {}).get("funding_rate", 0),
            "api_latency_ms": api_latency_ms,
            "pair": state["metadata"]["pair"],
        },
    )
    result = await gate.check(verdict, portfolio)

    if result.passed:
        logger.info("Risk gate PASSED for %s (action=%s)", state["metadata"]["pair"], verdict.action)
    else:
        logger.warning(
            "Risk gate REJECTED for %s: %s — %s",
            state["metadata"]["pair"],
            result.rejected_by,
            result.reason,
        )

    # Fire circuit_breaker notification if triggered
    if not result.passed and result.rejected_by == "daily_loss_limit" and "Circuit breaker" in (result.reason or ""):
        try:
            notifier = _get_notifier(state)
            await notifier.notify("circuit_breaker", {"pair": state["metadata"]["pair"], "reason": result.reason})
        except Exception:
            logger.debug("Circuit-breaker notification failed", exc_info=True)

    return {
        "data": {
            "risk_gate": {
                "passed": result.passed,
                "rejected_by": result.rejected_by,
                "reason": result.reason,
            }
        }
    }


def risk_router(state: ArenaState) -> str:
    """Route to 'approved' or 'rejected' based on risk gate result."""
    rg = state["data"].get("risk_gate", {})
    return "approved" if rg.get("passed", False) else "rejected"
