"""Verdict and risk gate nodes."""

from __future__ import annotations

import logging
from typing import Any

import structlog

from cryptotrader.metrics import get_metrics_collector
from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)
_structlog = structlog.get_logger(__name__)


async def _gather_risk_constraints(state: ArenaState) -> dict:
    """Collect current risk constraints for injection into verdict prompt."""
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager, read_portfolio_from_exchange
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    constraints: dict[str, Any] = {
        "max_position_pct": config.risk.position.max_single_pct,
        "max_drawdown_pct": config.risk.loss.max_drawdown_pct,
    }

    # Portfolio state — read from exchange (source of truth)
    try:
        exchange_portfolio = await read_portfolio_from_exchange(state)
        total = exchange_portfolio["total_value"] if exchange_portfolio else 0

        # Historical metrics from DB snapshots
        db_url = state["metadata"].get("database_url")
        pm = PortfolioManager(db_url)
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()

        if total > 0:
            positions = exchange_portfolio.get("positions", {}) if exchange_portfolio else {}
            price = state["data"].get("snapshot_summary", {}).get("price", 0)
            current_exposure = sum(abs(p.get("amount", 0)) * price for p in positions.values())
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


async def _should_downgrade_to_weighted(state: ArenaState) -> bool:
    """Check if verdict can safely use weighted average instead of AI.

    Safe when: position is flat AND no circuit breaker active.
    """
    # 1. Position flat?
    pos = state["data"].get("position_context") or {}
    if pos.get("side", "flat") != "flat":
        return False
    # 2. No circuit breaker?
    redis_url = state["metadata"].get("redis_url")
    try:
        from cryptotrader.risk.state import RedisStateManager

        rsm = RedisStateManager(redis_url)
        if await rsm.is_circuit_breaker_active():
            return False
    except Exception:
        logger.debug("Redis unavailable for downgrade check, keeping AI verdict", exc_info=True)
        return False
    return True


@node_logger()
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
        get_metrics_collector().inc_verdict(action=verdict.action)
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
                    "verdict_source": "hold_all_mock",
                }
            }
        }
    use_llm_verdict = state["metadata"].get("llm_verdict", True)
    debate_skipped = state["data"].get("debate_skipped", False)
    verdict_source: str

    if use_llm_verdict and debate_skipped and await _should_downgrade_to_weighted(state):
        logger.info("Verdict downgraded to weighted (debate skipped, flat, no circuit breaker)")
        scores = state.get("divergence_scores") or [0.0]
        threshold = state["metadata"].get("divergence_hold_threshold", 0.7)
        verdict = make_verdict_weighted(analyses, scores[-1] if scores else 0.0, threshold)
        verdict_source = "weighted"
    elif use_llm_verdict:
        from cryptotrader.config import load_config as _load_config

        _cfg = _load_config()
        _default_model = _cfg.models.verdict or _cfg.models.fallback
        model = state["metadata"].get("verdict_model", state["metadata"].get("debate_model", _default_model))
        # In backtest mode, use constraints constructed by BacktestEngine from its own
        # state variables (equity, position, peak) instead of querying PortfolioManager/Redis
        if state["metadata"].get("backtest_mode"):
            constraints = state["data"].get("backtest_constraints", {})
        else:
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
        verdict_source = "ai"
    else:
        scores = state.get("divergence_scores") or [0.0]
        threshold = state["metadata"].get("divergence_hold_threshold", 0.7)
        verdict = make_verdict_weighted(analyses, scores[-1], threshold)
        verdict_source = "weighted"

    logger.info(
        "Verdict: action=%s confidence=%.2f scale=%.2f divergence=%.2f | %s",
        verdict.action,
        verdict.confidence,
        verdict.position_scale,
        verdict.divergence,
        verdict.reasoning[:120],
    )
    # Metrics instrumentation: ct_verdict_total[action] (req 9.5)
    get_metrics_collector().inc_verdict(action=verdict.action)
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
                "verdict_source": verdict_source,
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

    from cryptotrader.nodes.execution import _get_exchange

    pair = state["metadata"].get("pair", "BTC/USDT")
    try:
        exchange, _ = await _get_exchange(state, pair)
        t0 = time.monotonic()
        await exchange.get_balance()
        return int((time.monotonic() - t0) * 1000)
    except Exception:
        logger.warning("API latency check failed, using default", exc_info=True)
        return 100


@node_logger()
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

    # In backtest mode, portfolio is pre-built by the engine — skip exchange queries
    pre_built_portfolio = state["data"].get("portfolio")
    if pre_built_portfolio:
        portfolio = pre_built_portfolio
    else:
        recent_prices, returns_daily = _extract_ohlcv_returns(state)

        # Read portfolio from exchange (source of truth)
        from cryptotrader.portfolio.manager import read_portfolio_from_exchange

        exchange_portfolio = await read_portfolio_from_exchange(state)

        # Historical metrics from DB snapshots (PnL, drawdown, returns)
        db_url = state["metadata"].get("database_url")
        pm = PortfolioManager(db_url)
        try:
            daily_pnl = await pm.get_daily_pnl()
            drawdown = await pm.get_drawdown()
            pm_returns = await pm.get_returns()
        except Exception:
            logger.debug("Portfolio snapshot data fetch failed", exc_info=True)
            daily_pnl = 0.0
            drawdown = 0.0
            pm_returns = []

        if exchange_portfolio and exchange_portfolio.get("total_value", 0) > 0:
            total_value = exchange_portfolio["total_value"]
            positions = exchange_portfolio.get("positions", {})
        elif state["metadata"].get("engine") == "live":
            # Live mode: exchange returned 0 — reject to prevent trading with unknown portfolio
            logger.warning("Exchange returned 0 total_value in live mode — rejecting")
            return {
                "data": {
                    "risk_gate": {
                        "passed": False,
                        "rejected_by": "portfolio_unknown",
                        "reason": "Exchange returned 0 balance — cannot trade safely",
                    }
                }
            }
        else:
            # Paper mode fallback: use initial capital from config
            total_value = config.backtest.initial_capital
            positions = {}

        api_latency_ms = await _measure_api_latency(state)
        portfolio = {
            "total_value": total_value,
            "positions": positions,
            "cash": exchange_portfolio.get("cash", 0) if exchange_portfolio else 0,
            "daily_pnl": daily_pnl,
            "drawdown": drawdown,
            "returns_60d": _merge_returns(pm_returns, returns_daily),
            "recent_prices": recent_prices,
            "funding_rate": state["data"].get("snapshot_summary", {}).get("funding_rate", 0),
            "api_latency_ms": api_latency_ms,
            "pair": state["metadata"]["pair"],
        }

    result = await gate.check(verdict, portfolio)

    if result.passed:
        logger.info("Risk gate PASSED for %s (action=%s)", state["metadata"]["pair"], verdict.action)
    else:
        # Structured rejection log with required fields: check_name, reason summary (req 9.4).
        # The reason string from each RiskCheck already embeds current_value/threshold details
        # (e.g. "Funding rate 0.001 exceeds threshold 0.0002"), so structlog fields carry
        # all the context needed for alerting and log-based metrics.
        _structlog.warning(
            "risk_gate_rejected",
            pair=state["metadata"]["pair"],
            check_name=result.rejected_by or "unknown",
            reason=result.reason or "",
        )
        logger.warning(
            "Risk gate REJECTED for %s: check_name=%s reason=%s",
            state["metadata"]["pair"],
            result.rejected_by,
            result.reason,
        )
        # Metrics instrumentation: ct_risk_rejected_total[check_name] (req 9.5)
        get_metrics_collector().inc_risk_rejected(check_name=result.rejected_by or "unknown")

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
