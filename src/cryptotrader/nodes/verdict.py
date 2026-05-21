"""Verdict and risk gate nodes."""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

from cryptotrader.metrics import get_metrics_collector
from cryptotrader.state import ArenaState, get_pair
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)
_structlog = structlog.get_logger(__name__)

# Detects the canonical FR-026 attribution string (`applied: <agent>::<pattern>` or
# `applied: <pattern>`). The two-form match is intentional: agents may cite either
# their own bare skill or another agent's namespaced skill.
_APPLIED_RE = re.compile(r"applied:\s*([A-Za-z0-9_:]+)")


def _inject_weighted_sl_tp(vd_dict: dict, entry: float | None, atr: float | None) -> None:
    """Synthesize deterministic SL/TP for a directional weighted verdict.

    The weighted-fallback path (debate skipped on strong consensus / Redis
    unreachable / circuit-breaker) has no LLM to produce ``stop_loss`` /
    ``take_profit``, so Phase 1's ``missing_sl_tp`` hard-reject would force
    every high-consensus pair to hold — defeating spec 012's "skip debate
    to save LLM tokens on strong-consensus pairs" intent.

    Mirror the floors that ``_post_process_verdict`` enforces so the
    injected values always pass the gate:
      stop_distance  = max(1.5 × ATR, 1% × entry) × 1.001   ← 0.1% pad
      target_distance = 2.0 × stop_distance                  ← R:R = 2.0 ≥ 1.5 floor

    The 0.1% pad on stop_distance prevents a float round-trip miss: gate
    re-derives stop_distance as ``abs(sl - entry)``, and the subtraction
    can land 1 ULP below the originally computed floor (observed on
    ETH @ 2256.30 → 22.563 vs 22.562999... = rejected as stop_too_tight).
    Padding 0.1% guarantees the recovered distance stays ≥ floor without
    materially widening the stop.

    No-op when entry is missing/non-positive or LLM already supplied SL/TP
    (caller may have populated them via a different path).
    """
    action = vd_dict.get("action", "hold")
    if action not in ("long", "short"):
        return
    if vd_dict.get("stop_loss") is not None or vd_dict.get("take_profit") is not None:
        return
    if not entry or entry <= 0:
        return

    atr_floor = 1.5 * atr if (atr and atr > 0) else 0.0
    pct_floor = entry * 0.01
    stop_distance = max(atr_floor, pct_floor) * 1.001
    target_distance = 2.0 * stop_distance

    if action == "long":
        sl = entry - stop_distance
        tp = entry + target_distance
    else:
        sl = entry + stop_distance
        tp = entry - target_distance

    vd_dict["stop_loss"] = round(sl, 8)
    vd_dict["take_profit"] = round(tp, 8)
    vd_dict.setdefault("guardrails", []).append("weighted_sl_tp_injected")
    logger.info(
        "Weighted SL/TP injected: action=%s entry=%.6f sl=%.6f tp=%.6f stop_dist=%.6f (R:R=2.0)",
        action,
        entry,
        sl,
        tp,
        stop_distance,
    )


def _force_hold(vd_dict: dict, reason: str) -> None:
    """Force the verdict to hold, log the rejection, record the reason as a guardrail.

    Used when SL/TP is missing / malformed / out of bounds — we cannot place
    a protected algo order, so we must refuse the trade entirely rather than
    silently downgrade confidence and proceed with a naked position.
    """
    logger.warning("Verdict REJECTED → hold; reason=%s", reason)
    vd_dict["action"] = "hold"
    vd_dict["confidence"] = 0.0
    vd_dict["position_scale"] = 0.0
    vd_dict["stop_loss"] = None
    vd_dict["take_profit"] = None
    vd_dict.setdefault("guardrails", []).append(f"rejected:{reason}")
    # spec 022 FR-022-15: store trace context for async journal hook (added by make_verdict)
    vd_dict["_phase1_rejected_reason"] = reason


def _post_process_verdict(
    verdict,
    raw_analyses: dict,
    vd_dict: dict,
    *,
    entry_price: float | None = None,
    atr: float | None = None,
) -> dict:
    """Apply deterministic server-side guardrails on top of the LLM verdict.

    Three guardrails (in order — each only LOWERS confidence, never raises):
      1. Confidence-based sizing cap. The LLM's ``position_scale`` is capped by
         a Kelly-ish linear ramp from confidence (cf 0.50 → 0%, cf 0.70 → 40%,
         cf 1.00 → 100% of max_single). Prevents medium-conviction trades from
         going max-size.
      2. Missing FR-026 ``applied:`` citation on a directional verdict halves
         confidence. Skill PnL attribution requires a citation; without one we
         cannot learn from this trade.
      3. Any agent that returned ``is_mock`` or ``confidence==0`` (data outage)
         caps verdict.confidence at ``raw - 0.20``. The LLM does not know that
         one of the inputs was synthetic, so the cap is enforced here.

    Returns the modified verdict dict (in-place via ``vd_dict``).
    """
    action = vd_dict.get("action", "hold")
    cf = float(vd_dict.get("confidence", 0.0) or 0.0)
    scale = float(vd_dict.get("position_scale", cf) or 0.0)
    reasoning = vd_dict.get("reasoning", "") or ""

    # Guardrail 1: confidence-based sizing cap.
    # Linear ramp: cf=0.50 → 0, cf=0.70 → 0.4, cf=1.00 → 1.0
    # Floors at 0 (sub-coin-flip confidence carries no size). Hold/close
    # actions exit fully and use scale 0 / 1 already, so skip them.
    if action in ("long", "short"):
        confidence_cap = max(0.0, min(1.0, (cf - 0.5) * 2))
        if scale > confidence_cap:
            logger.info(
                "Verdict scale capped by confidence ramp: %.2f -> %.2f (cf=%.2f, action=%s)",
                scale,
                confidence_cap,
                cf,
                action,
            )
            scale = confidence_cap
            vd_dict["position_scale"] = scale
            vd_dict.setdefault("guardrails", []).append("confidence_scale_cap")

    # Guardrail 2: directional verdict missing `applied:` halves confidence.
    if action in ("long", "short", "close") and not _APPLIED_RE.search(reasoning):
        new_cf = round(cf * 0.5, 4)
        logger.info(
            "Verdict missing FR-026 applied: citation — confidence %.2f -> %.2f",
            cf,
            new_cf,
        )
        cf = new_cf
        vd_dict["confidence"] = cf
        vd_dict.setdefault("guardrails", []).append("missing_applied")

    # Guardrail 3: any agent silent/mock → cap confidence at raw - 0.20.
    silent_agents: list[str] = []
    for aid, a in (raw_analyses or {}).items():
        if not isinstance(a, dict):
            a = {} if a is None else getattr(a, "__dict__", {})
        is_mock = a.get("is_mock", False)
        a_conf = float(a.get("confidence", 0.0) or 0.0)
        if is_mock or a_conf == 0.0:
            silent_agents.append(aid)
    if silent_agents and action in ("long", "short", "close"):
        capped = max(0.0, min(cf, max(0.0, cf - 0.20)))
        if capped < cf:
            logger.info(
                "Verdict confidence capped due to silent/mock agents %s: %.2f -> %.2f",
                silent_agents,
                cf,
                capped,
            )
            cf = capped
            vd_dict["confidence"] = cf
            vd_dict.setdefault("guardrails", []).append(f"silent_agents:{','.join(silent_agents)}")
        # Re-apply confidence ramp since cf dropped.
        if action in ("long", "short"):
            ramp = max(0.0, min(1.0, (cf - 0.5) * 2))
            if scale > ramp:
                vd_dict["position_scale"] = ramp

    # SL/TP HARD-REJECT GATE (replaces former Guardrails 4/5 confidence penalties).
    # Any of the following → force action=hold (cannot place protected algo order):
    #   • stop_loss / take_profit missing or non-numeric
    #   • stop too tight   (< max(1.5×ATR, 1.0% of entry))
    #   • R:R < 1.5
    #   • direction inverted (long but stop ≥ entry, or short but stop ≤ entry)
    if action in ("long", "short") and entry_price and entry_price > 0:
        sl = vd_dict.get("stop_loss")
        tp = vd_dict.get("take_profit")

        # Coerce to float; non-numeric → None
        try:
            sl_f = float(sl) if sl is not None else None
        except (TypeError, ValueError):
            sl_f = None
        try:
            tp_f = float(tp) if tp is not None else None
        except (TypeError, ValueError):
            tp_f = None

        if sl_f is None or tp_f is None:
            _force_hold(vd_dict, "missing_sl_tp")
            return vd_dict

        # Direction sanity
        if action == "long" and not (sl_f < entry_price < tp_f):
            _force_hold(vd_dict, "direction_inverted")
            return vd_dict
        if action == "short" and not (tp_f < entry_price < sl_f):
            _force_hold(vd_dict, "direction_inverted")
            return vd_dict

        # Stop-distance floor
        stop_distance = abs(sl_f - entry_price)
        atr_floor = 1.5 * atr if (atr and atr > 0) else 0.0
        pct_floor = entry_price * 0.01
        min_distance = max(atr_floor, pct_floor)
        if stop_distance < min_distance:
            logger.info(
                "stop_too_tight: distance=%.6f < min=%.6f (1.5×ATR=%.6f, 1%%=%.6f)",
                stop_distance,
                min_distance,
                atr_floor,
                pct_floor,
            )
            _force_hold(vd_dict, "stop_too_tight")
            return vd_dict

        # R:R minimum
        target_distance = abs(tp_f - entry_price)
        rr = target_distance / stop_distance
        if rr < 1.5:
            logger.info(
                "low_rr: %.2f < 1.5 (entry=%.6f stop=%.6f target=%.6f)",
                rr,
                entry_price,
                sl_f,
                tp_f,
            )
            _force_hold(vd_dict, "low_rr")
            return vd_dict

        vd_dict["risk_reward_ratio"] = round(rr, 3)

    return vd_dict


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
            if daily_pnl is None:
                # No snapshot in today's UTC window — leave constraint absent so the
                # verdict prompt does not advertise a budget that we cannot back up.
                pass
            elif daily_pnl < 0:
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
        pair = get_pair(state).canonical()
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
        logger.warning("Redis unavailable for downgrade check, keeping AI verdict", exc_info=True)
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
        from cryptotrader.chat.runtime_registry import get_event_bus
        from cryptotrader.debate.verdict import TradeVerdict

        verdict = TradeVerdict(action="hold", confidence=0.0, reasoning="All agents failed — no real data")
        get_metrics_collector().inc_verdict(action=verdict.action)
        # Publish verdict_ready even on the all-mock fallback path so the
        # frontend always sees a verdict event before stream_done.
        bus = get_event_bus((state.get("metadata") or {}).get("session_id"))
        if bus is not None:
            await bus.publish(
                "verdict_ready",
                {
                    "action": verdict.action,
                    "confidence": verdict.confidence,
                    "position_scale": 0.0,
                    "reasoning": verdict.reasoning,
                },
            )
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
        position_context = state["data"].get("position_context")
        trend_context = state["data"].get("trend_context")
        verdict = await make_verdict_llm(
            analyses,
            model=model,
            constraints=constraints,
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

    verdict_data = {
        "action": verdict.action,
        "confidence": verdict.confidence,
        "position_scale": verdict.position_scale,
        "divergence": verdict.divergence,
        "reasoning": verdict.reasoning,
        "thesis": verdict.thesis,
        "stop_loss": getattr(verdict, "stop_loss", None),
        "take_profit": getattr(verdict, "take_profit", None),
        "invalidation": verdict.invalidation,
        "target_price": getattr(verdict, "target_price", "") or "",
        "verdict_source": verdict_source,
    }

    # Server-side guardrails: confidence-scaled sizing, applied: enforcement,
    # silent-agent confidence cap, ATR-based stop, R:R ≥ 1.5. These run on AI
    # and weighted verdicts alike so the safety net does not depend on the LLM
    # following the prompt.
    if verdict_source != "hold_all_mock":
        # Pull live price + ATR from the snapshot/trend context for the
        # invalidation distance + R:R checks (Guardrails 4 + 5).
        _summary = state["data"].get("snapshot_summary") or {}
        _trend = state["data"].get("trend_context") or {}
        _entry = float(_summary.get("price") or _trend.get("current_price") or 0.0) or None
        _atr = _trend.get("atr_14")
        # Weighted path has no LLM to produce SL/TP — synthesize them from
        # entry+ATR so the hard-reject gates don't force every high-consensus
        # pair to hold. AI path keeps the LLM-provided values.
        if verdict_source == "weighted":
            _inject_weighted_sl_tp(verdict_data, _entry, _atr)
        _post_process_verdict(
            verdict,
            raw_analyses,
            verdict_data,
            entry_price=_entry,
            atr=_atr,
        )

    # spec 022 FR-022-15/T009: fire journal hook for Phase 1 hard-reject paths
    _phase1_reason = verdict_data.pop("_phase1_rejected_reason", None)
    if _phase1_reason:
        try:
            from cryptotrader.nodes.journal import record_phase1_rejection
            from cryptotrader.tracing import get_trace_id

            pair_str = get_pair(state).canonical()
            await record_phase1_rejection(
                trace_id=get_trace_id() or "",
                pair=pair_str,
                reason=_phase1_reason,
                payload={"action": verdict_data.get("action"), "verdict_source": verdict_source},
            )
        except Exception:
            logger.debug("phase1_rejection journal hook failed (non-blocking)", exc_info=True)

    await _process_schedule_follow_up(state, verdict_data)

    from cryptotrader.chat.runtime_registry import get_event_bus

    event_bus = get_event_bus((state.get("metadata") or {}).get("session_id"))
    if event_bus is not None:
        await event_bus.publish(
            "verdict_ready",
            {
                "action": verdict_data["action"],
                "confidence": verdict_data["confidence"],
                "position_scale": verdict_data["position_scale"],
                "reasoning": verdict_data["reasoning"][:300],
            },
        )

    return {"data": {"verdict": verdict_data}}


async def _process_schedule_follow_up(state: ArenaState, verdict_data: dict) -> None:
    """Register a temporary trigger rule if the verdict includes schedule_follow_up."""
    follow_up = verdict_data.get("schedule_follow_up")
    if not follow_up or not isinstance(follow_up, dict):
        return

    depth = state["metadata"].get("schedule_depth", 0)
    if depth >= 3:
        logger.info("Schedule depth %d >= 3, skipping follow-up registration", depth)
        return

    try:
        from datetime import datetime, timedelta
        from functools import partial

        from cryptotrader._compat import UTC
        from cryptotrader.config import load_config
        from cryptotrader.db import get_async_session
        from cryptotrader.triggers.store import TriggerRuleStore

        config = load_config()
        if not config.infrastructure.database_url:
            return

        session_factory = partial(get_async_session, config.infrastructure.database_url)
        store = TriggerRuleStore(session_factory)

        ttl_hours = min(int(follow_up.get("ttl_hours", 24)), 72)
        pair_str = get_pair(state).canonical()
        await store.create_rule(
            {
                "name": follow_up.get("name", f"agent-follow-up-{pair_str}"),
                "trigger_type": follow_up.get("trigger_type", "price_threshold"),
                "pair": pair_str,
                "parameters": follow_up.get("parameters", {}),
                "cooldown_minutes": int(follow_up.get("cooldown_minutes", 60)),
                "created_by": "agent",
                "schedule_depth": depth + 1,
                "ttl_expires_at": datetime.now(UTC) + timedelta(hours=ttl_hours),
            }
        )
        logger.info("Registered agent follow-up trigger (depth=%d)", depth + 1)
    except Exception:
        logger.warning("Failed to register schedule follow-up", exc_info=True)


# Module-level cache for RiskGate to preserve circuit breaker state
_risk_gate_cache: dict[tuple[str, int], Any] = {}

# Lazy notifier
_notifier_instance: Any = None


def _get_notifier(state: ArenaState) -> Any:
    global _notifier_instance
    if _notifier_instance is None:
        from cryptotrader.config import load_config
        from cryptotrader.notifications import Notifier

        cfg = load_config().notifications
        _notifier_instance = Notifier(cfg.webhook_url, cfg.enabled, cfg.events, cfg.webhook_timeout, cfg.telegram)
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

    pair = get_pair(state).canonical()
    try:
        exchange, _ = await _get_exchange(state, pair)
        t0 = time.monotonic()
        await exchange.get_balance()
        return int((time.monotonic() - t0) * 1000)
    except Exception:
        logger.warning("API latency check failed, using default", exc_info=True)
        return 100


async def _build_risk_portfolio(state: ArenaState, config) -> dict | None:
    """Assemble the portfolio dict for risk checks. Returns None when the live exchange
    reports zero balance and trading must be rejected upstream."""
    from cryptotrader.portfolio.manager import PortfolioManager, read_portfolio_from_exchange

    recent_prices, returns_daily = _extract_ohlcv_returns(state)
    exchange_portfolio = await read_portfolio_from_exchange(state)

    db_url = state["metadata"].get("database_url")
    pm = PortfolioManager(db_url)
    try:
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
        pm_returns = await pm.get_returns()
    except Exception:
        logger.warning("Portfolio snapshot data fetch failed", exc_info=True)
        daily_pnl = None  # surface "unknown" downstream rather than synthetic 0
        drawdown = 0.0
        pm_returns = []

    if exchange_portfolio and exchange_portfolio.get("total_value", 0) > 0:
        total_value = exchange_portfolio["total_value"]
        positions = exchange_portfolio.get("positions", {})
    elif state["metadata"].get("engine") == "live":
        return None
    else:
        total_value = config.backtest.initial_capital
        positions = {}

    api_latency_ms = await _measure_api_latency(state)
    # spec 021 H3 (option 2): expose the exchange trade-endpoint cooldown
    # so ExchangeHealthCheck can short-circuit actionable verdicts when an
    # earlier pair in this cycle already hit OKX 50013 / ExchangeNotAvailable.
    exchange_id = state["metadata"].get("exchange_id", "okx")
    try:
        from cryptotrader.execution.exchange import trade_unavailable_remaining_s

        trade_unavail_s = trade_unavailable_remaining_s(exchange_id)
    except Exception:
        trade_unavail_s = 0.0
    return {
        "total_value": total_value,
        "positions": positions,
        "cash": exchange_portfolio.get("cash", 0) if exchange_portfolio else 0,
        "daily_pnl": daily_pnl,
        "drawdown": drawdown,
        "returns_60d": _merge_returns(pm_returns, returns_daily),
        "recent_prices": recent_prices,
        "funding_rate": state["data"].get("snapshot_summary", {}).get("funding_rate", 0),
        "api_latency_ms": api_latency_ms,
        "trade_unavailable_remaining_s": trade_unavail_s,
        "pair": get_pair(state).canonical(),
    }


def _log_risk_outcome(state: ArenaState, result, vd_action: str) -> None:
    pair_str = get_pair(state).canonical()
    if result.passed:
        logger.info("Risk gate PASSED for %s (action=%s)", pair_str, vd_action)
        return
    # Structured rejection log — check_name + reason carry the full context for alerting.
    _structlog.warning(
        "risk_gate_rejected",
        pair=pair_str,
        check_name=result.rejected_by or "unknown",
        reason=result.reason or "",
    )
    logger.warning(
        "Risk gate REJECTED for %s: check_name=%s reason=%s",
        pair_str,
        result.rejected_by,
        result.reason,
    )
    get_metrics_collector().inc_risk_rejected(check_name=result.rejected_by or "unknown")


@node_logger()
async def risk_check(state: ArenaState) -> dict:
    """Run all risk gate checks on the verdict."""
    from cryptotrader.config import load_config
    from cryptotrader.models import TradeVerdict
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    redis_url = state["metadata"].get("redis_url")
    # Resolve perp leverage for the dual-cap risk model. Order:
    #   1. state["metadata"]["leverage"] — explicit override (tests, backtests)
    #   2. config.exchanges[active_exchange].leverage — production live path
    #   3. 1 (no leverage) — safe fallback
    leverage = state["metadata"].get("leverage")
    if leverage is None:
        exchange_id = state["metadata"].get("exchange") or config.exchange_id or "okx"
        creds = config.exchanges.get(exchange_id)
        leverage = creds.leverage if creds else 1
    cache_key = (redis_url or "_default", int(leverage))
    if cache_key not in _risk_gate_cache:
        _risk_gate_cache[cache_key] = RiskGate(config.risk, RedisStateManager(redis_url), leverage=leverage)
    gate = _risk_gate_cache[cache_key]

    vd = state["data"]["verdict"]
    # Filter to only the TradeVerdict dataclass fields. The verdict dict can
    # legitimately carry extra keys that the dataclass doesn't model — e.g.
    # ``guardrails`` (audit trail of which server-side guardrails fired in
    # ``_post_process_verdict``) or ``verdict_source``-adjacent metadata.
    # Without this filter every cycle whose verdict triggered a guardrail
    # crashed at ``TradeVerdict.__init__() got an unexpected keyword argument``,
    # which silently dropped the pair from the cycle output (seen at 22:39 SOL,
    # 03:55 LINK/SOL — the "silent drop" bug observed during overnight monitoring).
    verdict = TradeVerdict(**{k: v for k, v in vd.items() if k in TradeVerdict.__dataclass_fields__})

    # In backtest mode, portfolio is pre-built by the engine — skip exchange queries
    portfolio = state["data"].get("portfolio")
    if portfolio is None:
        portfolio = await _build_risk_portfolio(state, config)
        if portfolio is None:
            # Close action is risk-reduction by definition — never block it on
            # portfolio-read failure. The close order will either succeed at
            # the exchange (proving the read failure was transient / cache /
            # margin-mode quirk) or fail naturally at execute() with a real
            # exchange-side error. Blocking close here means the operator
            # cannot exit a losing position whenever OKX has a hiccup, which
            # is the exact opposite of what risk gating should achieve.
            if verdict.action == "close":
                logger.info(
                    "Portfolio unreadable but verdict is close — letting it through "
                    "(risk reduction must not be blocked)"
                )
                # Synthesize a minimal portfolio so the gate has something to
                # work with — gate.check whitelists close anyway, so values
                # don't matter. We just need a non-None dict.
                portfolio = {"total_value": 0, "positions": {}, "cash": 0}
            else:
                err = state["data"].get("_portfolio_read_error") or {}
                err_suffix = f" [{err['type']}: {err['msg']}]" if err else ""
                logger.warning("Exchange returned 0 total_value in live mode — rejecting%s", err_suffix)
                return {
                    "data": {
                        "risk_gate": {
                            "passed": False,
                            "rejected_by": "portfolio_unknown",
                            "reason": f"Exchange returned 0 balance — cannot trade safely{err_suffix}",
                        }
                    }
                }

    result = await gate.check(verdict, portfolio)
    _log_risk_outcome(state, result, verdict.action)

    pair_str = get_pair(state).canonical()

    # Fire circuit_breaker notification if triggered
    if not result.passed and result.rejected_by == "daily_loss_limit" and "Circuit breaker" in (result.reason or ""):
        try:
            notifier = _get_notifier(state)
            await notifier.notify("circuit_breaker", {"pair": pair_str, "reason": result.reason})
        except Exception:
            logger.info("Circuit-breaker notification failed", exc_info=True)

    # PROD-I3: Apply scale proposals via the return delta (LangGraph contract),
    # not by mutating state["data"]["verdict"] in place. ``GateResult.scale_adjustment``
    # is the min of all CheckResult proposals from passing checks.
    final_scale = vd.get("position_scale", 1.0)
    if result.passed and result.scale_adjustment is not None:
        final_scale = min(final_scale, result.scale_adjustment)
        if final_scale != vd.get("position_scale", 1.0):
            logger.info(
                "Position scale adjusted by risk gate: %.2f -> %.2f for %s",
                vd.get("position_scale", 1.0),
                final_scale,
                pair_str,
            )

    risk_gate_data = {
        "passed": result.passed,
        "rejected_by": result.rejected_by,
        "reason": result.reason,
        "scale_adjustment": result.scale_adjustment,
    }

    from cryptotrader.chat.runtime_registry import get_event_bus

    event_bus = get_event_bus((state.get("metadata") or {}).get("session_id"))
    if event_bus is not None:
        await event_bus.publish(
            "risk_checked",
            {
                "allowed": result.passed,
                "flags": [result.rejected_by] if result.rejected_by else [],
                "reason": result.reason or "",
            },
        )

    delta: dict = {"data": {"risk_gate": risk_gate_data}}
    if result.passed and final_scale != vd.get("position_scale", 1.0):
        delta["data"]["verdict"] = {**vd, "position_scale": final_scale}
    return delta


def risk_router(state: ArenaState) -> str:
    """Route to 'approved' or 'rejected' based on risk gate result."""
    rg = state["data"].get("risk_gate", {})
    return "approved" if rg.get("passed", False) else "rejected"
