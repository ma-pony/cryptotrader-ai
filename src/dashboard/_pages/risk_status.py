"""Risk Status page — risk gate status and circuit-breaker management.

Displays:
- Hourly and daily trade counts
- Circuit breaker state (ACTIVE shown in st.error() red container, INACTIVE in st.success())
- Key risk thresholds from config
- Reset circuit breaker button (calls run_async(rsm.reset_circuit_breaker()) then st.rerun())

When Redis is unreachable, load_risk_status() returns None — the page renders
st.warning("风控状态不可用 — Redis 未连接") and returns early without calling st.stop().
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.data_loader import get_dashboard_config, load_risk_status, run_async


def _render_trade_counts(status: dict[str, Any]) -> None:
    """Render hourly and daily trade count metrics.

    Args:
        status: Risk status dict with keys hourly_trade_count and daily_trade_count.
    """
    hourly = status.get("hourly_trade_count", 0)
    daily = status.get("daily_trade_count", 0)

    cols = st.columns(2)
    with cols[0]:
        st.metric("Hourly Trade Count", hourly)
    with cols[1]:
        st.metric("Daily Trade Count", daily)


def _render_circuit_breaker(
    status: dict[str, Any],
    redis_url: str | None,
) -> None:
    """Render the circuit breaker state card.

    When ACTIVE the entire card is rendered inside st.error() (red warning).
    When INACTIVE the state is shown in a success banner.

    A reset button is provided which calls rsm.reset_circuit_breaker() via
    run_async() and then calls st.rerun() to refresh the page.

    Args:
        status:    Risk status dict with key circuit_breaker_active.
        redis_url: Redis connection URL (used to instantiate RedisStateManager
                   for the reset action).
    """
    active = status.get("circuit_breaker_active", False)

    if active:
        st.error("Circuit Breaker: ACTIVE — trading is halted until manually reset or TTL expires.")
        if st.button("Reset Circuit Breaker", key="reset_cb"):
            try:
                from cryptotrader.risk.state import RedisStateManager

                rsm = RedisStateManager(redis_url=redis_url)
                run_async(rsm.reset_circuit_breaker())
            except Exception:
                pass
            st.rerun()
    else:
        st.success("Circuit Breaker: INACTIVE — trading is allowed.")


def _render_risk_thresholds() -> None:
    """Render key risk threshold parameters from config.

    Reads thresholds from config.risk when available.  Falls back to showing
    a generic placeholder when config is unavailable.
    """
    st.subheader("Risk Thresholds")
    try:
        from cryptotrader.config import load_config

        cfg = load_config()
        risk = cfg.risk
        cols = st.columns(3)
        with cols[0]:
            st.metric("Max Daily Loss %", f"{risk.max_daily_loss_pct:.1%}")
        with cols[1]:
            max_stop = getattr(risk, "max_stop_loss_pct", 0.05)
            st.metric("Max Stop Loss %", f"{max_stop:.1%}")
        with cols[2]:
            max_trades = getattr(risk, "max_daily_trades", "—")
            st.metric("Max Daily Trades", max_trades)
    except Exception:
        st.write("Risk threshold configuration unavailable.")


def render() -> None:
    """Render the Risk Status page.

    Layout:
    1. Early return with st.warning() if Redis is unreachable (no st.stop()).
    2. Hourly/daily trade count metrics.
    3. Circuit breaker state card (red st.error() when ACTIVE).
    4. Key risk threshold parameters.
    """
    st.header("Risk Status")

    # ---------------------------------------------------------------------------
    # Resolve redis_url from unified config
    # ---------------------------------------------------------------------------
    redis_url = get_dashboard_config()["redis_url"]

    # ---------------------------------------------------------------------------
    # Load risk status — returns None when Redis is unavailable
    # ---------------------------------------------------------------------------
    status = load_risk_status(redis_url)

    if status is None:
        st.warning("风控状态不可用 — Redis 未连接")
        # Graceful degradation: do NOT call st.stop()
        return

    # ---------------------------------------------------------------------------
    # Trade counts
    # ---------------------------------------------------------------------------
    st.subheader("Trade Counts")
    _render_trade_counts(status)

    # ---------------------------------------------------------------------------
    # Circuit breaker card
    # ---------------------------------------------------------------------------
    st.subheader("Circuit Breaker")
    _render_circuit_breaker(status, redis_url)

    # ---------------------------------------------------------------------------
    # Risk thresholds
    # ---------------------------------------------------------------------------
    _render_risk_thresholds()
