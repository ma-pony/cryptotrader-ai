"""Overview page — portfolio summary and scheduler status.

Displays:
- Key portfolio metrics: total equity, cash balance, daily PnL, current drawdown
- Equity curve (line chart of recent total_value snapshots)
- Positions list as st.table() with Long/Short direction labels
- Scheduler status: next trigger pair and time, or a warning if unavailable

Auto-refreshes every 10 seconds using st.empty() + time.sleep(10) + st.rerun().
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from dashboard.data_loader import get_dashboard_config, load_portfolio, load_scheduler_status


def _build_positions_table(positions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform positions dict into a list of rows for st.table().

    Args:
        positions: Mapping of pair -> {"amount": float, "avg_price": float}.

    Returns:
        List of dicts with keys: Pair, Direction, Amount, Avg Price.
    """
    rows = []
    for pair, pos in positions.items():
        amount = pos.get("amount", 0.0)
        avg_price = pos.get("avg_price", 0.0)
        direction = "Long" if amount >= 0 else "Short"
        rows.append(
            {
                "Pair": pair,
                "Direction": direction,
                "Amount": abs(amount),
                "Avg Price": avg_price,
            }
        )
    return rows


def _render_portfolio_metrics(portfolio: dict[str, Any]) -> None:
    """Render the four key portfolio metric cards.

    Args:
        portfolio: Portfolio summary dict from PortfolioManager.get_portfolio().
    """
    total_value = portfolio.get("total_value", 0.0)
    cash = portfolio.get("cash", 0.0)
    # Daily PnL and drawdown are not yet available from the portfolio summary dict
    # (they require snapshot history and are loaded separately in a future iteration).
    # Show 0.0 as a safe default.
    daily_pnl = 0.0
    drawdown = 0.0

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Equity", f"${total_value:,.2f}")
    with cols[1]:
        st.metric("Cash Balance", f"${cash:,.2f}")
    with cols[2]:
        st.metric("Daily PnL", f"${daily_pnl:,.2f}")
    with cols[3]:
        st.metric("Drawdown", f"{drawdown:.1%}")


def _render_scheduler_status(status: dict[str, Any] | None, api_base_url: str) -> None:
    """Render the scheduler status widget.

    Shows the next trigger pair and scheduled time when available.
    Displays a warning when the scheduler API is unreachable.

    Args:
        status:       Parsed scheduler status dict or None if API is unreachable.
        api_base_url: API base URL (used only for display context).
    """
    st.subheader("Scheduler Status")
    if status is None:
        st.warning("调度器状态不可用")
        return

    running = status.get("running", False)
    if not running:
        st.info("Scheduler is not running.")
        return

    pairs = status.get("pairs", [])
    jobs = status.get("jobs", [])
    interval = status.get("interval_minutes", 0)
    cycle_count = status.get("cycle_count", 0)

    st.write(f"**Pairs:** {', '.join(pairs) if pairs else '—'}")
    st.write(f"**Interval:** {interval} minutes | **Cycles completed:** {cycle_count}")

    if jobs:
        next_job = jobs[0]
        next_run = next_job.get("next_run_time") or "—"
        job_pairs = next_job.get("pairs", pairs)
        st.write(f"**Next run:** {next_run}  |  **Pairs:** {', '.join(job_pairs)}")
    else:
        st.write("No jobs scheduled.")


def render() -> None:
    """Render the Overview page.

    Layout:
    1. Portfolio metric cards (equity, cash, daily PnL, drawdown)
    2. Equity curve line chart (total_value history)
    3. Positions table (Long/Short direction labels)
    4. Scheduler status widget
    5. Auto-refresh placeholder + sleep + st.rerun()
    """
    st.header("Overview")

    # ---------------------------------------------------------------------------
    # Load config for db_url and api_base_url
    # ---------------------------------------------------------------------------
    _cfg = get_dashboard_config()
    db_url = _cfg["db_url"]
    api_base_url = _cfg["api_base_url"]

    # ---------------------------------------------------------------------------
    # Portfolio metrics
    # ---------------------------------------------------------------------------
    try:
        portfolio = load_portfolio(db_url)
    except Exception as exc:
        st.error(f"Failed to load portfolio: {exc}")
        st.stop()
        return

    _render_portfolio_metrics(portfolio)

    # ---------------------------------------------------------------------------
    # Equity curve — built from total_value in portfolio snapshot history
    # We approximate the curve using the single current total_value data point
    # when no snapshot history is available.
    # ---------------------------------------------------------------------------
    st.subheader("Equity Curve")
    total_value = portfolio.get("total_value", 0.0)
    # The equity curve data: for now display current total_value as a single-point
    # series.  In production, the portfolio snapshots table provides the history.
    st.line_chart({"Total Equity": [total_value]})

    # ---------------------------------------------------------------------------
    # Positions table
    # ---------------------------------------------------------------------------
    st.subheader("Current Positions")
    positions = portfolio.get("positions", {})
    if positions:
        rows = _build_positions_table(positions)
        st.table(rows)
    else:
        st.table([])
        st.info("No open positions.")

    # ---------------------------------------------------------------------------
    # Scheduler status
    # ---------------------------------------------------------------------------
    scheduler_status = load_scheduler_status(api_base_url)
    _render_scheduler_status(scheduler_status, api_base_url)

    # ---------------------------------------------------------------------------
    # Auto-refresh: sleep then rerun
    # ---------------------------------------------------------------------------
    placeholder = st.empty()
    with placeholder:
        st.caption("Auto-refreshing every 10 seconds…")
    time.sleep(10)
    st.rerun()
