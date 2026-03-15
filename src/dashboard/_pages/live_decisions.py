"""Live Decisions page — decision history and full pipeline detail.

Renders a paginated, filterable list of DecisionCommit records from the journal
store.  Clicking a row (or selecting a hash from the fallback selectbox) loads
the full pipeline detail for that decision.

Layout:
    Filter bar: pair dropdown + pagination controls
    Decision list: st.dataframe(on_select="rerun") with selectbox fallback
    Decision detail (conditional on selection):
        Header: timestamp, pair, price, trace_id (OTel link if OTLP_ENDPOINT set)
        Node trace pipeline
        Agent analysis grid
        Experience memory
        Debate section
        Verdict section
        Risk gate + execution section
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import streamlit as st

from dashboard.components import (
    render_agent_analysis_grid,
    render_debate_section,
    render_experience_memory_section,
    render_node_trace_pipeline,
    render_pagination_controls,
    render_risk_gate_section,
    render_verdict_section,
)
from dashboard.data_loader import get_dashboard_config, load_commit_detail, load_journal

if TYPE_CHECKING:
    from cryptotrader.models import DecisionCommit

logger = logging.getLogger(__name__)

# Default number of records per page.
_PAGE_SIZE = 20

# Jaeger-compatible trace URL template.  The endpoint base URL is taken from
# OTLP_ENDPOINT (e.g. "http://jaeger:16686").
_OTEL_TRACE_URL_TEMPLATE = "{base}/trace/{trace_id}"


# ---------------------------------------------------------------------------
# Pure helper functions (importable for unit testing)
# ---------------------------------------------------------------------------


def _extract_pairs(commits: list[DecisionCommit]) -> list[str]:
    """Return sorted unique pairs found in *commits*.

    Args:
        commits: List of DecisionCommit objects.

    Returns:
        Alphabetically sorted list of unique trading pair strings.
    """
    return sorted({c.pair for c in commits})


def _build_trace_link(trace_id: str | None) -> str | None:
    """Build an OTel trace viewer URL for *trace_id* if OTLP_ENDPOINT is set.

    Args:
        trace_id: The distributed trace ID string, or None/empty.

    Returns:
        A URL string containing the trace_id, or None when OTLP_ENDPOINT is
        not set or trace_id is empty / None.
    """
    if not trace_id:
        return None
    endpoint = os.environ.get("OTLP_ENDPOINT", "")
    if not endpoint:
        return None
    base = endpoint.rstrip("/")
    return _OTEL_TRACE_URL_TEMPLATE.format(base=base, trace_id=trace_id)


def _commits_to_rows(commits: list[DecisionCommit]) -> list[dict[str, Any]]:
    """Convert a list of DecisionCommit objects to a flat list of row dicts.

    The row dicts are used to populate the st.dataframe widget.

    Args:
        commits: List of DecisionCommit objects, newest-first.

    Returns:
        List of row dicts, one per commit.
    """
    rows: list[dict[str, Any]] = []
    for commit in commits:
        price = commit.snapshot_summary.get("price", "") if commit.snapshot_summary else ""
        action = commit.verdict.action if commit.verdict else "—"
        rows.append(
            {
                "hash": commit.hash,
                "pair": commit.pair,
                "time (UTC)": commit.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "price": price,
                "action": action,
                "source": commit.verdict_source,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Detail rendering
# ---------------------------------------------------------------------------


def _render_decision_header(commit: DecisionCommit) -> None:
    """Render the decision header: timestamp, pair, price, trace_id."""
    price = commit.snapshot_summary.get("price", "N/A") if commit.snapshot_summary else "N/A"
    st.subheader(f"{commit.pair} — {commit.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    st.write(f"**Market price:** {price}")

    if commit.trace_id:
        trace_link = _build_trace_link(commit.trace_id)
        if trace_link:
            st.markdown(f"**Trace ID:** [{commit.trace_id}]({trace_link})")
        else:
            st.write(f"**Trace ID:** `{commit.trace_id}`")


def _render_portfolio_after(portfolio: dict[str, Any]) -> None:
    """Render portfolio state (equity, cash, positions, stop-loss) after execution."""
    stop_triggered = portfolio.get("stop_loss_triggered")
    if stop_triggered:
        stop_price = portfolio.get("stop_loss_price", "N/A")
        st.warning(f"Stop-loss triggered at {stop_price}")

    total_value = portfolio.get("total_value")
    cash = portfolio.get("cash")
    if total_value is not None:
        st.write(f"**Portfolio total value:** {total_value:,.2f}")
    if cash is not None:
        st.write(f"**Cash:** {cash:,.2f}")

    positions = portfolio.get("positions", {})
    if positions:
        st.write("**Positions after execution:**")
        for symbol, pos in positions.items():
            if isinstance(pos, dict):
                size = pos.get("size", "N/A")
                side = pos.get("side", "")
                st.write(f"  - {symbol}: {size} ({side})")
            else:
                st.write(f"  - {symbol}: {pos}")


def _render_execution_section(commit: DecisionCommit) -> None:
    """Render the execution action, stop-loss status, and portfolio changes."""
    st.subheader("Execution")

    if commit.order is not None:
        st.write(
            f"**Order:** {commit.order.side.upper()} {commit.order.amount} {commit.pair}"
            f" @ {commit.order.price} — status: `{commit.order.status}`"
        )
    else:
        st.write("**Execution action:** No order placed.")

    if commit.fill_price is not None:
        st.write(f"**Fill price:** {commit.fill_price}")
    if commit.slippage is not None:
        st.write(f"**Slippage:** {commit.slippage:.4f}")
    if commit.pnl is not None:
        st.write(f"**PnL:** {commit.pnl:+.4f}")

    _render_portfolio_after(commit.portfolio_after or {})


def _render_decision_detail(commit: DecisionCommit) -> None:
    """Render the full pipeline detail for a single DecisionCommit.

    Sections rendered in order:
        1. Header (timestamp, pair, price, trace_id)
        2. Node trace pipeline
        3. Agent analysis grid
        4. Experience memory
        5. Debate section
        6. Verdict section
        7. Risk gate + execution section

    Args:
        commit: The fully-populated DecisionCommit to display.
    """
    st.divider()
    _render_decision_header(commit)

    # Node trace pipeline
    if commit.node_trace:
        st.subheader("Node Execution Pipeline")
        render_node_trace_pipeline(commit.node_trace)

    # Agent analysis grid
    if commit.analyses:
        st.subheader("Agent Analyses")
        render_agent_analysis_grid(commit.analyses)

    # Experience memory
    st.subheader("Experience Memory")
    render_experience_memory_section(commit.experience_memory)

    # Debate section
    st.subheader("Debate")
    render_debate_section(
        debate_rounds=commit.debate_rounds,
        challenges=commit.challenges or [],
        debate_skip_reason=commit.debate_skip_reason or "",
        consensus_metrics=commit.consensus_metrics,
    )

    # Verdict section
    if commit.verdict is not None:
        st.subheader("Verdict")
        render_verdict_section(commit.verdict, commit.verdict_source)

    # Risk gate section
    if commit.risk_gate is not None:
        st.subheader("Risk Gate")
        render_risk_gate_section(commit.risk_gate)

    # Execution section
    _render_execution_section(commit)


# ---------------------------------------------------------------------------
# Main page render function
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Live Decisions page.

    Catches DB exceptions at the top level and shows st.error + st.stop so
    that the rest of the app is not affected.
    """
    st.header("Live Decisions")
    db_url = get_dashboard_config()["db_url"]

    # --- Load journal for pair extraction (needed to build the filter) ---
    try:
        # Fetch a larger batch to extract all available pairs for the dropdown.
        # We'll refetch with the filter applied below.
        all_commits = load_journal(db_url, limit=200)
    except Exception as exc:
        logger.warning("Failed to load journal for pair extraction", exc_info=True)
        st.error(f"Database error: {exc}")
        st.stop()
        return

    available_pairs = _extract_pairs(all_commits)

    # --- Filter bar ---
    col_filter, col_page = st.columns([2, 3])
    with col_filter:
        pair_options: list[str | None] = [None, *available_pairs]
        selected_pair: str | None = st.selectbox(
            "Pair",
            options=pair_options,
            format_func=lambda x: "All pairs" if x is None else x,
            key="ld_pair_filter",
        )

    with col_page:
        offset, limit = render_pagination_controls(
            total=len(all_commits),
            page_size=_PAGE_SIZE,
            key="ld_page",
        )

    # --- Load filtered / paginated journal ---
    try:
        commits = load_journal(db_url, limit=limit, pair=selected_pair, offset=offset)
    except Exception as exc:
        logger.warning("Failed to load journal", exc_info=True)
        st.error(f"Database error: {exc}")
        st.stop()
        return

    if not commits:
        st.info("No decisions found.")
        return

    # --- Decision list ---
    rows = _commits_to_rows(commits)
    selected_hash: str | None = None

    try:
        result = st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="ld_df",
        )
        sel_rows: list[int] = result.selection.rows if result else []
        if sel_rows:
            selected_hash = rows[sel_rows[0]]["hash"]
    except TypeError:
        # Streamlit version does not support on_select — fall back to selectbox.
        hash_options = [r["hash"] for r in rows]
        selected_hash = st.selectbox(
            "Select a decision by hash",
            options=hash_options,
            key="ld_hash_select",
        )

    # --- Decision detail ---
    if selected_hash:
        try:
            commit = load_commit_detail(db_url, selected_hash)
        except Exception as exc:
            logger.warning("Failed to load commit detail for %s", selected_hash, exc_info=True)
            st.error(f"Failed to load decision detail: {exc}")
            return

        if commit is not None:
            _render_decision_detail(commit)
        else:
            st.warning(f"Decision {selected_hash!r} not found in the journal.")
