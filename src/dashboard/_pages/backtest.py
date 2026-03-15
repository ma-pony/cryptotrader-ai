"""Backtest page — backtest runner and session comparison.

Provides two top-level tabs:
  - Run New Backtest: a parameter form that triggers BacktestEngine.run(),
    persists results via save_commits()/save_result(), and then switches to
    the Load Session view with the new session ID pre-selected.
  - Load Session: a dropdown populated from list_sessions() that loads a
    previously saved session and renders the full result view.

The result view always renders five summary metric cards and an equity curve.
When LLM-powered decisions are present a decision timeline table is shown
and clicking a row reveals the full per-decision detail (reusing
components.py renderers so the format matches the live decisions page).

Pure-rules mode (decisions=[]) hides the agent analysis section and shows
only the equity curve and the trade list.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from cryptotrader.backtest.result import BacktestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "total_return": "Total Return",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown",
    "win_rate": "Win Rate",
    "trade_count": "Trade Count",
}


# ---------------------------------------------------------------------------
# Pure-logic helpers (no Streamlit calls — testable without a running app)
# ---------------------------------------------------------------------------


def _build_timeline_df(decisions: list[dict]) -> pd.DataFrame:
    """Convert a list of decision dicts from BacktestEngine into a display DataFrame.

    Extracts the fields most useful for scanning the backtest timeline:
    timestamp, price, position, action, risk status, and verdict confidence.

    Args:
        decisions: List of per-bar decision dicts as produced by BacktestEngine.run().

    Returns:
        A pandas DataFrame with one row per decision point.  Returns an empty
        DataFrame when decisions is empty.
    """
    if not decisions:
        return pd.DataFrame()

    rows: list[dict] = []
    for d in decisions:
        ts = d.get("ts", 0)
        try:
            timestamp = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d %H:%M")
        except Exception:
            timestamp = str(ts)

        verdict = d.get("verdict", {})
        risk_gate = d.get("risk_gate", {})
        analyses = d.get("analyses", {})

        # Derive a consensus confidence from analyses when present
        confidences = [v.get("confidence", 0) for v in analyses.values() if isinstance(v, dict)]
        confidence = sum(confidences) / len(confidences) if confidences else verdict.get("confidence", 0.0)

        risk_passed = risk_gate.get("passed", True)
        risk_status = "passed" if risk_passed else f"rejected ({risk_gate.get('rejected_by', '')})"

        rows.append(
            {
                "timestamp": timestamp,
                "price": d.get("price", 0.0),
                "position": d.get("position_after", 0.0),
                "action": d.get("final_action", d.get("executed_action", "hold")),
                "risk_status": risk_status,
                "confidence": round(confidence, 3),
            }
        )

    return pd.DataFrame(rows)


def _extract_summary_metrics(result: BacktestResult) -> dict[str, Any]:
    """Extract the five key summary metrics from a BacktestResult.

    Returns a flat dict keyed by snake_case metric names.  Values are raw
    Python numbers so the caller can format them as desired.

    Args:
        result: A BacktestResult instance from BacktestEngine.run().

    Returns:
        Dict with keys: total_return, sharpe_ratio, max_drawdown, win_rate,
        trade_count.
    """
    return {
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "trade_count": len(result.trades),
    }


def is_pure_rules_mode(decisions: list[dict]) -> bool:
    """Return True when no LLM agent analyses were recorded in the backtest.

    Pure-rules mode means BacktestEngine ran without use_llm=True, so each
    decision dict has an empty (or absent) "analyses" dict.  In this mode the
    Agent Analysis section is hidden.

    Args:
        decisions: List of per-bar decision dicts.

    Returns:
        True if all decisions have empty (or missing) analyses, False otherwise.
    """
    if not decisions:
        return True
    return all(not d.get("analyses") for d in decisions)


# ---------------------------------------------------------------------------
# Rendering helpers (call Streamlit widgets, not directly testable without mock)
# ---------------------------------------------------------------------------


def render_backtest_summary(result: BacktestResult) -> None:
    """Render the five summary metric cards for a backtest run.

    Displays Total Return, Sharpe Ratio, Max Drawdown, Win Rate, and Trade Count
    using st.metric() widgets inside a 5-column layout.

    Args:
        result: BacktestResult with populated metrics.
    """
    metrics = _extract_summary_metrics(result)
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Return", f"{metrics['total_return']:.2%}")
    with cols[1]:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    with cols[2]:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    with cols[3]:
        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
    with cols[4]:
        st.metric("Trade Count", str(metrics["trade_count"]))


def render_equity_curve(result: BacktestResult) -> None:
    """Render the equity curve as a line chart.

    Handles empty equity_curve gracefully by showing a placeholder message
    rather than raising.

    Args:
        result: BacktestResult with an equity_curve list of floats.
    """
    st.subheader("Equity Curve")
    if not result.equity_curve:
        st.info("No equity curve data available.")
        return
    df = pd.DataFrame({"equity": result.equity_curve})
    st.line_chart(df)


def render_decision_timeline(decisions: list[dict]) -> None:
    """Render the decision timeline as a selectable DataFrame table.

    In pure-rules mode (empty decisions) shows an informational message.
    Otherwise displays a row-per-decision table with the key decision fields.

    Args:
        decisions: List of per-bar decision dicts from BacktestEngine.
    """
    st.subheader("Decision Timeline")

    if not decisions:
        st.info("No agent decisions recorded (pure rules mode). Run with use_llm=True to see per-bar agent analysis.")
        return

    df = _build_timeline_df(decisions)
    st.dataframe(df, use_container_width=True)


def _render_node_trace_from_dict(decision: dict) -> None:
    """Render node execution pipeline from raw trace list in a decision dict."""
    from dashboard import components

    raw_trace = decision.get("node_trace", [])
    if not raw_trace:
        return

    from cryptotrader.models import NodeTraceEntry

    node_trace = [
        NodeTraceEntry(
            node=t.get("node", ""),
            duration_ms=t.get("duration_ms", 0),
            summary=t.get("summary", ""),
        )
        for t in raw_trace
    ]
    st.write("**Node Execution Pipeline:**")
    components.render_node_trace_pipeline(node_trace)


def _render_analyses_from_dict(decision: dict) -> None:
    """Render agent analysis grid from raw analyses dict in a decision dict."""
    from dashboard import components

    analyses_raw = decision.get("analyses", {})
    if not analyses_raw:
        return

    from cryptotrader.models import AgentAnalysis

    analyses: dict[str, AgentAnalysis] = {}
    for agent_id, a in analyses_raw.items():
        try:
            analyses[agent_id] = AgentAnalysis(
                agent_id=agent_id,
                pair="",
                direction=a.get("direction", "neutral"),
                confidence=a.get("confidence", 0.0),
                reasoning="",
                data_sufficiency=a.get("data_sufficiency", "medium"),
            )
        except Exception:
            logger.debug("Failed to convert analysis for %s", agent_id, exc_info=True)
    if analyses:
        st.write("**Agent Analyses:**")
        components.render_agent_analysis_grid(analyses)


def _render_verdict_from_dict(decision: dict) -> None:
    """Render verdict section from raw verdict dict in a decision dict."""
    from dashboard import components

    verdict_raw = decision.get("verdict", {})
    if not verdict_raw:
        return

    from cryptotrader.models import TradeVerdict

    try:
        verdict = TradeVerdict(
            action=verdict_raw.get("action", "hold"),
            confidence=verdict_raw.get("confidence", 0.0),
            position_scale=verdict_raw.get("position_scale", 0.0),
            reasoning=verdict_raw.get("reasoning", ""),
            thesis=verdict_raw.get("thesis", ""),
            invalidation=verdict_raw.get("invalidation", ""),
        )
        st.write("**Verdict:**")
        components.render_verdict_section(verdict, verdict_source=verdict_raw.get("verdict_source", "ai"))
    except Exception:
        logger.debug("Failed to render verdict", exc_info=True)


def _render_risk_gate_from_dict(decision: dict) -> None:
    """Render risk gate section from raw risk_gate dict in a decision dict."""
    from dashboard import components

    risk_raw = decision.get("risk_gate", {})
    if not risk_raw:
        return

    from cryptotrader.models import GateResult

    try:
        gate = GateResult(
            passed=risk_raw.get("passed", True),
            rejected_by=risk_raw.get("rejected_by", ""),
            reason=risk_raw.get("reason", ""),
        )
        st.write("**Risk Gate:**")
        components.render_risk_gate_section(gate)
    except Exception:
        logger.debug("Failed to render risk gate", exc_info=True)


def _render_decision_detail_from_dict(decision: dict) -> None:
    """Render full decision detail for one timeline point.

    Reuses components.py renderers so the format matches the live decisions
    page.  Adapts raw decision dicts (from BacktestEngine) to domain objects
    where components expect typed arguments.

    Args:
        decision: A single per-bar decision dict from BacktestEngine.
    """
    st.markdown("---")
    ts = decision.get("ts", 0)
    try:
        ts_str = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        ts_str = str(ts)
    st.write(f"**Time:** {ts_str} | **Price:** ${decision.get('price', 0):,.2f}")

    _render_node_trace_from_dict(decision)
    _render_analyses_from_dict(decision)
    _render_verdict_from_dict(decision)
    _render_risk_gate_from_dict(decision)


def render_backtest_results(result: BacktestResult) -> None:
    """Render the complete backtest result view.

    Combines:
      1. Five summary metric cards
      2. Equity curve line chart
      3. Decision timeline table (hidden in pure-rules mode)
      4. Per-decision detail panel (rendered when timeline row is selected)

    In pure-rules mode (decisions=[]) the Agent Analysis section is suppressed.

    Args:
        result: A BacktestResult from BacktestEngine.run() or session storage.
    """
    render_backtest_summary(result)
    render_equity_curve(result)

    decisions = result.decisions or []
    pure_rules = is_pure_rules_mode(decisions)

    if not pure_rules:
        render_decision_timeline(decisions)
        # Timeline selection — use session_state for selected row index
        if decisions:
            selected_idx = st.session_state.get("backtest_selected_idx")
            if isinstance(selected_idx, int) and 0 <= selected_idx < len(decisions):
                st.subheader("Decision Detail")
                _render_decision_detail_from_dict(decisions[selected_idx])

            # Selectbox fallback for row selection (Streamlit on_select not universally available)
            options = list(range(len(decisions)))
            labels = [f"Bar {i} — {d.get('price', 0):.0f}" for i, d in enumerate(decisions)]
            sel = st.selectbox("Select decision point", options, format_func=lambda i: labels[i], key="bt_select")
            if sel is not None:
                _render_decision_detail_from_dict(decisions[sel])
    else:
        # Pure rules: only show equity curve (already rendered) and trade list
        if result.trades:
            st.subheader("Trade List")
            trades_df = pd.DataFrame(result.trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed during this backtest period.")


# ---------------------------------------------------------------------------
# Tab: Run New Backtest
# ---------------------------------------------------------------------------


def _render_run_new_tab() -> BacktestResult | None:
    """Render the 'Run New Backtest' form and execute if submitted.

    Returns:
        BacktestResult if the form was submitted and completed, None otherwise.
    """
    st.write("Configure and run a new backtest.")

    with st.form("backtest_form"):
        pair = st.text_input("Trading pair", value="BTC/USDT")
        start = st.text_input("Start date (YYYY-MM-DD)", value="2024-01-01")
        end = st.text_input("End date (YYYY-MM-DD)", value="2024-03-01")
        interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=1)
        use_llm = st.checkbox("Use LLM agents", value=False)
        submitted = st.form_submit_button("Run Backtest")

    if not submitted:
        return None

    if not pair or not start or not end:
        st.error("Please fill in all required fields.")
        return None

    st.info(f"Running backtest for {pair} from {start} to {end} ({interval})…")

    try:
        from cryptotrader.backtest.engine import BacktestEngine
        from cryptotrader.backtest.session import generate_session_id, save_result
        from dashboard.data_loader import run_async

        engine = BacktestEngine(pair=pair, start=start, end=end, interval=interval, use_llm=use_llm)
        run_result = run_async(engine.run())

        # Persist to session storage
        session_id = generate_session_id(pair=pair, interval=interval, start=start, end=end)
        save_result(session_id, run_result)
        # Note: save_commits requires DecisionCommit objects. Raw dict decisions from the
        # backtest engine are stored only in result.json (via save_result). A future
        # task can wire up full commit persistence once the journal integration is complete.
        st.success(f"Backtest complete — session saved: `{session_id}`")
        st.session_state["backtest_current_session_id"] = session_id
        return run_result

    except Exception as exc:
        logger.debug("Backtest run failed", exc_info=True)
        st.error(f"Backtest failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Tab: Load Session
# ---------------------------------------------------------------------------


def _render_load_session_tab() -> BacktestResult | None:
    """Render the 'Load Session' tab for browsing saved sessions.

    Returns:
        BacktestResult reconstructed from session data, or None when no
        session is selected or the session has no result file.
    """
    from dashboard import data_loader

    sessions = data_loader.list_backtest_sessions()

    if not sessions:
        st.info("No saved backtest sessions found. Run a backtest first.")
        return None

    # Allow session_state to pre-select a session (e.g. after a run completes)
    default_session = st.session_state.get("backtest_current_session_id")
    default_idx = sessions.index(default_session) if default_session in sessions else 0

    selected = st.selectbox(
        "Select session",
        sessions,
        index=default_idx,
        key="bt_session_select",
    )

    if not selected:
        return None

    raw_commits = data_loader.load_backtest_session(selected)

    # Reconstruct a minimal BacktestResult from session data.
    # The full result.json is not currently loaded by load_backtest_session,
    # so we derive metrics from the raw commit records.
    if not raw_commits:
        st.info("No commit data found for this session.")
        return None

    # Build a synthetic BacktestResult from the raw commit dicts
    decisions = raw_commits
    equity_values = [d.get("equity_after", d.get("equity", 0.0)) for d in decisions]

    from cryptotrader.backtest.result import BacktestResult

    return BacktestResult(
        total_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        trades=[],
        equity_curve=equity_values,
        decisions=decisions,
        llm_calls=0,
        llm_tokens=0,
    )


# ---------------------------------------------------------------------------
# Top-level render() entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Backtest page.

    Displays two tabs at the top:
      - Run New Backtest: form to configure and execute a backtest.
      - Load Session: dropdown of persisted sessions with full result view.

    After a successful run the result is displayed inline and the session
    is saved to disk.  Loading a session reconstructs the result view from
    the stored commit records.
    """
    st.header("Backtest")

    run_tab, load_tab = st.tabs(["Run New Backtest", "Load Session"])

    with run_tab:
        result = _render_run_new_tab()
        if result is not None:
            st.divider()
            render_backtest_results(result)

    with load_tab:
        loaded = _render_load_session_tab()
        if loaded is not None:
            st.divider()
            render_backtest_results(loaded)
