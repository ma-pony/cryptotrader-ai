"""Metrics page — Prometheus metrics snapshot and latency trends.

Displays key MetricsCollector counters and histogram percentiles fetched from
the FastAPI /metrics/summary endpoint.  When the endpoint is unavailable the
page shows a warning and returns early without affecting other pages.

Historical trend data is accumulated in st.session_state across Streamlit reruns
so that latency trends can be visualised as a line chart over time.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.data_loader import get_dashboard_config, load_metrics_summary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of trend samples kept in session_state per metric key.
# Older samples are discarded (FIFO) once this limit is reached.
_MAX_TREND_SAMPLES: int = 60

# Session state keys for trend history
_KEY_PIPELINE_P50 = "metrics_trend_pipeline_p50"
_KEY_PIPELINE_P95 = "metrics_trend_pipeline_p95"
_KEY_EXEC_P50 = "metrics_trend_exec_p50"
_KEY_EXEC_P95 = "metrics_trend_exec_p95"


# ---------------------------------------------------------------------------
# Pure helper: trend data accumulation
# ---------------------------------------------------------------------------


def _accumulate_trend_sample(
    state: dict[str, Any],
    key: str,
    snapshot_time: str,
    value: float,
) -> list[dict[str, Any]]:
    """Append a {timestamp, value} sample to the trend list stored in state[key].

    When the list reaches _MAX_TREND_SAMPLES entries the oldest entry is removed
    (FIFO) before appending the new sample so the list never grows beyond the cap.

    Args:
        state:         The session_state dict (or any plain dict) used for storage.
        key:           The dict key under which the trend list is stored.
        snapshot_time: ISO 8601 timestamp string from the metrics summary.
        value:         The metric value to record.

    Returns:
        The updated trend list (same object stored in state[key]).
    """
    samples: list[dict[str, Any]] = state.get(key, [])
    if len(samples) >= _MAX_TREND_SAMPLES:
        samples = samples[1:]
    samples = [*samples, {"timestamp": snapshot_time, "value": value}]
    state[key] = samples
    return samples


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Metrics page.

    Loads metrics summary from the FastAPI /metrics/summary endpoint.
    Shows a warning and returns early when the endpoint is unavailable.
    """
    st.header("Metrics")

    _cfg = get_dashboard_config()
    api_base_url: str = _cfg["api_base_url"]
    summary = load_metrics_summary(api_base_url)

    if summary is None:
        st.warning("指标端点不可用")
        return

    # ------------------------------------------------------------------
    # Key counters
    # ------------------------------------------------------------------
    st.subheader("Key Counters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("LLM Calls", summary["llm_calls_total"])
    with col2:
        st.metric("Debate Skipped", summary["debate_skipped_total"])
    with col3:
        st.metric("Risk Rejected", summary["risk_rejected_total"])
    with col4:
        st.metric("Trades Executed", summary["trade_executed_total"])
    with col5:
        verdict_dist = summary.get("verdict_distribution", {})
        long_count = verdict_dist.get("long", 0)
        st.metric("Longs", long_count)

    # ------------------------------------------------------------------
    # Verdict distribution
    # ------------------------------------------------------------------
    verdict_distribution = summary.get("verdict_distribution", {})
    if verdict_distribution:
        st.subheader("Verdict Distribution")
        st.bar_chart(verdict_distribution)

    # ------------------------------------------------------------------
    # Latency percentiles (st.metric)
    # ------------------------------------------------------------------
    st.subheader("Pipeline Latency (ms)")
    lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(4)
    with lat_col1:
        st.metric("Pipeline p50 ms", summary["pipeline_duration_p50_ms"])
    with lat_col2:
        st.metric("Pipeline p95 ms", summary["pipeline_duration_p95_ms"])
    with lat_col3:
        st.metric("Execution p50 ms", summary["execution_latency_p50_ms"])
    with lat_col4:
        st.metric("Execution p95 ms", summary["execution_latency_p95_ms"])

    # ------------------------------------------------------------------
    # Historical trend charts — accumulate samples in session_state
    # ------------------------------------------------------------------
    snapshot_time = summary.get("snapshot_time", "")

    p50_pipeline = _accumulate_trend_sample(
        st.session_state,
        _KEY_PIPELINE_P50,
        snapshot_time,
        summary["pipeline_duration_p50_ms"],
    )
    p95_pipeline = _accumulate_trend_sample(
        st.session_state,
        _KEY_PIPELINE_P95,
        snapshot_time,
        summary["pipeline_duration_p95_ms"],
    )
    p50_exec = _accumulate_trend_sample(
        st.session_state,
        _KEY_EXEC_P50,
        snapshot_time,
        summary["execution_latency_p50_ms"],
    )
    p95_exec = _accumulate_trend_sample(
        st.session_state,
        _KEY_EXEC_P95,
        snapshot_time,
        summary["execution_latency_p95_ms"],
    )

    st.subheader("Pipeline Duration Trend")
    # Build a simple list-of-dicts for st.line_chart
    pipeline_chart_data = [
        {"p50": s50["value"], "p95": s95["value"]} for s50, s95 in zip(p50_pipeline, p95_pipeline, strict=False)
    ]
    st.line_chart(pipeline_chart_data)

    st.subheader("Execution Latency Trend")
    exec_chart_data = [{"p50": s50["value"], "p95": s95["value"]} for s50, s95 in zip(p50_exec, p95_exec, strict=False)]
    st.line_chart(exec_chart_data)

    # ------------------------------------------------------------------
    # Link to raw Prometheus endpoint
    # ------------------------------------------------------------------
    prometheus_url = f"{api_base_url.rstrip('/')}/metrics"
    st.link_button("View Prometheus /metrics endpoint", url=prometheus_url)
