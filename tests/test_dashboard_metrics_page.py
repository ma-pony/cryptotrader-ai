"""Tests for src/dashboard/pages/metrics.py — MetricsPage (task 10).

Testing strategy (minimize mocks rule):
- Trend data accumulation logic is pure Python — test directly with real dicts
  matching the MetricsSummaryResponse schema. No Streamlit needed.
- None fallback path: verify that when load_metrics_summary returns None,
  the page calls st.warning and returns early without touching other widgets.
- Key metric display: verify that the render function calls st.metric at least
  once for each latency field when a valid summary is provided.

All tests import metrics.py with Streamlit patched via sys.modules so the module
can be loaded without a running Streamlit app.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build real MetricsSummaryResponse-shaped dicts
# ---------------------------------------------------------------------------

_SNAPSHOT_TIME = "2026-03-15T10:00:00Z"


def _make_summary(
    llm_calls_total: int = 1024,
    debate_skipped_total: int = 312,
    verdict_distribution: dict[str, int] | None = None,
    risk_rejected_total: int = 45,
    risk_rejected_by_check: dict[str, int] | None = None,
    trade_executed_total: int = 153,
    pipeline_duration_p50_ms: float = 4200.0,
    pipeline_duration_p95_ms: float = 11500.0,
    execution_latency_p50_ms: float = 150.0,
    execution_latency_p95_ms: float = 800.0,
    snapshot_time: str = _SNAPSHOT_TIME,
) -> dict[str, Any]:
    """Return a dict matching the MetricsSummaryResponse schema."""
    return {
        "llm_calls_total": llm_calls_total,
        "debate_skipped_total": debate_skipped_total,
        "verdict_distribution": verdict_distribution or {"long": 88, "short": 65, "hold": 201},
        "risk_rejected_total": risk_rejected_total,
        "risk_rejected_by_check": risk_rejected_by_check or {"daily_loss_limit": 12, "volatility": 33},
        "trade_executed_total": trade_executed_total,
        "pipeline_duration_p50_ms": pipeline_duration_p50_ms,
        "pipeline_duration_p95_ms": pipeline_duration_p95_ms,
        "execution_latency_p50_ms": execution_latency_p50_ms,
        "execution_latency_p95_ms": execution_latency_p95_ms,
        "snapshot_time": snapshot_time,
    }


# ---------------------------------------------------------------------------
# Helpers — import metrics page with Streamlit mocked
# ---------------------------------------------------------------------------


def _make_col_mock() -> MagicMock:
    """Return a context-manager-compatible column mock."""
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st_mock(session_state: dict | None = None) -> MagicMock:
    """Return a MagicMock simulating the Streamlit API used by metrics.py."""
    st = MagicMock()
    # cache_data / cache_resource as pass-through decorators
    st.cache_data = lambda *a, **kw: lambda fn: fn
    st.cache_resource = lambda fn: fn
    # session_state behaves like a real dict
    st.session_state = session_state if session_state is not None else {}

    # columns() is called with an int — return that many column mocks dynamically
    def _columns(n, *args, **kwargs):
        return [_make_col_mock() for _ in range(n if isinstance(n, int) else 4)]

    st.columns.side_effect = _columns
    return st


def _import_metrics_page(
    st_mock: MagicMock | None = None,
    summary: dict[str, Any] | None = None,
) -> Any:
    """Import (or reload) dashboard._pages.metrics with Streamlit mocked.

    Also patches get_dashboard_config and load_metrics_summary so that
    render() can run without real infrastructure.
    """
    if st_mock is None:
        st_mock = _make_st_mock()
    for key in list(sys.modules.keys()):
        if "dashboard._pages.metrics" in key or "dashboard.data_loader" in key:
            del sys.modules[key]

    # Provide a minimal data_loader stub so the page can import it.
    _cfg = {"db_url": None, "redis_url": None, "api_base_url": "http://localhost:8003"}
    dl_mock = MagicMock()
    dl_mock.get_dashboard_config = lambda: _cfg
    dl_mock.load_metrics_summary = lambda *a, **kw: summary

    with patch.dict("sys.modules", {"streamlit": st_mock, "dashboard.data_loader": dl_mock}):
        import importlib

        mod = importlib.import_module("dashboard._pages.metrics")
        # Ensure the module-level references point to our stubs so render() works.
        mod.get_dashboard_config = dl_mock.get_dashboard_config
        mod.load_metrics_summary = dl_mock.load_metrics_summary
        # Allow tests to swap summary after import by re-assigning load_metrics_summary.
        mod._dl_mock = dl_mock
        return mod


# ---------------------------------------------------------------------------
# Pure Python logic: trend data accumulation
# ---------------------------------------------------------------------------


class TestAccumulateTrendSample:
    """Tests for _accumulate_trend_sample() pure function.

    This function takes a session_state dict, a snapshot_time string,
    and a float value; it appends {timestamp, value} to the list stored
    under the given key and returns the updated list.
    """

    def test_first_sample_creates_list(self):
        """First call creates a one-element list under the key."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        result = mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4200.0)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_first_sample_has_correct_fields(self):
        """First sample dict contains 'timestamp' and 'value' keys."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        result = mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4200.0)
        assert result[0]["timestamp"] == "2026-03-15T10:00:00Z"
        assert result[0]["value"] == pytest.approx(4200.0)

    def test_second_sample_appends(self):
        """Second call appends to the existing list."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4200.0)
        result = mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:01:00Z", 4300.0)
        assert len(result) == 2
        assert result[1]["value"] == pytest.approx(4300.0)

    def test_duplicate_timestamp_still_appended(self):
        """Same snapshot_time appended again (idempotency not required)."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4200.0)
        result = mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4250.0)
        assert len(result) == 2

    def test_different_keys_stored_independently(self):
        """Accumulation under separate keys does not cross-contaminate."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        mod._accumulate_trend_sample(state, "key_a", "2026-03-15T10:00:00Z", 1.0)
        mod._accumulate_trend_sample(state, "key_b", "2026-03-15T10:00:00Z", 2.0)
        assert len(state["key_a"]) == 1
        assert len(state["key_b"]) == 1
        assert state["key_a"][0]["value"] == pytest.approx(1.0)
        assert state["key_b"][0]["value"] == pytest.approx(2.0)

    def test_state_mutated_in_place(self):
        """The session_state dict is mutated in place (same reference)."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T10:00:00Z", 4200.0)
        assert "trend_p50" in state
        assert len(state["trend_p50"]) == 1

    def test_accumulates_up_to_max_samples(self):
        """When MAX_TREND_SAMPLES entries exist, oldest entry is dropped."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        max_samples = mod._MAX_TREND_SAMPLES
        # Fill to capacity
        for i in range(max_samples):
            mod._accumulate_trend_sample(state, "trend_p50", f"2026-03-15T10:{i:02d}:00Z", float(i))
        # Add one more — list should not exceed max_samples
        mod._accumulate_trend_sample(state, "trend_p50", "2026-03-15T11:00:00Z", 999.0)
        assert len(state["trend_p50"]) == max_samples

    def test_oldest_entry_dropped_when_full(self):
        """When at capacity, the oldest (first) entry is removed."""
        mod = _import_metrics_page()
        state: dict[str, Any] = {}
        max_samples = mod._MAX_TREND_SAMPLES
        # Fill to capacity with known values
        for i in range(max_samples):
            mod._accumulate_trend_sample(state, "trend_p50", f"ts_{i}", float(i))
        # The first entry has value 0.0; after adding one more it should be gone
        mod._accumulate_trend_sample(state, "trend_p50", "ts_new", 999.0)
        values = [s["value"] for s in state["trend_p50"]]
        assert 0.0 not in values
        assert 999.0 in values


# ---------------------------------------------------------------------------
# None fallback: load_metrics_summary returns None
# ---------------------------------------------------------------------------


class TestMetricsPageNoneFallback:
    """Tests for the None-fallback path when the metrics endpoint is unavailable."""

    def test_none_summary_calls_st_warning(self):
        """When summary is None, st.warning must be called."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)

        mod.render()

        st_mock.warning.assert_called()

    def test_none_summary_does_not_call_st_metric(self):
        """When summary is None, st.metric must NOT be called (early return)."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)

        mod.render()

        st_mock.metric.assert_not_called()

    def test_none_summary_warning_message_contains_endpoint_text(self):
        """The warning message should mention the metrics endpoint is unavailable."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)

        mod.render()

        all_warning_calls = str(st_mock.warning.call_args_list)
        assert "指标端点不可用" in all_warning_calls or "unavailable" in all_warning_calls.lower()

    def test_none_summary_does_not_call_line_chart(self):
        """When summary is None, no charts should be rendered."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)

        mod.render()

        st_mock.line_chart.assert_not_called()


# ---------------------------------------------------------------------------
# Valid summary: key metrics displayed via st.metric
# ---------------------------------------------------------------------------


class TestMetricsPageWithValidSummary:
    """Tests for the happy path when a valid MetricsSummaryResponse dict is provided."""

    def test_pipeline_p50_shown_via_metric(self):
        """pipeline_duration_p50_ms must be shown via st.metric."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(pipeline_duration_p50_ms=4200.0)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        metric_calls = str(st_mock.metric.call_args_list)
        # Value 4200.0 or label containing p50 should appear
        assert "4200" in metric_calls or "p50" in metric_calls.lower()

    def test_pipeline_p95_shown_via_metric(self):
        """pipeline_duration_p95_ms must be shown via st.metric."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(pipeline_duration_p95_ms=11500.0)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        metric_calls = str(st_mock.metric.call_args_list)
        assert "11500" in metric_calls or "p95" in metric_calls.lower()

    def test_execution_latency_p50_shown_via_metric(self):
        """execution_latency_p50_ms must be shown via st.metric."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(execution_latency_p50_ms=150.0)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        metric_calls = str(st_mock.metric.call_args_list)
        assert "150" in metric_calls or "p50" in metric_calls.lower()

    def test_execution_latency_p95_shown_via_metric(self):
        """execution_latency_p95_ms must be shown via st.metric."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(execution_latency_p95_ms=800.0)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        metric_calls = str(st_mock.metric.call_args_list)
        assert "800" in metric_calls or "p95" in metric_calls.lower()

    def test_llm_calls_total_shown(self):
        """llm_calls_total should appear somewhere in the rendered output."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(llm_calls_total=1024)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        all_output = " ".join(
            str(c)
            for c in (st_mock.metric.call_args_list + st_mock.write.call_args_list + st_mock.markdown.call_args_list)
        )
        assert "1024" in all_output

    def test_debate_skipped_shown(self):
        """debate_skipped_total should appear in rendered output."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(debate_skipped_total=312)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        all_output = " ".join(
            str(c)
            for c in (st_mock.metric.call_args_list + st_mock.write.call_args_list + st_mock.markdown.call_args_list)
        )
        assert "312" in all_output

    def test_risk_rejected_total_shown(self):
        """risk_rejected_total should appear in rendered output."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(risk_rejected_total=45)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        all_output = " ".join(
            str(c)
            for c in (st_mock.metric.call_args_list + st_mock.write.call_args_list + st_mock.markdown.call_args_list)
        )
        assert "45" in all_output

    def test_trade_executed_total_shown(self):
        """trade_executed_total should appear in rendered output."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(trade_executed_total=153)
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        all_output = " ".join(
            str(c)
            for c in (st_mock.metric.call_args_list + st_mock.write.call_args_list + st_mock.markdown.call_args_list)
        )
        assert "153" in all_output

    def test_link_button_called_for_prometheus_endpoint(self):
        """st.link_button should be called with the /metrics Prometheus URL."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary()
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        st_mock.link_button.assert_called()
        link_calls = str(st_mock.link_button.call_args_list)
        assert "/metrics" in link_calls or "Prometheus" in link_calls or "prometheus" in link_calls.lower()

    def test_no_crash_with_zero_values(self):
        """All-zero summary should not raise."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(
            llm_calls_total=0,
            debate_skipped_total=0,
            verdict_distribution={},
            risk_rejected_total=0,
            risk_rejected_by_check={},
            trade_executed_total=0,
            pipeline_duration_p50_ms=0.0,
            pipeline_duration_p95_ms=0.0,
            execution_latency_p50_ms=0.0,
            execution_latency_p95_ms=0.0,
        )
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()


# ---------------------------------------------------------------------------
# Historical trend chart: line_chart called after accumulation
# ---------------------------------------------------------------------------


class TestMetricsPageTrendChart:
    """Tests for the historical trend chart section."""

    def test_line_chart_called_with_valid_summary(self):
        """st.line_chart should be called when a valid summary is provided."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary()
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        st_mock.line_chart.assert_called()

    def test_trend_state_grows_across_calls(self):
        """Session state trend list grows on each call with a new snapshot_time."""
        # Use a shared mutable session_state across calls
        session_state: dict[str, Any] = {}
        st_mock = _make_st_mock(session_state=session_state)
        mod = _import_metrics_page(st_mock)

        summary1 = _make_summary(pipeline_duration_p50_ms=4000.0, snapshot_time="2026-03-15T10:00:00Z")
        mod.load_metrics_summary = lambda *a, **kw: summary1
        mod.render()

        summary2 = _make_summary(pipeline_duration_p50_ms=4100.0, snapshot_time="2026-03-15T10:01:00Z")
        # Re-import with same session_state — simulate second Streamlit rerun
        st_mock2 = _make_st_mock(session_state=session_state)
        mod.load_metrics_summary = lambda *a, **kw: summary2
        with patch.dict("sys.modules", {"streamlit": st_mock2}):
            mod.render()

        # At least one trend key should have 2 entries
        trend_keys = [k for k in session_state if isinstance(session_state[k], list)]
        assert any(len(session_state[k]) == 2 for k in trend_keys)

    def test_verdict_distribution_rendered(self):
        """verdict_distribution dict values should appear in rendered output."""
        st_mock = _make_st_mock()
        mod = _import_metrics_page(st_mock)
        summary = _make_summary(verdict_distribution={"long": 88, "short": 65, "hold": 201})
        mod.load_metrics_summary = lambda *a, **kw: summary
        mod.render()

        all_output = " ".join(
            str(c)
            for c in (
                st_mock.metric.call_args_list
                + st_mock.write.call_args_list
                + st_mock.markdown.call_args_list
                + st_mock.bar_chart.call_args_list
                + st_mock.table.call_args_list
                + st_mock.json.call_args_list
                + st_mock.dataframe.call_args_list
            )
        )
        # At least one of the verdict labels/counts should appear
        assert "88" in all_output or "65" in all_output or "201" in all_output or "long" in all_output
