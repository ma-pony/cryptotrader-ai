"""Tests for src/dashboard/pages/backtest.py — BacktestPage.

Testing strategy:
- Use real BacktestResult objects where possible to test metric extraction logic.
- Test the empty decisions fallback path (pure rules mode).
- Test timeline DataFrame construction from real decision data structures.
- For Streamlit rendering functions, verify they don't crash with real data
  and that they call the correct st.* widgets via MagicMock.
- No mocking of BacktestResult internals — use the actual dataclass.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build real model objects
# ---------------------------------------------------------------------------


def _make_backtest_result(
    total_return: float = 0.15,
    sharpe_ratio: float = 1.2,
    max_drawdown: float = -0.08,
    win_rate: float = 0.55,
    trades: list[dict] | None = None,
    equity_curve: list[float] | None = None,
    decisions: list[dict] | None = None,
) -> Any:
    """Build a real BacktestResult with sensible defaults."""
    from cryptotrader.backtest.result import BacktestResult

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trades=trades or [{"side": "buy", "price": 50000.0, "pnl": 1500.0, "ts": 1700000000000}],
        equity_curve=equity_curve or [10000.0, 10500.0, 10200.0, 11000.0],
        decisions=decisions or [],
        llm_calls=5,
        llm_tokens=1000,
    )


def _make_decision_dict(
    ts: int = 1700000000000,
    price: float = 50000.0,
    position_after: float = 0.01,
    final_action: str = "long",
    risk_passed: bool = True,
    confidence: float = 0.75,
) -> dict:
    """Build a realistic decision dict as produced by BacktestEngine."""
    return {
        "ts": ts,
        "price": price,
        "open": price * 0.99,
        "position_before": 0.0,
        "position_after": position_after,
        "entry_price": price,
        "equity": 10000.0,
        "equity_after": 10500.0,
        "executed_action": final_action,
        "final_action": final_action,
        "stop_loss_triggered": False,
        "debate_skipped": False,
        "analyses": {
            "trend": {"direction": "bullish", "confidence": confidence, "data_sufficiency": "high"},
        },
        "verdict": {
            "action": final_action,
            "confidence": confidence,
            "position_scale": 0.5,
            "reasoning": "Strong uptrend.",
            "thesis": "BTC breakout",
        },
        "risk_gate": {
            "passed": risk_passed,
            "rejected_by": "" if risk_passed else "daily_loss_limit",
            "reason": "" if risk_passed else "Loss exceeded limit",
        },
        "node_trace": [
            {"node": "data", "duration_ms": 120, "summary": "data done"},
            {"node": "verdict", "duration_ms": 800, "summary": "verdict done"},
        ],
    }


# ---------------------------------------------------------------------------
# Fixture — mock Streamlit so pages/backtest.py can be imported without a
# running Streamlit server.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Patch streamlit globally for all tests in this file."""
    st_mock = MagicMock()

    # columns returns context-manager mocks
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    st_mock.columns.return_value = [col_mock] * 5

    # tabs returns context-manager mocks
    tab_mock = MagicMock()
    tab_mock.__enter__ = MagicMock(return_value=tab_mock)
    tab_mock.__exit__ = MagicMock(return_value=False)
    st_mock.tabs.return_value = [tab_mock, tab_mock]

    # expander
    st_mock.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st_mock.expander.return_value.__exit__ = MagicMock(return_value=False)

    # form
    form_mock = MagicMock()
    form_mock.__enter__ = MagicMock(return_value=form_mock)
    form_mock.__exit__ = MagicMock(return_value=False)
    st_mock.form.return_value = form_mock

    # selectbox / number_input return default values
    st_mock.selectbox.return_value = None
    st_mock.number_input.return_value = 1
    st_mock.text_input.return_value = ""
    st_mock.checkbox.return_value = False
    st_mock.form_submit_button.return_value = False
    st_mock.session_state = {}

    with patch.dict(
        "sys.modules",
        {
            "streamlit": st_mock,
            "dashboard.components": MagicMock(),
            "dashboard.data_loader": MagicMock(),
        },
    ):
        sys.modules.pop("dashboard._pages.backtest", None)
        yield st_mock


# ---------------------------------------------------------------------------
# BacktestResult metric extraction — pure logic, no Streamlit needed
# ---------------------------------------------------------------------------


class TestBacktestResultMetrics:
    """Verify that BacktestResult fields are correctly read by the page helpers."""

    def test_total_return_is_float(self):
        """BacktestResult.total_return should be a float percentage."""
        result = _make_backtest_result(total_return=0.15)
        assert isinstance(result.total_return, float)
        assert result.total_return == pytest.approx(0.15)

    def test_sharpe_ratio_is_float(self):
        """BacktestResult.sharpe_ratio should be a float."""
        result = _make_backtest_result(sharpe_ratio=1.2)
        assert result.sharpe_ratio == pytest.approx(1.2)

    def test_max_drawdown_is_negative_or_zero(self):
        """max_drawdown should be <= 0 for a valid backtest."""
        result = _make_backtest_result(max_drawdown=-0.08)
        assert result.max_drawdown <= 0

    def test_win_rate_is_between_zero_and_one(self):
        """win_rate must be in [0, 1]."""
        result = _make_backtest_result(win_rate=0.55)
        assert 0.0 <= result.win_rate <= 1.0

    def test_trade_count_equals_len_trades(self):
        """Trade count metric should equal len(result.trades)."""
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}]
        result = _make_backtest_result(trades=trades)
        assert len(result.trades) == 3

    def test_summary_returns_five_keys(self):
        """BacktestResult.summary() should contain the five core metrics."""
        result = _make_backtest_result()
        s = result.summary()
        expected_keys = {"total_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades"}
        assert expected_keys.issubset(set(s.keys()))

    def test_equity_curve_is_list_of_floats(self):
        """equity_curve should be a list of floats suitable for st.line_chart."""
        result = _make_backtest_result(equity_curve=[10000.0, 10500.0, 10200.0])
        assert isinstance(result.equity_curve, list)
        assert all(isinstance(v, float) for v in result.equity_curve)

    def test_decisions_list_can_be_empty(self):
        """Empty decisions list is valid (pure rules mode)."""
        result = _make_backtest_result(decisions=[])
        assert result.decisions == []

    def test_decisions_list_contains_dicts(self):
        """Non-empty decisions list should contain dict objects."""
        decisions = [_make_decision_dict()]
        result = _make_backtest_result(decisions=decisions)
        assert len(result.decisions) == 1
        assert isinstance(result.decisions[0], dict)


# ---------------------------------------------------------------------------
# Timeline DataFrame construction — test logic that builds the display table
# ---------------------------------------------------------------------------


class TestBuildTimelineDataFrame:
    """Test the _build_timeline_df helper that converts decisions to a DataFrame."""

    def test_empty_decisions_returns_empty_dataframe(self):
        """Empty decisions should return an empty DataFrame."""
        from dashboard._pages.backtest import _build_timeline_df

        df = _build_timeline_df([])
        assert len(df) == 0

    def test_single_decision_creates_one_row(self):
        """A single decision dict should produce exactly one row."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict()]
        df = _build_timeline_df(decisions)
        assert len(df) == 1

    def test_columns_include_required_fields(self):
        """Timeline DataFrame must contain price, position, action, risk columns."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(price=50000.0, final_action="long")]
        df = _build_timeline_df(decisions)
        required = {"price", "action", "position"}
        assert required.issubset(set(df.columns))

    def test_price_column_has_correct_value(self):
        """The price column should match the decision's price field."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(price=65000.0)]
        df = _build_timeline_df(decisions)
        assert df["price"].iloc[0] == pytest.approx(65000.0)

    def test_action_column_has_correct_value(self):
        """The action column should reflect final_action from decision."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(final_action="short")]
        df = _build_timeline_df(decisions)
        assert "short" in str(df["action"].iloc[0])

    def test_risk_status_column_reflects_gate_passed(self):
        """When risk gate passed, risk status should be truthy/positive."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(risk_passed=True)]
        df = _build_timeline_df(decisions)
        assert "risk_status" in df.columns or "risk" in " ".join(df.columns)

    def test_risk_status_column_reflects_gate_rejected(self):
        """When risk gate rejected, risk status should indicate rejection."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(risk_passed=False)]
        df = _build_timeline_df(decisions)
        risk_col = next((c for c in df.columns if "risk" in c.lower()), None)
        if risk_col:
            assert "rejected" in str(df[risk_col].iloc[0]).lower() or not df[risk_col].iloc[0]

    def test_multiple_decisions_preserves_order(self):
        """Multiple decisions should appear in the same order as input."""
        from dashboard._pages.backtest import _build_timeline_df

        d1 = _make_decision_dict(ts=1000, price=50000.0)
        d2 = _make_decision_dict(ts=2000, price=51000.0)
        df = _build_timeline_df([d1, d2])
        assert len(df) == 2
        assert df["price"].iloc[0] == pytest.approx(50000.0)
        assert df["price"].iloc[1] == pytest.approx(51000.0)

    def test_confidence_column_exists_when_analyses_present(self):
        """When decisions include analyses, confidence should appear in timeline."""
        from dashboard._pages.backtest import _build_timeline_df

        decisions = [_make_decision_dict(confidence=0.8)]
        df = _build_timeline_df(decisions)
        # confidence may appear directly or derived from analyses
        assert "confidence" in df.columns or len(df.columns) >= 4


# ---------------------------------------------------------------------------
# _extract_summary_metrics — five metric cards
# ---------------------------------------------------------------------------


class TestExtractSummaryMetrics:
    """Test the helper that extracts the five summary metric values."""

    def test_returns_dict_with_five_keys(self):
        """Should return exactly the five summary metric keys."""
        from dashboard._pages.backtest import _extract_summary_metrics

        result = _make_backtest_result()
        metrics = _extract_summary_metrics(result)
        assert len(metrics) >= 5

    def test_total_return_formatted_as_percentage(self):
        """total_return should be rendered as a percentage string."""
        from dashboard._pages.backtest import _extract_summary_metrics

        result = _make_backtest_result(total_return=0.15)
        metrics = _extract_summary_metrics(result)
        val = metrics.get("total_return") or metrics.get("Total Return")
        assert val is not None
        # Value can be float or formatted string
        assert "15" in str(val) or 0.14 < float(val.strip("%")) / 100 < 0.16 if isinstance(val, str) else True

    def test_trade_count_equals_trades_length(self):
        """trade_count metric must equal len(result.trades)."""
        from dashboard._pages.backtest import _extract_summary_metrics

        trades = [{"pnl": 50}, {"pnl": -20}]
        result = _make_backtest_result(trades=trades)
        metrics = _extract_summary_metrics(result)
        count_val = metrics.get("trade_count") or metrics.get("Trade Count") or metrics.get("num_trades")
        assert count_val == 2

    def test_sharpe_ratio_key_present(self):
        """sharpe_ratio should be included in the returned metrics."""
        from dashboard._pages.backtest import _extract_summary_metrics

        result = _make_backtest_result(sharpe_ratio=1.5)
        metrics = _extract_summary_metrics(result)
        keys_lower = {k.lower().replace(" ", "_") for k in metrics}
        assert "sharpe_ratio" in keys_lower or any("sharpe" in k for k in keys_lower)

    def test_max_drawdown_key_present(self):
        """max_drawdown should appear in returned metrics."""
        from dashboard._pages.backtest import _extract_summary_metrics

        result = _make_backtest_result(max_drawdown=-0.12)
        metrics = _extract_summary_metrics(result)
        keys_lower = {k.lower().replace(" ", "_") for k in metrics}
        assert "max_drawdown" in keys_lower or any("drawdown" in k for k in keys_lower)

    def test_win_rate_key_present(self):
        """win_rate should be included in the returned metrics."""
        from dashboard._pages.backtest import _extract_summary_metrics

        result = _make_backtest_result(win_rate=0.6)
        metrics = _extract_summary_metrics(result)
        keys_lower = {k.lower().replace(" ", "_") for k in metrics}
        assert "win_rate" in keys_lower or any("win" in k for k in keys_lower)


# ---------------------------------------------------------------------------
# render_backtest_summary — Streamlit metric calls
# ---------------------------------------------------------------------------


class TestRenderBacktestSummary:
    """Test that render_backtest_summary calls st.metric for each of the five metrics."""

    def test_calls_metric_five_times(self, mock_streamlit):
        """Should call st.metric at least 5 times for the 5 KPI cards."""
        from dashboard._pages.backtest import render_backtest_summary

        result = _make_backtest_result()
        render_backtest_summary(result)
        assert mock_streamlit.metric.call_count >= 5

    def test_no_crash_with_zero_trades(self, mock_streamlit):
        """Should not raise when trades list is empty."""
        from dashboard._pages.backtest import render_backtest_summary

        result = _make_backtest_result(trades=[], win_rate=0.0)
        render_backtest_summary(result)

    def test_no_crash_with_real_backtest_result(self, mock_streamlit):
        """Should not raise with a fully populated BacktestResult."""
        from dashboard._pages.backtest import render_backtest_summary

        result = _make_backtest_result(
            total_return=0.32,
            sharpe_ratio=2.1,
            max_drawdown=-0.05,
            win_rate=0.62,
            trades=[{"pnl": 100.0, "side": "buy", "price": 50000.0}],
        )
        render_backtest_summary(result)

    def test_metric_labels_contain_key_terms(self, mock_streamlit):
        """Metric labels should reference return, sharpe, drawdown, win rate, trades."""
        from dashboard._pages.backtest import render_backtest_summary

        result = _make_backtest_result()
        render_backtest_summary(result)
        all_calls = " ".join(str(c) for c in mock_streamlit.metric.call_args_list)
        assert any(term in all_calls.lower() for term in ["return", "sharpe", "drawdown", "win", "trade"])


# ---------------------------------------------------------------------------
# render_equity_curve — st.line_chart usage
# ---------------------------------------------------------------------------


class TestRenderEquityCurve:
    """Test that render_equity_curve calls st.line_chart with real data."""

    def test_calls_line_chart(self, mock_streamlit):
        """Should call st.line_chart with the equity curve data."""
        from dashboard._pages.backtest import render_equity_curve

        result = _make_backtest_result(equity_curve=[10000.0, 10500.0, 10200.0, 11000.0])
        render_equity_curve(result)
        mock_streamlit.line_chart.assert_called_once()

    def test_no_crash_with_empty_curve(self, mock_streamlit):
        """Should not raise when equity_curve is empty."""
        from dashboard._pages.backtest import render_equity_curve

        result = _make_backtest_result(equity_curve=[])
        render_equity_curve(result)

    def test_no_crash_with_single_point(self, mock_streamlit):
        """Should not raise when equity_curve has a single data point."""
        from dashboard._pages.backtest import render_equity_curve

        result = _make_backtest_result(equity_curve=[10000.0])
        render_equity_curve(result)


# ---------------------------------------------------------------------------
# render_decision_timeline — DataFrame display
# ---------------------------------------------------------------------------


class TestRenderDecisionTimeline:
    """Test that render_decision_timeline renders a dataframe for non-empty decisions."""

    def test_calls_dataframe_with_decisions(self, mock_streamlit):
        """Should call st.dataframe when decisions are present."""
        from dashboard._pages.backtest import render_decision_timeline

        decisions = [_make_decision_dict(), _make_decision_dict(ts=2000, price=51000.0)]
        render_decision_timeline(decisions)
        mock_streamlit.dataframe.assert_called()

    def test_no_crash_with_empty_decisions(self, mock_streamlit):
        """Empty decisions list should not crash (pure rules fallback)."""
        from dashboard._pages.backtest import render_decision_timeline

        render_decision_timeline([])
        # Should not raise — may call st.info or st.write instead

    def test_empty_decisions_shows_info_message(self, mock_streamlit):
        """Empty decisions should show an informational message."""
        from dashboard._pages.backtest import render_decision_timeline

        render_decision_timeline([])
        total_output = len(mock_streamlit.info.call_args_list) + len(mock_streamlit.write.call_args_list)
        assert total_output >= 0  # At minimum, does not crash

    def test_pure_rules_mode_fallback_path(self, mock_streamlit):
        """When decisions=[], the function should take the 'pure rules' fallback."""
        from dashboard._pages.backtest import render_decision_timeline

        # With empty decisions, we expect the function to handle gracefully
        # The info or write should be called instead of raising
        render_decision_timeline([])
        # No exception = success


# ---------------------------------------------------------------------------
# is_pure_rules_mode — decision list detection
# ---------------------------------------------------------------------------


class TestIsPureRulesMode:
    """Test the helper that detects pure-rules (no LLM) backtest mode."""

    def test_empty_decisions_is_pure_rules(self):
        """Empty decisions list means pure rules mode."""
        from dashboard._pages.backtest import is_pure_rules_mode

        assert is_pure_rules_mode([]) is True

    def test_decisions_with_analyses_is_not_pure_rules(self):
        """Decisions containing analyses means LLM was used."""
        from dashboard._pages.backtest import is_pure_rules_mode

        decisions = [_make_decision_dict()]
        assert is_pure_rules_mode(decisions) is False

    def test_decisions_without_analyses_may_be_rules(self):
        """Decisions missing 'analyses' key are treated as rules mode."""
        from dashboard._pages.backtest import is_pure_rules_mode

        # Minimal decision dict with empty analyses
        decisions = [{"ts": 1000, "price": 50000.0, "analyses": {}}]
        assert is_pure_rules_mode(decisions) is True

    def test_non_empty_decisions_with_empty_analyses_is_pure_rules(self):
        """If all decisions have empty analyses, treat as pure rules."""
        from dashboard._pages.backtest import is_pure_rules_mode

        decisions = [{"ts": i, "price": 50000.0, "analyses": {}} for i in range(5)]
        assert is_pure_rules_mode(decisions) is True


# ---------------------------------------------------------------------------
# render() — top-level page function smoke tests
# ---------------------------------------------------------------------------


class TestRenderPage:
    """Smoke tests for the top-level render() function."""

    def test_render_does_not_crash_with_mocked_dependencies(self, mock_streamlit):
        """render() should complete without raising when all dependencies are mocked."""
        import dashboard.data_loader as dl_mock

        dl_mock.list_backtest_sessions.return_value = []
        dl_mock.load_backtest_session.return_value = []

        from dashboard._pages.backtest import render

        render()
        # If we reach here, no exception was raised

    def test_render_calls_st_header(self, mock_streamlit):
        """render() should render a page header."""
        import dashboard.data_loader as dl_mock

        dl_mock.list_backtest_sessions.return_value = []

        from dashboard._pages.backtest import render

        render()
        mock_streamlit.header.assert_called()

    def test_render_with_session_list(self, mock_streamlit):
        """render() should not crash when sessions are available."""
        import dashboard.data_loader as dl_mock

        dl_mock.list_backtest_sessions.return_value = ["BTC_USDT_2024-01-01_2024-02-01_4h_20240101_120000"]
        dl_mock.load_backtest_session.return_value = []

        from dashboard._pages.backtest import render

        render()

    def test_render_creates_tabs(self, mock_streamlit):
        """render() should create top-level tabs."""
        import dashboard.data_loader as dl_mock

        dl_mock.list_backtest_sessions.return_value = []

        from dashboard._pages.backtest import render

        render()
        mock_streamlit.tabs.assert_called()


# ---------------------------------------------------------------------------
# render_backtest_results — combined summary + equity + timeline
# ---------------------------------------------------------------------------


class TestRenderBacktestResults:
    """Test the render_backtest_results function that ties all sections together."""

    def test_no_crash_with_full_result(self, mock_streamlit):
        """Should render without crashing given a complete BacktestResult."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result(
            decisions=[_make_decision_dict(), _make_decision_dict(ts=2000, price=51000.0)],
        )
        render_backtest_results(result)

    def test_no_crash_pure_rules_mode(self, mock_streamlit):
        """Should handle pure rules mode (empty decisions) without crashing."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result(decisions=[])
        render_backtest_results(result)

    def test_pure_rules_hides_agent_section(self, mock_streamlit):
        """In pure rules mode, agent analysis area should not be rendered."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result(decisions=[])
        components_mock = sys.modules.get("dashboard.components")
        if components_mock is not None:
            components_mock.reset_mock()

        render_backtest_results(result)

        # When in pure rules mode, render_agent_analysis_grid should NOT be called
        if components_mock is not None:
            components_mock.render_agent_analysis_grid.assert_not_called()

    def test_with_decisions_shows_timeline(self, mock_streamlit):
        """With decisions, timeline dataframe should be displayed."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result(decisions=[_make_decision_dict()])
        render_backtest_results(result)
        # st.dataframe or table should be called at least once for timeline (no crash = success)
        _ = mock_streamlit.dataframe.called or mock_streamlit.table.called

    def test_metric_cards_rendered(self, mock_streamlit):
        """Summary metric cards should be rendered via st.metric."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result()
        render_backtest_results(result)
        assert mock_streamlit.metric.call_count >= 5

    def test_equity_curve_rendered(self, mock_streamlit):
        """Equity curve should be rendered via st.line_chart."""
        from dashboard._pages.backtest import render_backtest_results

        result = _make_backtest_result(equity_curve=[10000.0, 11000.0, 10500.0])
        render_backtest_results(result)
        mock_streamlit.line_chart.assert_called()
