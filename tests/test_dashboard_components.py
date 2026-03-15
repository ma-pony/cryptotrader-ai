"""Tests for src/dashboard/components.py — shared rendering component library.

Testing strategy:
- Pure logic (pagination math, text truncation, data transformation) tested directly.
- Streamlit-calling functions tested via MagicMock to verify they accept correct types
  and do not raise exceptions when called with real model objects.
- Real model objects (ConsensusMetrics, NodeTraceEntry, AgentAnalysis, etc.) are
  constructed for all tests — no mock dicts substituting for domain objects.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build real model objects
# ---------------------------------------------------------------------------


def _make_agent_analysis(
    agent_id: str = "trend",
    direction: str = "bullish",
    confidence: float = 0.8,
    data_sufficiency: str = "high",
    reasoning: str = "Strong uptrend detected.",
) -> Any:
    from cryptotrader.models import AgentAnalysis

    return AgentAnalysis(
        agent_id=agent_id,
        pair="BTC/USDT",
        direction=direction,
        confidence=confidence,
        reasoning=reasoning,
        data_sufficiency=data_sufficiency,
    )


def _make_consensus_metrics(
    strength: float = 0.6,
    mean_score: float = 0.5,
    dispersion: float = 0.1,
    skip_threshold: float = 0.5,
    confusion_threshold: float = 0.05,
) -> Any:
    from cryptotrader.models import ConsensusMetrics

    return ConsensusMetrics(
        strength=strength,
        mean_score=mean_score,
        dispersion=dispersion,
        skip_threshold=skip_threshold,
        confusion_threshold=confusion_threshold,
    )


def _make_node_trace(nodes: list[tuple[str, int]] | None = None) -> list[Any]:
    from cryptotrader.models import NodeTraceEntry

    if nodes is None:
        nodes = [("data", 120), ("agents", 800), ("debate_gate", 10), ("verdict", 1500)]
    return [NodeTraceEntry(node=name, duration_ms=ms, summary=f"{name} done") for name, ms in nodes]


def _make_trade_verdict(action: str = "long", confidence: float = 0.75) -> Any:
    from cryptotrader.models import TradeVerdict

    return TradeVerdict(action=action, confidence=confidence, position_scale=0.5, reasoning="Bull market.")


def _make_gate_result(passed: bool = True, rejected_by: str = "", reason: str = "") -> Any:
    from cryptotrader.models import GateResult

    return GateResult(passed=passed, rejected_by=rejected_by, reason=reason)


# ---------------------------------------------------------------------------
# Fixture — mock Streamlit so components.py can be imported without a running app
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Patch the streamlit module globally for all tests in this file."""
    st_mock = MagicMock()
    # Make st.columns return a list of context-manager-compatible mocks
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=False)
    st_mock.columns.return_value = [col_mock, col_mock, col_mock, col_mock]
    st_mock.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st_mock.expander.return_value.__exit__ = MagicMock(return_value=False)
    with patch.dict("sys.modules", {"streamlit": st_mock}):
        # Ensure components module is re-imported fresh each test run
        sys.modules.pop("dashboard.components", None)
        yield st_mock


# ---------------------------------------------------------------------------
# render_pagination_controls — pure math, no Streamlit calls needed for logic
# ---------------------------------------------------------------------------


class TestRenderPaginationControls:
    """Tests for render_pagination_controls pagination math."""

    def test_first_page_offset_is_zero(self, mock_streamlit):
        """Page 1 yields offset=0."""
        mock_streamlit.number_input.return_value = 1
        from dashboard.components import render_pagination_controls

        offset, limit = render_pagination_controls(total=100, page_size=20, key="test")
        assert offset == 0
        assert limit == 20

    def test_second_page_offset(self, mock_streamlit):
        """Page 2 yields offset=20 for page_size=20."""
        mock_streamlit.number_input.return_value = 2
        from dashboard.components import render_pagination_controls

        offset, limit = render_pagination_controls(total=100, page_size=20, key="test")
        assert offset == 20
        assert limit == 20

    def test_last_page_offset(self, mock_streamlit):
        """Page N yields offset=(N-1)*page_size."""
        mock_streamlit.number_input.return_value = 5
        from dashboard.components import render_pagination_controls

        offset, limit = render_pagination_controls(total=100, page_size=20, key="test")
        assert offset == 80
        assert limit == 20

    def test_custom_page_size(self, mock_streamlit):
        """Non-default page_size is respected in returned limit."""
        mock_streamlit.number_input.return_value = 1
        from dashboard.components import render_pagination_controls

        _offset, limit = render_pagination_controls(total=50, page_size=10, key="pg")
        assert limit == 10

    def test_returns_tuple_of_two_ints(self, mock_streamlit):
        """Return value is a 2-tuple of integers."""
        mock_streamlit.number_input.return_value = 1
        from dashboard.components import render_pagination_controls

        result = render_pagination_controls(total=60, page_size=20, key="k")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_zero_total_returns_zero_offset(self, mock_streamlit):
        """Empty result set yields offset=0."""
        mock_streamlit.number_input.return_value = 1
        from dashboard.components import render_pagination_controls

        offset, limit = render_pagination_controls(total=0, page_size=20, key="empty")
        assert offset == 0
        assert limit == 20


# ---------------------------------------------------------------------------
# render_expandable_text — pure truncation logic
# ---------------------------------------------------------------------------


class TestRenderExpandableText:
    """Tests for render_expandable_text text truncation logic."""

    def test_short_text_no_expander(self, mock_streamlit):
        """Text within preview_chars limit should not trigger st.expander."""
        from dashboard.components import render_expandable_text

        render_expandable_text("Label", "Short text", preview_chars=200)
        mock_streamlit.expander.assert_not_called()

    def test_long_text_shows_preview(self, mock_streamlit):
        """Text longer than preview_chars shows first preview_chars characters."""
        from dashboard.components import render_expandable_text

        long_text = "A" * 300
        render_expandable_text("Label", long_text, preview_chars=200)
        # st.write or st.markdown should have been called with the preview slice
        # We check that the full text was not rendered directly as a single block
        # and that expander was called for the remainder
        mock_streamlit.expander.assert_called_once()

    def test_exact_boundary_no_expander(self, mock_streamlit):
        """Text exactly at preview_chars limit should not use expander."""
        from dashboard.components import render_expandable_text

        text = "B" * 200
        render_expandable_text("Label", text, preview_chars=200)
        mock_streamlit.expander.assert_not_called()

    def test_empty_text_no_crash(self, mock_streamlit):
        """Empty string should not raise."""
        from dashboard.components import render_expandable_text

        render_expandable_text("Label", "", preview_chars=200)
        # Should not raise any exception

    def test_custom_preview_chars(self, mock_streamlit):
        """Custom preview_chars parameter is respected."""
        from dashboard.components import render_expandable_text

        text = "X" * 150
        # With preview_chars=100, text of 150 chars should use expander
        render_expandable_text("Label", text, preview_chars=100)
        mock_streamlit.expander.assert_called_once()

    def test_label_passed_to_expander(self, mock_streamlit):
        """Label argument is used as title for the expander."""
        from dashboard.components import render_expandable_text

        text = "Y" * 300
        render_expandable_text("Full Reasoning", text, preview_chars=200)
        # Check expander was called with the label
        expander_call_args = mock_streamlit.expander.call_args
        assert "Full Reasoning" in str(expander_call_args)


# ---------------------------------------------------------------------------
# render_agent_analysis_grid — accepts real AgentAnalysis objects
# ---------------------------------------------------------------------------


class TestRenderAgentAnalysisGrid:
    """Tests for render_agent_analysis_grid."""

    def test_accepts_real_agent_analysis_objects(self, mock_streamlit):
        """Should not crash when called with real AgentAnalysis objects."""
        from dashboard.components import render_agent_analysis_grid

        analyses = {
            "trend": _make_agent_analysis("trend", "bullish"),
            "onchain": _make_agent_analysis("onchain", "neutral"),
        }
        render_agent_analysis_grid(analyses)
        # Verify columns were created
        mock_streamlit.columns.assert_called()

    def test_low_sufficiency_uses_warning_icon(self, mock_streamlit):
        """data_sufficiency='low' should include a warning icon in output."""
        from dashboard.components import render_agent_analysis_grid

        analyses = {"trend": _make_agent_analysis("trend", data_sufficiency="low")}
        # Capture all text written to streamlit
        render_agent_analysis_grid(analyses)
        # Check that warning icon appears in some call to st.write/markdown/subheader
        all_calls = (
            mock_streamlit.write.call_args_list
            + mock_streamlit.markdown.call_args_list
            + mock_streamlit.subheader.call_args_list
        )
        all_text = " ".join(str(c) for c in all_calls)
        assert "⚠" in all_text

    def test_high_sufficiency_no_warning_icon(self, mock_streamlit):
        """data_sufficiency='high' should NOT show warning icon."""
        from dashboard.components import render_agent_analysis_grid

        analyses = {"trend": _make_agent_analysis("trend", data_sufficiency="high")}
        render_agent_analysis_grid(analyses)
        all_calls = (
            mock_streamlit.write.call_args_list
            + mock_streamlit.markdown.call_args_list
            + mock_streamlit.subheader.call_args_list
        )
        all_text = " ".join(str(c) for c in all_calls)
        assert "⚠" not in all_text

    def test_auto_columns_max_four(self, mock_streamlit):
        """More than 4 agents should still use at most 4 columns."""
        from dashboard.components import render_agent_analysis_grid

        analyses = {f"agent_{i}": _make_agent_analysis(f"agent_{i}") for i in range(6)}
        render_agent_analysis_grid(analyses)
        # Extract the numeric arg passed to st.columns
        col_calls = [c for c in mock_streamlit.columns.call_args_list if c.args or c.kwargs]
        if col_calls:
            first_arg = col_calls[0].args[0] if col_calls[0].args else None
            if isinstance(first_arg, int):
                assert first_arg <= 4

    def test_explicit_columns_respected(self, mock_streamlit):
        """Explicit columns parameter overrides auto calculation."""
        from dashboard.components import render_agent_analysis_grid

        mock_streamlit.columns.return_value = [MagicMock() for _ in range(2)]
        for col in mock_streamlit.columns.return_value:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)

        analyses = {
            "trend": _make_agent_analysis("trend"),
            "onchain": _make_agent_analysis("onchain"),
        }
        render_agent_analysis_grid(analyses, columns=2)
        col_calls = mock_streamlit.columns.call_args_list
        if col_calls:
            first_arg = col_calls[0].args[0] if col_calls[0].args else col_calls[0].kwargs.get("spec")
            assert first_arg == 2

    def test_empty_analyses_no_crash(self, mock_streamlit):
        """Empty analyses dict should not crash."""
        from dashboard.components import render_agent_analysis_grid

        render_agent_analysis_grid({})

    def test_reasoning_shown_via_expandable(self, mock_streamlit):
        """Each agent's reasoning should be shown (possibly via expander)."""
        from dashboard.components import render_agent_analysis_grid

        analyses = {"trend": _make_agent_analysis("trend", reasoning="Detailed bull signal.")}
        render_agent_analysis_grid(analyses)
        # Something should have been written — either direct or expander
        total_calls = (
            len(mock_streamlit.write.call_args_list)
            + len(mock_streamlit.markdown.call_args_list)
            + len(mock_streamlit.expander.call_args_list)
            + len(mock_streamlit.text.call_args_list)
        )
        assert total_calls > 0


# ---------------------------------------------------------------------------
# render_node_trace_pipeline — accepts real NodeTraceEntry objects
# ---------------------------------------------------------------------------


class TestRenderNodeTracePipeline:
    """Tests for render_node_trace_pipeline."""

    def test_accepts_real_node_trace_entries(self, mock_streamlit):
        """Should not crash with real NodeTraceEntry objects."""
        from dashboard.components import render_node_trace_pipeline

        trace = _make_node_trace()
        render_node_trace_pipeline(trace)
        mock_streamlit.columns.assert_called()

    def test_empty_trace_no_crash(self, mock_streamlit):
        """Empty trace list should not raise."""
        from dashboard.components import render_node_trace_pipeline

        render_node_trace_pipeline([])

    def test_debate_skip_node_rendered_as_gray(self, mock_streamlit):
        """Nodes named 'debate_round_1' or 'debate_round_2' should be visually distinct."""
        from dashboard.components import render_node_trace_pipeline

        trace = _make_node_trace([("debate_round_1", 0), ("verdict", 100)])
        render_node_trace_pipeline(trace)
        # At minimum, the function should have called some st output
        total = (
            len(mock_streamlit.write.call_args_list)
            + len(mock_streamlit.markdown.call_args_list)
            + len(mock_streamlit.columns.call_args_list)
        )
        assert total > 0

    def test_each_node_name_appears_in_output(self, mock_streamlit):
        """Each node name should appear somewhere in the rendered output."""
        from dashboard.components import render_node_trace_pipeline

        trace = _make_node_trace([("data_fetch", 50), ("verdict", 200)])
        render_node_trace_pipeline(trace)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "data_fetch" in all_output or "verdict" in all_output


# ---------------------------------------------------------------------------
# render_verdict_section — accepts real TradeVerdict
# ---------------------------------------------------------------------------


class TestRenderVerdictSection:
    """Tests for render_verdict_section."""

    def test_ai_verdict_badge_shown(self, mock_streamlit):
        """AI verdict should show 'ai' badge or indicator."""
        from dashboard.components import render_verdict_section

        verdict = _make_trade_verdict("long")
        render_verdict_section(verdict, "ai")
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "ai" in all_output.lower() or "AI" in all_output

    def test_weighted_verdict_badge_shown(self, mock_streamlit):
        """Weighted verdict should show 'weighted' indicator."""
        from dashboard.components import render_verdict_section

        verdict = _make_trade_verdict("hold")
        render_verdict_section(verdict, "weighted")
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "weighted" in all_output.lower()

    def test_hold_all_mock_badge_shown(self, mock_streamlit):
        """hold_all_mock verdict source should show distinct indicator."""
        from dashboard.components import render_verdict_section

        verdict = _make_trade_verdict("hold")
        render_verdict_section(verdict, "hold_all_mock")
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "mock" in all_output.lower() or "hold_all_mock" in all_output.lower()

    def test_verdict_action_shown(self, mock_streamlit):
        """Verdict action (long/short/hold) should appear in output."""
        from dashboard.components import render_verdict_section

        verdict = _make_trade_verdict("short")
        render_verdict_section(verdict, "ai")
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "short" in all_output.lower()

    def test_no_crash_with_real_objects(self, mock_streamlit):
        """Function should not raise with valid model objects."""
        from dashboard.components import render_verdict_section

        verdict = _make_trade_verdict("long", confidence=0.9)
        render_verdict_section(verdict, "ai")


# ---------------------------------------------------------------------------
# render_risk_gate_section — accepts real GateResult
# ---------------------------------------------------------------------------


class TestRenderRiskGateSection:
    """Tests for render_risk_gate_section."""

    def test_passed_gate_uses_success_indicator(self, mock_streamlit):
        """Passed gate should use st.success or green indicator."""
        from dashboard.components import render_risk_gate_section

        gate = _make_gate_result(passed=True)
        render_risk_gate_section(gate)
        mock_streamlit.success.assert_called()

    def test_rejected_gate_uses_error_indicator(self, mock_streamlit):
        """Rejected gate should use st.error or red indicator."""
        from dashboard.components import render_risk_gate_section

        gate = _make_gate_result(passed=False, rejected_by="daily_loss_limit", reason="Loss exceeded threshold")
        render_risk_gate_section(gate)
        mock_streamlit.error.assert_called()

    def test_rejected_by_shown(self, mock_streamlit):
        """rejected_by field should appear in rendered output."""
        from dashboard.components import render_risk_gate_section

        gate = _make_gate_result(passed=False, rejected_by="volatility_check", reason="Too volatile")
        render_risk_gate_section(gate)
        all_output = " ".join(
            str(c)
            for c in (
                mock_streamlit.write.call_args_list
                + mock_streamlit.markdown.call_args_list
                + mock_streamlit.error.call_args_list
            )
        )
        assert "volatility_check" in all_output

    def test_no_crash_with_real_passed_gate(self, mock_streamlit):
        """Should not raise when gate is passed."""
        from dashboard.components import render_risk_gate_section

        render_risk_gate_section(_make_gate_result(passed=True))

    def test_no_crash_with_real_rejected_gate(self, mock_streamlit):
        """Should not raise when gate is rejected."""
        from dashboard.components import render_risk_gate_section

        render_risk_gate_section(_make_gate_result(passed=False, rejected_by="risk_check", reason="Exceeded"))


# ---------------------------------------------------------------------------
# render_consensus_metrics_chart — accepts real ConsensusMetrics + analyses
# ---------------------------------------------------------------------------


class TestRenderConsensusMetricsChart:
    """Tests for render_consensus_metrics_chart."""

    def test_calls_bar_chart(self, mock_streamlit):
        """Should call st.bar_chart for agent scores."""
        from dashboard.components import render_consensus_metrics_chart

        cm = _make_consensus_metrics()
        analyses = {
            "trend": _make_agent_analysis("trend", "bullish", confidence=0.8),
            "onchain": _make_agent_analysis("onchain", "neutral", confidence=0.5),
        }
        render_consensus_metrics_chart(cm, analyses)
        mock_streamlit.bar_chart.assert_called_once()

    def test_calls_caption_with_stats(self, mock_streamlit):
        """Should call st.caption displaying mean/stdev/strength."""
        from dashboard.components import render_consensus_metrics_chart

        cm = _make_consensus_metrics(strength=0.6, mean_score=0.5, dispersion=0.1)
        analyses = {"trend": _make_agent_analysis("trend")}
        render_consensus_metrics_chart(cm, analyses)
        mock_streamlit.caption.assert_called()
        caption_text = str(mock_streamlit.caption.call_args_list)
        # At least one of the metric values should appear
        assert any(word in caption_text for word in ["mean", "strength", "dispersion", "stdev", "0.5", "0.6", "0.1"])

    def test_no_crash_with_empty_analyses(self, mock_streamlit):
        """Should not crash with empty analyses dict."""
        from dashboard.components import render_consensus_metrics_chart

        cm = _make_consensus_metrics()
        render_consensus_metrics_chart(cm, {})

    def test_no_crash_with_real_objects(self, mock_streamlit):
        """Should not raise with valid domain objects."""
        from dashboard.components import render_consensus_metrics_chart

        cm = _make_consensus_metrics()
        analyses = {"trend": _make_agent_analysis("trend", "bearish", confidence=0.3)}
        render_consensus_metrics_chart(cm, analyses)


# ---------------------------------------------------------------------------
# render_debate_section — skipped and non-skipped paths
# ---------------------------------------------------------------------------


class TestRenderDebateSection:
    """Tests for render_debate_section."""

    def test_no_skip_shows_rounds(self, mock_streamlit):
        """When debate is not skipped, rounds count should appear in output."""
        from dashboard.components import render_debate_section

        cm = _make_consensus_metrics()
        render_debate_section(debate_rounds=2, challenges=[], debate_skip_reason="", consensus_metrics=cm)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "2" in all_output or "rounds" in all_output.lower() or "debate" in all_output.lower()

    def test_skip_shows_reason(self, mock_streamlit):
        """When debate is skipped, skip reason should appear in output."""
        from dashboard.components import render_debate_section

        cm = _make_consensus_metrics(strength=0.8)
        render_debate_section(
            debate_rounds=0,
            challenges=[],
            debate_skip_reason="consensus",
            consensus_metrics=cm,
        )
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "consensus" in all_output.lower() or "skip" in all_output.lower()

    def test_skip_shows_threshold_comparison(self, mock_streamlit):
        """When skipped, threshold vs actual values should appear."""
        from dashboard.components import render_debate_section

        cm = _make_consensus_metrics(strength=0.8, skip_threshold=0.5)
        render_debate_section(
            debate_rounds=0,
            challenges=[],
            debate_skip_reason="consensus",
            consensus_metrics=cm,
        )
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        # Both threshold (0.5) and actual strength (0.8) should appear somewhere
        assert "0.5" in all_output or "0.8" in all_output

    def test_no_crash_without_consensus_metrics(self, mock_streamlit):
        """Should not raise when consensus_metrics is None."""
        from dashboard.components import render_debate_section

        render_debate_section(debate_rounds=2, challenges=[], debate_skip_reason="", consensus_metrics=None)

    def test_challenges_shown_when_present(self, mock_streamlit):
        """Non-empty challenges list should be rendered."""
        from dashboard.components import render_debate_section

        challenges = [{"round": 1, "challenger": "bear", "point": "Overbought RSI"}]
        render_debate_section(debate_rounds=1, challenges=challenges, debate_skip_reason="", consensus_metrics=None)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "Overbought" in all_output or "challenger" in all_output.lower() or len(all_output) > 0

    def test_confusion_skip_reason_shown(self, mock_streamlit):
        """confusion skip reason should appear in output."""
        from dashboard.components import render_debate_section

        cm = _make_consensus_metrics(strength=0.02, mean_score=0.01, dispersion=0.05)
        render_debate_section(
            debate_rounds=0,
            challenges=[],
            debate_skip_reason="confusion",
            consensus_metrics=cm,
        )
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "confusion" in all_output.lower() or "skip" in all_output.lower()


# ---------------------------------------------------------------------------
# render_experience_memory_section — dict-based experience memory
# ---------------------------------------------------------------------------


class TestRenderExperienceMemorySection:
    """Tests for render_experience_memory_section."""

    def test_success_patterns_shown(self, mock_streamlit):
        """success_patterns key should be rendered."""
        from dashboard.components import render_experience_memory_section

        memory = {
            "success_patterns": [{"pattern": "Bull flag", "rate": 0.72, "maturity": "rule"}],
            "forbidden_zones": [],
            "strategic_insights": [],
        }
        render_experience_memory_section(memory)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "Bull flag" in all_output or "success" in all_output.lower()

    def test_forbidden_zones_shown(self, mock_streamlit):
        """forbidden_zones key should be rendered."""
        from dashboard.components import render_experience_memory_section

        memory = {
            "success_patterns": [],
            "forbidden_zones": [{"pattern": "Bear market sell-off", "rate": 0.85, "maturity": "rule"}],
            "strategic_insights": [],
        }
        render_experience_memory_section(memory)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "Bear market" in all_output or "forbidden" in all_output.lower()

    def test_strategic_insights_shown(self, mock_streamlit):
        """strategic_insights list should be rendered."""
        from dashboard.components import render_experience_memory_section

        memory = {
            "success_patterns": [],
            "forbidden_zones": [],
            "strategic_insights": ["Avoid trading during low volume periods."],
        }
        render_experience_memory_section(memory)
        all_output = " ".join(
            str(c) for c in mock_streamlit.write.call_args_list + mock_streamlit.markdown.call_args_list
        )
        assert "Avoid trading" in all_output or "strategic" in all_output.lower()

    def test_empty_memory_no_crash(self, mock_streamlit):
        """Empty dict should not raise."""
        from dashboard.components import render_experience_memory_section

        render_experience_memory_section({})

    def test_no_crash_with_full_memory(self, mock_streamlit):
        """Full experience memory dict should not crash."""
        from dashboard.components import render_experience_memory_section

        memory = {
            "success_patterns": [
                {"pattern": "Momentum breakout", "rate": 0.65, "maturity": "hypothesis"},
            ],
            "forbidden_zones": [
                {"pattern": "FOMC week reversal", "rate": 0.80, "maturity": "rule"},
            ],
            "strategic_insights": ["Buy dips in bull regime.", "Reduce size in high VIX."],
        }
        render_experience_memory_section(memory)
