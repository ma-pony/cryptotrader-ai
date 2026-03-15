"""Tests for tracing.py helper functions — covering the summarization and
merge utilities that are not exercised by test_tracing.py or test_node_logger.py.

Covers:
- _build_trace_entry()
- _summarize_node_output()
- _summarize_analyses() / _summarize_verdict() / _summarize_risk_gate()
- _summarize_snapshot() / _summarize_regime()
- _merge_update()
- add_timing_to_trace()
"""

from __future__ import annotations

from cryptotrader.tracing import (
    _build_trace_entry,
    _merge_update,
    _summarize_analyses,
    _summarize_node_output,
    _summarize_regime,
    _summarize_risk_gate,
    _summarize_snapshot,
    _summarize_verdict,
    add_timing_to_trace,
)

# ---------------------------------------------------------------------------
# _build_trace_entry
# ---------------------------------------------------------------------------


class TestBuildTraceEntry:
    def test_returns_dict_with_required_keys(self) -> None:
        entry = _build_trace_entry("debate_gate", {"data": {"debate_skipped": True}})
        assert "node" in entry
        assert "output" in entry
        assert "summary" in entry
        assert "duration_ms" in entry
        assert "ts" in entry

    def test_node_name_is_preserved(self) -> None:
        entry = _build_trace_entry("make_verdict", {})
        assert entry["node"] == "make_verdict"

    def test_output_is_dict_when_update_is_dict(self) -> None:
        update = {"data": {"verdict": {"action": "long"}}}
        entry = _build_trace_entry("make_verdict", update)
        assert entry["output"] == update

    def test_output_is_empty_dict_when_update_is_not_dict(self) -> None:
        entry = _build_trace_entry("some_node", "not-a-dict")
        assert entry["output"] == {}

    def test_duration_ms_is_zero_initially(self) -> None:
        entry = _build_trace_entry("data_collect", {})
        assert entry["duration_ms"] == 0

    def test_ts_is_positive_float(self) -> None:
        import time

        before = time.time()
        entry = _build_trace_entry("data_collect", {})
        after = time.time()
        assert before <= entry["ts"] <= after


# ---------------------------------------------------------------------------
# _summarize_node_output — debates, analyses, verdict, risk_gate, snapshot, regime
# ---------------------------------------------------------------------------


class TestSummarizeNodeOutput:
    def test_non_dict_update_returns_str_slice(self) -> None:
        result = _summarize_node_output("node", "hello world")
        assert result == "hello world"

    def test_non_dict_update_truncated_to_200_chars(self) -> None:
        long_str = "x" * 300
        result = _summarize_node_output("node", long_str)
        assert len(result) == 200

    def test_debate_skipped_true_returns_skipped(self) -> None:
        update = {"data": {"debate_skipped": True, "debate_skip_reason": "consensus"}}
        result = _summarize_node_output("debate_gate", update)
        assert "SKIPPED" in result
        assert "consensus" in result

    def test_debate_skipped_false_returns_debate(self) -> None:
        update = {"data": {"debate_skipped": False, "debate_skip_reason": ""}}
        result = _summarize_node_output("debate_gate", update)
        assert "DEBATE" in result

    def test_analyses_in_data_uses_analyses_handler(self) -> None:
        update = {"data": {"analyses": {"tech_agent": {"direction": "bullish", "confidence": 0.9}}}}
        result = _summarize_node_output("agents", update)
        assert "tech_agent" in result
        assert "bullish" in result

    def test_verdict_in_data_uses_verdict_handler(self) -> None:
        verdict = {"action": "long", "confidence": 0.8, "position_scale": 0.5, "thesis": "uptrend"}
        update = {"data": {"verdict": verdict}}
        result = _summarize_node_output("make_verdict", update)
        assert "long" in result

    def test_risk_gate_passed_returns_passed(self) -> None:
        update = {"data": {"risk_gate": {"passed": True}}}
        result = _summarize_node_output("risk_check", update)
        assert "PASSED" in result

    def test_risk_gate_failed_returns_rejected(self) -> None:
        update = {"data": {"risk_gate": {"passed": False, "rejected_by": "daily_loss", "reason": "exceeded"}}}
        result = _summarize_node_output("risk_check", update)
        assert "REJECTED" in result
        assert "daily_loss" in result

    def test_snapshot_summary_uses_snapshot_handler(self) -> None:
        update = {"data": {"snapshot_summary": {"price": 50000, "volatility": 0.025}}}
        result = _summarize_node_output("data_collect", update)
        assert "price" in result or "50,000" in result

    def test_regime_tags_uses_regime_handler(self) -> None:
        update = {"data": {"regime_tags": ["bull_trend", "low_volatility"]}}
        result = _summarize_node_output("verbal_reinforcement", update)
        assert "regime" in result

    def test_empty_data_returns_empty(self) -> None:
        update = {"data": {}}
        result = _summarize_node_output("some_node", update)
        assert "(empty)" in result

    def test_unknown_keys_returns_keys_preview(self) -> None:
        update = {"data": {"some_key": "some_value"}}
        result = _summarize_node_output("some_node", update)
        # Should show keys list since no known handler matched
        assert "some_key" in result or "keys=" in result


# ---------------------------------------------------------------------------
# _summarize_analyses
# ---------------------------------------------------------------------------


class TestSummarizeAnalyses:
    def test_single_agent_formats_correctly(self) -> None:
        analyses = {"tech_agent": {"direction": "bullish", "confidence": 0.85}}
        result = _summarize_analyses(analyses)
        assert "tech_agent" in result
        assert "bullish" in result
        assert "85%" in result

    def test_multiple_agents_joined_by_pipe(self) -> None:
        analyses = {
            "tech_agent": {"direction": "bullish", "confidence": 0.9},
            "chain_agent": {"direction": "bearish", "confidence": 0.7},
        }
        result = _summarize_analyses(analyses)
        assert "|" in result
        assert "tech_agent" in result
        assert "chain_agent" in result

    def test_non_dict_analysis_skipped(self) -> None:
        analyses = {"tech_agent": "not-a-dict"}
        result = _summarize_analyses(analyses)
        # Non-dict analyses are skipped silently
        assert result == ""

    def test_missing_direction_shows_question_mark(self) -> None:
        analyses = {"macro_agent": {"confidence": 0.5}}
        result = _summarize_analyses(analyses)
        assert "?" in result

    def test_missing_confidence_shows_zero_percent(self) -> None:
        analyses = {"news_agent": {"direction": "neutral"}}
        result = _summarize_analyses(analyses)
        assert "0%" in result


# ---------------------------------------------------------------------------
# _summarize_verdict
# ---------------------------------------------------------------------------


class TestSummarizeVerdict:
    def test_formats_action_confidence_scale(self) -> None:
        verdict = {"action": "long", "confidence": 0.8, "position_scale": 0.5, "thesis": ""}
        result = _summarize_verdict(verdict)
        assert "long" in result
        assert "80%" in result
        assert "50%" in result

    def test_thesis_truncated_to_100_chars(self) -> None:
        long_thesis = "x" * 200
        verdict = {"action": "hold", "confidence": 0.5, "position_scale": 0.0, "thesis": long_thesis}
        result = _summarize_verdict(verdict)
        # Thesis in result should not exceed 100 chars (plus surrounding format)
        assert long_thesis not in result
        assert "x" * 100 in result

    def test_missing_fields_default_gracefully(self) -> None:
        verdict = {}
        result = _summarize_verdict(verdict)
        assert "?" in result  # action default
        assert "0%" in result  # confidence default


# ---------------------------------------------------------------------------
# _summarize_risk_gate
# ---------------------------------------------------------------------------


class TestSummarizeRiskGate:
    def test_passed_true_returns_passed(self) -> None:
        result = _summarize_risk_gate({"passed": True})
        assert result == "PASSED"

    def test_passed_false_returns_rejected_with_details(self) -> None:
        rg = {"passed": False, "rejected_by": "volatility", "reason": "too high"}
        result = _summarize_risk_gate(rg)
        assert "REJECTED" in result
        assert "volatility" in result
        assert "too high" in result

    def test_missing_passed_defaults_to_passed(self) -> None:
        result = _summarize_risk_gate({})
        assert result == "PASSED"


# ---------------------------------------------------------------------------
# _summarize_snapshot
# ---------------------------------------------------------------------------


class TestSummarizeSnapshot:
    def test_formats_price_and_volatility(self) -> None:
        snapshot = {"price": 50000, "volatility": 0.025}
        result = _summarize_snapshot(snapshot)
        assert "price" in result or "50,000" in result
        assert "vol" in result

    def test_missing_fields_use_zero_defaults(self) -> None:
        result = _summarize_snapshot({})
        assert "0" in result


# ---------------------------------------------------------------------------
# _summarize_regime
# ---------------------------------------------------------------------------


class TestSummarizeRegime:
    def test_formats_tags_list(self) -> None:
        result = _summarize_regime(["bull_trend", "low_volatility"])
        assert "bull_trend" in result
        assert "low_volatility" in result

    def test_empty_list(self) -> None:
        result = _summarize_regime([])
        assert "regime" in result


# ---------------------------------------------------------------------------
# _merge_update
# ---------------------------------------------------------------------------


class TestMergeUpdate:
    def test_simple_key_merge(self) -> None:
        state: dict = {"a": 1}
        _merge_update(state, {"b": 2})
        assert state == {"a": 1, "b": 2}

    def test_nested_dict_deep_merge(self) -> None:
        state: dict = {"data": {"x": 1, "y": 2}}
        _merge_update(state, {"data": {"y": 99, "z": 3}})
        assert state["data"] == {"x": 1, "y": 99, "z": 3}

    def test_non_dict_value_overwrites(self) -> None:
        state: dict = {"a": {"nested": 1}}
        _merge_update(state, {"a": "string_value"})
        assert state["a"] == "string_value"

    def test_empty_update_does_nothing(self) -> None:
        state: dict = {"a": 1}
        _merge_update(state, {})
        assert state == {"a": 1}

    def test_multiple_levels_deep(self) -> None:
        state: dict = {"data": {"analyses": {"tech": {"direction": "bullish"}}}}
        _merge_update(state, {"data": {"analyses": {"tech": {"confidence": 0.9}}}})
        assert state["data"]["analyses"]["tech"]["direction"] == "bullish"
        assert state["data"]["analyses"]["tech"]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# add_timing_to_trace
# ---------------------------------------------------------------------------


class TestAddTimingToTrace:
    def test_empty_trace_does_nothing(self) -> None:
        trace: list = []
        add_timing_to_trace(trace)
        assert trace == []

    def test_single_entry_duration_is_zero(self) -> None:
        trace = [{"node": "a", "ts": 1000.0, "duration_ms": 0}]
        add_timing_to_trace(trace)
        assert trace[0]["duration_ms"] == 0

    def test_two_entries_duration_computed(self) -> None:
        trace = [
            {"node": "a", "ts": 1000.000, "duration_ms": 0},
            {"node": "b", "ts": 1000.500, "duration_ms": 0},
        ]
        add_timing_to_trace(trace)
        assert trace[0]["duration_ms"] == 0
        assert trace[1]["duration_ms"] == 500  # 0.5s = 500ms

    def test_three_entries_durations_computed(self) -> None:
        trace = [
            {"node": "a", "ts": 0.0, "duration_ms": 0},
            {"node": "b", "ts": 1.0, "duration_ms": 0},
            {"node": "c", "ts": 1.25, "duration_ms": 0},
        ]
        add_timing_to_trace(trace)
        assert trace[0]["duration_ms"] == 0
        assert trace[1]["duration_ms"] == 1000  # 1.0s
        assert trace[2]["duration_ms"] == 250  # 0.25s
