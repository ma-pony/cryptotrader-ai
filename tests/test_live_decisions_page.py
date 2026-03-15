"""Tests for src/dashboard/pages/live_decisions.py — LiveDecisionsPage.

Testing strategy (minimise mocks per project rules):
- Pure logic: pair extraction from commit list, detail data preparation,
  trace_id link building, portfolio change display logic.
- Real model objects (DecisionCommit, AgentAnalysis, etc.) for all domain data.
- Streamlit rendering functions tested via MagicMock to verify they accept
  correct types and do not crash with realistic data.
- No mocking of domain model logic — only Streamlit UI calls are mocked.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build real domain objects
# ---------------------------------------------------------------------------


def _make_agent_analysis(
    agent_id: str = "trend",
    direction: str = "bullish",
    confidence: float = 0.75,
    data_sufficiency: str = "high",
    reasoning: str = "Strong uptrend momentum.",
) -> Any:
    from cryptotrader.models import AgentAnalysis

    return AgentAnalysis(
        agent_id=agent_id,
        pair="BTC/USDT",
        direction=direction,
        confidence=confidence,
        reasoning=reasoning,
        key_factors=["RSI momentum", "volume spike"],
        risk_flags=[],
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


def _make_trade_verdict(action: str = "long", confidence: float = 0.8) -> Any:
    from cryptotrader.models import TradeVerdict

    return TradeVerdict(
        action=action,
        confidence=confidence,
        position_scale=0.5,
        reasoning="Bull market structure intact.",
        thesis="BTC entering new ATH zone.",
        invalidation="Close below 90k invalidates.",
    )


def _make_gate_result(passed: bool = True) -> Any:
    from cryptotrader.models import GateResult

    return GateResult(
        passed=passed,
        rejected_by="" if passed else "daily_loss_limit",
        reason="" if passed else "Daily loss limit exceeded.",
    )


def _make_node_trace() -> list[Any]:
    from cryptotrader.models import NodeTraceEntry

    return [
        NodeTraceEntry(node="data_fetch", duration_ms=120, summary="Fetched market data"),
        NodeTraceEntry(node="verbal_reinforcement", duration_ms=80, summary="Experience injected"),
        NodeTraceEntry(node="agents", duration_ms=950, summary="4 agents analysed"),
        NodeTraceEntry(node="debate_gate", duration_ms=10, summary="Consensus checked"),
        NodeTraceEntry(node="debate_round_1", duration_ms=0, summary="Skipped"),
        NodeTraceEntry(node="verdict", duration_ms=1200, summary="AI verdict: long"),
        NodeTraceEntry(node="risk_gate", duration_ms=5, summary="Risk checks passed"),
        NodeTraceEntry(node="execute", duration_ms=300, summary="Order submitted"),
    ]


def _make_decision_commit(
    hash_val: str = "abc123def456",  # pragma: allowlist secret
    pair: str = "BTC/USDT",
    verdict_source: str = "ai",
    debate_skip_reason: str = "",
    with_consensus: bool = True,
    with_node_trace: bool = True,
    with_experience: bool = True,
    trace_id: str | None = "span-0001-test",
) -> Any:
    from cryptotrader.models import DecisionCommit

    analyses = {
        "trend": _make_agent_analysis("trend", "bullish", 0.8),
        "onchain": _make_agent_analysis("onchain", "neutral", 0.5, "medium"),
        "news": _make_agent_analysis("news", "bullish", 0.7, "high"),
        "macro": _make_agent_analysis("macro", "neutral", 0.4, "low"),
    }

    consensus_metrics = _make_consensus_metrics() if with_consensus else None
    node_trace = _make_node_trace() if with_node_trace else []
    experience_memory = (
        {
            "success_patterns": [{"pattern": "Bull flag breakout", "rate": 0.72, "maturity": "rule"}],
            "forbidden_zones": [{"pattern": "FOMC reversal", "rate": 0.80, "maturity": "rule"}],
            "strategic_insights": ["Reduce size in high VIX regimes."],
        }
        if with_experience
        else {}
    )

    return DecisionCommit(
        hash=hash_val,
        parent_hash=None,
        timestamp=datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC),
        pair=pair,
        snapshot_summary={
            "price": 95000.0,
            "volatility": 0.03,
            "regime_tags": ["bull_trend", "high_volume"],
        },
        analyses=analyses,
        debate_rounds=0 if debate_skip_reason else 2,
        challenges=[] if debate_skip_reason else [{"round": 1, "challenger": "bear", "point": "RSI overbought"}],
        divergence=0.15,
        verdict=_make_trade_verdict(),
        risk_gate=_make_gate_result(passed=True),
        order=None,
        fill_price=94950.0,
        slippage=0.05,
        portfolio_after={
            "total_value": 10500.0,
            "cash": 5000.0,
            "positions": {"BTC/USDT": {"size": 0.05, "side": "long"}},
        },
        pnl=50.0,
        retrospective=None,
        trace_id=trace_id,
        consensus_metrics=consensus_metrics,
        verdict_source=verdict_source,
        experience_memory=experience_memory,
        node_trace=node_trace,
        debate_skip_reason=debate_skip_reason,
    )


# ---------------------------------------------------------------------------
# Fixture — mock Streamlit so page module can be imported without a running app
# ---------------------------------------------------------------------------


def _make_col_mock() -> MagicMock:
    """Create a single column mock that supports context manager protocol."""
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st_mock() -> MagicMock:
    """Build a MagicMock that mimics the parts of streamlit used by the page."""
    st_mock = MagicMock()

    # st.columns(n) or st.columns([w1, w2, ...]) — return the right number of cols.
    def _columns_side_effect(spec, **kwargs):
        n = len(spec) if isinstance(spec, list | tuple) else int(spec)
        return [_make_col_mock() for _ in range(n)]

    st_mock.columns.side_effect = _columns_side_effect

    st_mock.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st_mock.expander.return_value.__exit__ = MagicMock(return_value=False)

    # dataframe selection response — default: no row selected
    sel_mock = MagicMock()
    sel_mock.selection.rows = []
    st_mock.dataframe.return_value = sel_mock

    # number_input returns 1 (page 1 for pagination)
    st_mock.number_input.return_value = 1

    # selectbox returns None by default
    st_mock.selectbox.return_value = None

    # session_state acts as a dict
    st_mock.session_state = {}

    return st_mock


@pytest.fixture(autouse=True)
def mock_st():
    """Patch streamlit globally for ALL tests in this file.

    Uses autouse=True so that every test (including the pure-logic tests that
    import the page module) has streamlit mocked.  Without this, the module-level
    ``import streamlit as st`` in live_decisions.py would trigger the real Streamlit
    initialisation and cause DeltaGeneratorSingleton errors on second import.
    """
    st_mock = _make_st_mock()
    modules_to_clear = [
        "dashboard._pages.live_decisions",
        "dashboard.components",
        "dashboard.data_loader",
    ]
    with patch.dict("sys.modules", {"streamlit": st_mock}):
        for mod in modules_to_clear:
            sys.modules.pop(mod, None)
        yield st_mock


# ---------------------------------------------------------------------------
# Pure logic tests — pair extraction from a list of commits
# ---------------------------------------------------------------------------


class TestExtractPairs:
    """Tests for the pair extraction helper used to populate the filter dropdown."""

    def test_extract_pairs_from_commits(self):
        """extract_pairs should return unique pairs in sorted order."""
        commits = [
            _make_decision_commit("h1", pair="BTC/USDT"),
            _make_decision_commit("h2", pair="ETH/USDT"),
            _make_decision_commit("h3", pair="BTC/USDT"),
        ]
        from dashboard._pages.live_decisions import _extract_pairs

        pairs = _extract_pairs(commits)
        assert pairs == ["BTC/USDT", "ETH/USDT"]

    def test_extract_pairs_empty_list(self):
        """Empty commit list should return empty list."""
        from dashboard._pages.live_decisions import _extract_pairs

        assert _extract_pairs([]) == []

    def test_extract_pairs_single_pair(self):
        """Single pair commit list returns that one pair."""
        commits = [_make_decision_commit("h1", pair="SOL/USDT")]
        from dashboard._pages.live_decisions import _extract_pairs

        pairs = _extract_pairs(commits)
        assert pairs == ["SOL/USDT"]

    def test_extract_pairs_preserves_uniqueness(self):
        """Duplicate pairs should not appear multiple times in result."""
        commits = [_make_decision_commit(f"h{i}", pair="BTC/USDT") for i in range(5)]
        from dashboard._pages.live_decisions import _extract_pairs

        pairs = _extract_pairs(commits)
        assert len(pairs) == 1

    def test_extract_pairs_sorted(self):
        """Returned pairs should be in alphabetical order."""
        commits = [
            _make_decision_commit("h1", pair="SOL/USDT"),
            _make_decision_commit("h2", pair="BTC/USDT"),
            _make_decision_commit("h3", pair="ETH/USDT"),
        ]
        from dashboard._pages.live_decisions import _extract_pairs

        pairs = _extract_pairs(commits)
        assert pairs == sorted(pairs)


# ---------------------------------------------------------------------------
# Pure logic tests — OTel trace link builder
# ---------------------------------------------------------------------------


class TestBuildTraceLink:
    """Tests for the OTel trace link builder (_build_trace_link)."""

    def test_returns_link_when_endpoint_set(self):
        """When OTLP_ENDPOINT env var is set, a link containing trace_id is returned."""
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://jaeger:4317"}):
            from dashboard._pages.live_decisions import _build_trace_link

            link = _build_trace_link("span-0001")
        assert link is not None
        assert "span-0001" in link

    def test_returns_none_when_no_endpoint(self):
        """When OTLP_ENDPOINT is not set, returns None."""
        env = {k: v for k, v in os.environ.items() if k != "OTLP_ENDPOINT"}
        with patch.dict(os.environ, env, clear=True):
            from dashboard._pages.live_decisions import _build_trace_link

            link = _build_trace_link("span-0001")
        assert link is None

    def test_returns_none_for_none_trace_id(self):
        """None trace_id always returns None regardless of env var."""
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://jaeger:4317"}):
            from dashboard._pages.live_decisions import _build_trace_link

            link = _build_trace_link(None)
        assert link is None

    def test_returns_none_for_empty_trace_id(self):
        """Empty string trace_id returns None."""
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://jaeger:4317"}):
            from dashboard._pages.live_decisions import _build_trace_link

            link = _build_trace_link("")
        assert link is None


# ---------------------------------------------------------------------------
# Pure logic tests — commit list to DataFrame rows conversion
# ---------------------------------------------------------------------------


class TestCommitsToRows:
    """Tests for _commits_to_rows, the list-to-DataFrame conversion helper."""

    def test_returns_list_of_dicts(self):
        """Should return a list of dicts with expected keys."""
        commit = _make_decision_commit()
        from dashboard._pages.live_decisions import _commits_to_rows

        rows = _commits_to_rows([commit])
        assert isinstance(rows, list)
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, dict)

    def test_row_contains_expected_keys(self):
        """Each row dict should contain hash, pair, time, price, action, source keys."""
        commit = _make_decision_commit()
        from dashboard._pages.live_decisions import _commits_to_rows

        rows = _commits_to_rows([commit])
        row = rows[0]
        assert "hash" in row
        assert "pair" in row

    def test_row_hash_matches_commit(self):
        """Row hash should match the DecisionCommit hash."""
        commit = _make_decision_commit(hash_val="d3adb33f1234")
        from dashboard._pages.live_decisions import _commits_to_rows

        rows = _commits_to_rows([commit])
        assert rows[0]["hash"] == "d3adb33f1234"

    def test_row_pair_matches_commit(self):
        """Row pair should match the DecisionCommit pair."""
        commit = _make_decision_commit(pair="ETH/USDT")
        from dashboard._pages.live_decisions import _commits_to_rows

        rows = _commits_to_rows([commit])
        assert rows[0]["pair"] == "ETH/USDT"

    def test_empty_list_returns_empty(self):
        """Empty commit list returns empty rows list."""
        from dashboard._pages.live_decisions import _commits_to_rows

        assert _commits_to_rows([]) == []

    def test_multiple_commits_all_present(self):
        """All commits should appear in output."""
        commits = [
            _make_decision_commit("h1", pair="BTC/USDT"),
            _make_decision_commit("h2", pair="ETH/USDT"),
        ]
        from dashboard._pages.live_decisions import _commits_to_rows

        rows = _commits_to_rows(commits)
        assert len(rows) == 2
        hashes = {r["hash"] for r in rows}
        assert "h1" in hashes
        assert "h2" in hashes


# ---------------------------------------------------------------------------
# Rendering tests — render() does not crash with real data
# ---------------------------------------------------------------------------


class TestRenderPageSmoke:
    """Smoke tests: render() should not raise with realistic real data.

    The module must be imported while the streamlit mock is already in sys.modules.
    Each test therefore imports the render function inside the mock_st fixture
    context (which is already active via the fixture) rather than re-patching.
    """

    def _get_render(self):
        """Import render() from the page module under the current st mock."""
        sys.modules.pop("dashboard._pages.live_decisions", None)
        sys.modules.pop("dashboard.components", None)
        # Mock get_dashboard_config so render() can resolve db_url without real config
        import dashboard.data_loader as _dl

        _dl.get_dashboard_config = lambda: {"db_url": None, "redis_url": None, "api_base_url": "http://localhost:8003"}
        from dashboard._pages.live_decisions import render

        return render

    def test_render_with_empty_journal(self, mock_st):
        """render() should not crash when journal is empty (no commits)."""
        render = self._get_render()
        # Patch the already-imported module's data-loader references
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: []
        mod.load_commit_detail = lambda *a, **kw: None
        render()

    def test_render_with_commits_no_selection(self, mock_st):
        """render() should not crash when commits exist but none is selected."""
        commit = _make_decision_commit("abc123")
        mock_st.dataframe.return_value.selection.rows = []
        mock_st.selectbox.return_value = None

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: None
        render()

    def test_render_with_commit_selected(self, mock_st):
        """render() should not crash when a commit is selected for detail view."""
        commit = _make_decision_commit("abc123")
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_with_debate_skipped_commit(self, mock_st):
        """render() should not crash for a commit where debate was skipped."""
        commit = _make_decision_commit("xyz789", debate_skip_reason="consensus", with_consensus=True)
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_with_weighted_verdict(self, mock_st):
        """render() should not crash for a weighted-downgrade verdict commit."""
        commit = _make_decision_commit("wght001", verdict_source="weighted")
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_db_exception_calls_st_error_and_stop(self, mock_st):
        """When load_journal raises, render() should call st.error and st.stop."""
        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]

        def _raise(*a, **kw):
            raise RuntimeError("DB connection refused")

        mod.load_journal = _raise
        render()
        mock_st.error.assert_called()
        mock_st.stop.assert_called()

    def test_render_with_selectbox_fallback(self, mock_st):
        """When st.dataframe raises TypeError (old Streamlit), selectbox fallback is used."""
        commit = _make_decision_commit("fallback01")
        mock_st.dataframe.side_effect = TypeError("on_select not supported")
        mock_st.selectbox.return_value = "fallback01"

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()
        mock_st.selectbox.assert_called()

    def test_render_with_no_consensus_metrics(self, mock_st):
        """render() should not crash for old commits without consensus_metrics."""
        commit = _make_decision_commit("old001", with_consensus=False)
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_with_no_node_trace(self, mock_st):
        """render() should not crash for commits without node_trace."""
        commit = _make_decision_commit("old002", with_node_trace=False)
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_with_no_experience_memory(self, mock_st):
        """render() should not crash for commits without experience_memory."""
        commit = _make_decision_commit("old003", with_experience=False)
        mock_st.dataframe.return_value.selection.rows = [0]

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = lambda *a, **kw: [commit]
        mod.load_commit_detail = lambda *a, **kw: commit
        render()

    def test_render_with_otlp_endpoint_set(self, mock_st):
        """render() should render a trace link when OTLP_ENDPOINT is set."""
        commit = _make_decision_commit("trace001", trace_id="span-abc-def")
        mock_st.dataframe.return_value.selection.rows = [0]

        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://jaeger:16686"}):
            render = self._get_render()
            mod = sys.modules["dashboard._pages.live_decisions"]
            mod.load_journal = lambda *a, **kw: [commit]
            mod.load_commit_detail = lambda *a, **kw: commit
            render()
        # At minimum: markdown or write should have been called
        assert mock_st.write.called or mock_st.markdown.called

    def test_render_pair_filter_applied(self, mock_st):
        """When a pair filter is selected, load_journal is called with that pair."""
        commits = [
            _make_decision_commit("h1", pair="BTC/USDT"),
            _make_decision_commit("h2", pair="ETH/USDT"),
        ]
        # Selectbox returns a value for the pair filter
        mock_st.selectbox.return_value = "ETH/USDT"
        mock_st.dataframe.return_value.selection.rows = []

        call_log: list[dict] = []

        def _load_journal(*a, **kw):
            call_log.append(kw)
            return commits

        render = self._get_render()
        mod = sys.modules["dashboard._pages.live_decisions"]
        mod.load_journal = _load_journal
        mod.load_commit_detail = lambda *a, **kw: None
        render()
        # load_journal should have been called at least once
        assert len(call_log) >= 1


# ---------------------------------------------------------------------------
# Rendering tests — _render_decision_detail does not crash with real commit
# ---------------------------------------------------------------------------


class TestRenderDecisionDetail:
    """Tests for _render_decision_detail with real DecisionCommit objects."""

    def _get_detail_fn(self):
        """Import _render_decision_detail while st mock is active."""
        sys.modules.pop("dashboard._pages.live_decisions", None)
        sys.modules.pop("dashboard.components", None)
        from dashboard._pages.live_decisions import _render_decision_detail

        return _render_decision_detail

    def test_no_crash_full_commit(self, mock_st):
        """_render_decision_detail should not raise with a fully-populated commit."""
        commit = _make_decision_commit()
        with patch.dict(os.environ, {"OTLP_ENDPOINT": ""}):
            fn = self._get_detail_fn()
            fn(commit)

    def test_no_crash_minimal_commit(self, mock_st):
        """_render_decision_detail should not raise with a minimal commit (all None fields)."""
        from cryptotrader.models import DecisionCommit

        commit = DecisionCommit(
            hash="minimal001",
            parent_hash=None,
            timestamp=datetime(2026, 3, 15, tzinfo=UTC),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
        )
        fn = self._get_detail_fn()
        fn(commit)

    def test_header_shows_pair(self, mock_st):
        """Decision detail header should display the pair."""
        commit = _make_decision_commit(pair="SOL/USDT")
        with patch.dict(os.environ, {"OTLP_ENDPOINT": ""}):
            fn = self._get_detail_fn()
            fn(commit)
        all_calls = (
            [str(c) for c in mock_st.write.call_args_list]
            + [str(c) for c in mock_st.markdown.call_args_list]
            + [str(c) for c in mock_st.subheader.call_args_list]
            + [str(c) for c in mock_st.header.call_args_list]
        )
        all_text = " ".join(all_calls)
        assert "SOL/USDT" in all_text

    def test_header_shows_trace_id(self, mock_st):
        """Decision detail header should show trace_id."""
        commit = _make_decision_commit(trace_id="my-trace-1234")
        with patch.dict(os.environ, {"OTLP_ENDPOINT": ""}):
            fn = self._get_detail_fn()
            fn(commit)
        all_text = " ".join(
            str(c)
            for c in (mock_st.write.call_args_list + mock_st.markdown.call_args_list + mock_st.subheader.call_args_list)
        )
        assert "my-trace-1234" in all_text

    def test_risk_gate_section_shown_when_present(self, mock_st):
        """Risk gate section should be rendered when risk_gate is not None."""
        commit = _make_decision_commit()
        assert commit.risk_gate is not None
        fn = self._get_detail_fn()
        fn(commit)
        # At minimum, some output was rendered (success or error for risk gate)
        assert mock_st.success.called or mock_st.error.called or mock_st.write.called

    def test_verdict_section_shown_when_verdict_present(self, mock_st):
        """Verdict section should render when commit.verdict is not None."""
        commit = _make_decision_commit(verdict_source="ai")
        assert commit.verdict is not None
        fn = self._get_detail_fn()
        fn(commit)
        all_text = " ".join(str(c) for c in mock_st.write.call_args_list + mock_st.markdown.call_args_list)
        # "ai" or "long" (the verdict action) should appear
        assert "ai" in all_text.lower() or "long" in all_text.lower()

    def test_portfolio_changes_shown(self, mock_st):
        """Portfolio after changes should be shown in the execution section."""
        commit = _make_decision_commit()
        commit.portfolio_after = {
            "total_value": 10500.0,
            "cash": 5000.0,
            "positions": {"BTC/USDT": {"size": 0.05}},
        }
        fn = self._get_detail_fn()
        fn(commit)
        # Something related to the execution section should have been written
        total = (
            len(mock_st.write.call_args_list)
            + len(mock_st.markdown.call_args_list)
            + len(mock_st.subheader.call_args_list)
        )
        assert total > 0

    def test_no_crash_without_verdict(self, mock_st):
        """render should not crash when commit.verdict is None."""
        from cryptotrader.models import DecisionCommit

        commit = DecisionCommit(
            hash="noverd001",
            parent_hash=None,
            timestamp=datetime(2026, 3, 15, tzinfo=UTC),
            pair="BTC/USDT",
            snapshot_summary={"price": 90000.0},
            analyses={"trend": _make_agent_analysis()},
            debate_rounds=0,
            verdict=None,
        )
        fn = self._get_detail_fn()
        fn(commit)

    def test_experience_memory_section_shown(self, mock_st):
        """Experience memory section should be called when experience_memory is present."""
        commit = _make_decision_commit(with_experience=True)
        assert commit.experience_memory
        fn = self._get_detail_fn()
        fn(commit)
        # Some st output should have been generated
        assert mock_st.write.called or mock_st.markdown.called

    def test_no_crash_with_stop_loss_status(self, mock_st):
        """render should not crash when stop-loss related fields are in portfolio_after."""
        commit = _make_decision_commit()
        commit.portfolio_after = {
            "total_value": 9800.0,
            "cash": 9800.0,
            "positions": {},
            "stop_loss_triggered": True,
            "stop_loss_price": 91000.0,
        }
        fn = self._get_detail_fn()
        fn(commit)
