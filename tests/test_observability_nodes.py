"""Tests for task 2.1, 2.2, 2.3: node-layer observability extensions.

Task 2.1 — debate_gate writes consensus_metrics + debate_skip_reason into state["data"].
Task 2.2 — make_verdict injects verdict_source into all three branches.
Task 2.3 — journal_trade and journal_rejection pass the 5 new fields to build_commit().
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

# ── helpers ──────────────────────────────────────────────────────────────────


def _base_analyses():
    return {
        "tech_agent": {"direction": "bullish", "confidence": 0.9},
        "chain_agent": {"direction": "bullish", "confidence": 0.85},
        "news_agent": {"direction": "bullish", "confidence": 0.8},
        "macro_agent": {"direction": "bullish", "confidence": 0.88},
    }


def _base_state(**extra_data):
    data = {
        "analyses": _base_analyses(),
        "snapshot_summary": {"pair": "BTC/USDT", "price": 50000.0},
        "verdict": {
            "action": "long",
            "confidence": 0.8,
            "position_scale": 0.5,
            "divergence": 0.1,
            "reasoning": "bullish",
            "thesis": "",
            "invalidation": "",
        },
        "risk_gate": {"passed": True, "rejected_by": "", "reason": ""},
        **extra_data,
    }
    return {
        "messages": [],
        "data": data,
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "models": {},
            "analysis_model": "",
            "debate_model": "",
            "verdict_model": "",
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _make_fake_commit(**kwargs):
    """Build a minimal DecisionCommit for use in mock build_commit."""
    from cryptotrader.models import DecisionCommit

    return DecisionCommit(
        hash="abc123",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair=kwargs.get("pair", "BTC/USDT"),
        snapshot_summary={},
        analyses={},
        debate_rounds=0,
    )


# ── Task 2.1: debate_gate ─────────────────────────────────────────────────────


class TestDebateGateObservability:
    """debate_gate must always write consensus_metrics and debate_skip_reason."""

    def _make_debate_config(self, skip_debate=True, consensus_threshold=0.5, confusion_threshold=0.05):
        cfg = MagicMock()
        cfg.debate.skip_debate = skip_debate
        cfg.debate.consensus_skip_threshold = consensus_threshold
        cfg.debate.confusion_skip_threshold = confusion_threshold
        cfg.debate.confusion_max_dispersion = 0.2
        return cfg

    async def test_debate_gate_no_skip_writes_consensus_metrics(self):
        """When debate is not skipped, consensus_metrics is still written to state."""
        from cryptotrader.nodes.debate import debate_gate

        cfg = self._make_debate_config(skip_debate=False)
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = await debate_gate(_base_state())

        data = result["data"]
        assert "consensus_metrics" in data
        metrics = data["consensus_metrics"]
        assert "strength" in metrics
        assert "mean_score" in metrics
        assert "dispersion" in metrics
        assert "skip_threshold" in metrics
        assert "confusion_threshold" in metrics

    async def test_debate_gate_no_skip_reason_is_empty_string(self):
        """When debate is not skipped, debate_skip_reason is empty string."""
        from cryptotrader.nodes.debate import debate_gate

        cfg = self._make_debate_config(skip_debate=False)
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = await debate_gate(_base_state())

        assert result["data"]["debate_skip_reason"] == ""

    async def test_debate_gate_consensus_skip_writes_reason(self):
        """When skipped by strong consensus, debate_skip_reason is 'consensus'."""
        from cryptotrader.nodes.debate import debate_gate

        # All bullish high confidence -> strong consensus > 0.3 threshold
        cfg = self._make_debate_config(skip_debate=True, consensus_threshold=0.3)
        with (
            patch("cryptotrader.config.load_config", return_value=cfg),
            patch("cryptotrader.metrics.get_metrics_collector") as mock_mc,
        ):
            mock_mc.return_value.inc_debate_skipped = MagicMock()
            result = await debate_gate(_base_state())

        data = result["data"]
        assert data["debate_skipped"] is True
        assert data["debate_skip_reason"] == "consensus"

    async def test_debate_gate_confusion_skip_writes_reason(self):
        """When skipped by shared confusion, debate_skip_reason is 'confusion'."""
        from cryptotrader.nodes.debate import debate_gate

        # Near-neutral analyses with low dispersion -> confusion skip
        confusion_analyses = {
            "tech_agent": {"direction": "neutral", "confidence": 0.02},
            "chain_agent": {"direction": "neutral", "confidence": 0.01},
            "news_agent": {"direction": "neutral", "confidence": 0.02},
            "macro_agent": {"direction": "neutral", "confidence": 0.01},
        }
        cfg = self._make_debate_config(
            skip_debate=True,
            consensus_threshold=0.99,  # won't trigger consensus
            confusion_threshold=0.1,
        )
        cfg.debate.confusion_max_dispersion = 0.5  # permissive

        state = _base_state()
        state["data"]["analyses"] = confusion_analyses

        with (
            patch("cryptotrader.config.load_config", return_value=cfg),
            patch("cryptotrader.metrics.get_metrics_collector") as mock_mc,
        ):
            mock_mc.return_value.inc_debate_skipped = MagicMock()
            result = await debate_gate(state)

        data = result["data"]
        assert data["debate_skipped"] is True
        assert data["debate_skip_reason"] == "confusion"

    async def test_debate_gate_consensus_metrics_has_config_thresholds(self):
        """consensus_metrics includes skip_threshold and confusion_threshold from config."""
        from cryptotrader.nodes.debate import debate_gate

        cfg = self._make_debate_config(
            skip_debate=False,
            consensus_threshold=0.42,
            confusion_threshold=0.07,
        )
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = await debate_gate(_base_state())

        metrics = result["data"]["consensus_metrics"]
        assert metrics["skip_threshold"] == 0.42
        assert metrics["confusion_threshold"] == 0.07

    async def test_debate_gate_consensus_metrics_values_are_numeric(self):
        """consensus_metrics strength/mean_score/dispersion are float values."""
        from cryptotrader.nodes.debate import debate_gate

        cfg = self._make_debate_config(skip_debate=False)
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = await debate_gate(_base_state())

        metrics = result["data"]["consensus_metrics"]
        assert isinstance(metrics["strength"], float)
        assert isinstance(metrics["mean_score"], float)
        assert isinstance(metrics["dispersion"], float)


# ── Task 2.2: make_verdict ────────────────────────────────────────────────────


class TestMakeVerdictSource:
    """make_verdict must inject verdict_source into the verdict dict for all three branches."""

    def _mock_verdict(self, action="long"):
        v = MagicMock()
        v.action = action
        v.confidence = 0.8
        v.position_scale = 0.5
        v.divergence = 0.1
        v.reasoning = "test"
        v.thesis = ""
        v.invalidation = ""
        return v

    async def test_make_verdict_ai_branch_sets_source_ai(self):
        """AI verdict branch sets verdict_source='ai'."""
        from cryptotrader.nodes.verdict import make_verdict

        state = _base_state()
        state["metadata"]["llm_verdict"] = True
        state["data"]["debate_skipped"] = False

        mock_v = self._mock_verdict("long")
        cfg = MagicMock()
        cfg.models.verdict = ""
        cfg.models.fallback = "gpt-4o-mini"

        with (
            patch("cryptotrader.config.load_config", return_value=cfg),
            patch("cryptotrader.nodes.verdict._gather_risk_constraints", new=AsyncMock(return_value={})),
            patch("cryptotrader.metrics.get_metrics_collector") as mock_mc,
        ):
            mock_mc.return_value.inc_verdict = MagicMock()
            # make_verdict_llm is imported locally inside make_verdict
            with patch("cryptotrader.debate.verdict.make_verdict_llm", new=AsyncMock(return_value=mock_v)):
                result = await make_verdict(state)

        assert result["data"]["verdict"]["verdict_source"] == "ai"

    async def test_make_verdict_weighted_branch_sets_source_weighted(self):
        """use_llm_verdict=False path sets verdict_source='weighted'."""
        from cryptotrader.nodes.verdict import make_verdict

        state = _base_state()
        state["metadata"]["llm_verdict"] = False

        mock_v = self._mock_verdict("hold")

        with (
            patch("cryptotrader.metrics.get_metrics_collector") as mock_mc,
            patch("cryptotrader.debate.verdict.make_verdict_weighted", return_value=mock_v),
        ):
            mock_mc.return_value.inc_verdict = MagicMock()
            result = await make_verdict(state)

        assert result["data"]["verdict"]["verdict_source"] == "weighted"

    async def test_make_verdict_downgraded_sets_source_weighted(self):
        """Downgraded verdict (debate skipped, flat, no circuit breaker) sets verdict_source='weighted'."""
        from cryptotrader.nodes.verdict import make_verdict

        state = _base_state()
        state["metadata"]["llm_verdict"] = True
        state["data"]["debate_skipped"] = True
        state["data"]["position_context"] = {"side": "flat"}

        mock_v = self._mock_verdict("hold")

        with (
            patch("cryptotrader.nodes.verdict._should_downgrade_to_weighted", new=AsyncMock(return_value=True)),
            patch("cryptotrader.metrics.get_metrics_collector") as mock_mc,
            patch("cryptotrader.debate.verdict.make_verdict_weighted", return_value=mock_v),
        ):
            mock_mc.return_value.inc_verdict = MagicMock()
            result = await make_verdict(state)

        assert result["data"]["verdict"]["verdict_source"] == "weighted"

    async def test_make_verdict_all_mock_hold_sets_source_hold_all_mock(self):
        """All-mock branch (all agents returned is_mock=True) sets verdict_source='hold_all_mock'."""
        from cryptotrader.nodes.verdict import make_verdict

        state = _base_state()
        state["data"]["analyses"] = {
            "tech_agent": {"direction": "hold", "confidence": 0.0, "is_mock": True},
            "chain_agent": {"direction": "hold", "confidence": 0.0, "is_mock": True},
        }

        with patch("cryptotrader.metrics.get_metrics_collector") as mock_mc:
            mock_mc.return_value.inc_verdict = MagicMock()
            result = await make_verdict(state)

        assert result["data"]["verdict"]["verdict_source"] == "hold_all_mock"


# ── Task 2.3: journal nodes ───────────────────────────────────────────────────


class TestJournalObservabilityFields:
    """journal_trade and journal_rejection must pass the 5 new fields to build_commit()."""

    def _make_build_commit_spy(self):
        """Return (spy_fn, captured_kwargs_dict).

        spy_fn replaces build_commit and records all kwargs.
        """
        captured: dict = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return _make_fake_commit(**kwargs)

        return spy, captured

    # ── journal_trade ────────────────────────────────────────────────────────

    async def test_journal_trade_passes_consensus_metrics(self):
        """journal_trade passes consensus_metrics from state to build_commit."""
        from cryptotrader.nodes.journal import journal_trade

        metrics = {
            "strength": 0.8,
            "mean_score": 0.7,
            "dispersion": 0.1,
            "skip_threshold": 0.5,
            "confusion_threshold": 0.05,
        }
        state = _base_state(consensus_metrics=metrics)
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("consensus_metrics") == metrics

    async def test_journal_trade_passes_verdict_source(self):
        """journal_trade passes verdict_source from state['data']['verdict'] to build_commit."""
        from cryptotrader.nodes.journal import journal_trade

        state = _base_state()
        state["data"]["verdict"]["verdict_source"] = "weighted"
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("verdict_source") == "weighted"

    async def test_journal_trade_verdict_source_defaults_to_ai_when_missing(self):
        """journal_trade uses 'ai' as verdict_source default when the key is absent."""
        from cryptotrader.nodes.journal import journal_trade

        state = _base_state()
        state["data"]["verdict"].pop("verdict_source", None)
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("verdict_source") == "ai"

    async def test_journal_trade_passes_experience_memory(self):
        """journal_trade passes experience_memory from state to build_commit."""
        from cryptotrader.nodes.journal import journal_trade

        exp_mem = {"success_patterns": ["bullish RSI"], "forbidden_zones": []}
        state = _base_state(experience_memory=exp_mem)
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("experience_memory") == exp_mem

    async def test_journal_trade_passes_node_trace(self):
        """journal_trade passes node_trace from state to build_commit."""
        from cryptotrader.nodes.journal import journal_trade

        trace = [{"node": "data_node", "duration_ms": 200, "summary": "fetched"}]
        state = _base_state(node_trace=trace)
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("node_trace") == trace

    async def test_journal_trade_passes_debate_skip_reason(self):
        """journal_trade passes debate_skip_reason from state to build_commit."""
        from cryptotrader.nodes.journal import journal_trade

        state = _base_state(debate_skip_reason="consensus")
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("debate_skip_reason") == "consensus"

    async def test_journal_trade_uses_defaults_for_missing_fields(self):
        """journal_trade uses safe defaults when observability fields are absent from state."""
        from cryptotrader.nodes.journal import journal_trade

        state = _base_state()  # no extra observability fields
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.nodes.journal._get_portfolio_snapshot", new=AsyncMock(return_value={})),
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-1"),
        ):
            mock_store.return_value.commit = AsyncMock()
            await journal_trade(state)

        assert captured.get("consensus_metrics") is None
        assert captured.get("verdict_source") == "ai"
        assert captured.get("experience_memory") == {}
        assert captured.get("node_trace") == []
        assert captured.get("debate_skip_reason") == ""

    # ── journal_rejection ────────────────────────────────────────────────────

    async def test_journal_rejection_passes_consensus_metrics(self):
        """journal_rejection passes consensus_metrics from state to build_commit."""
        from cryptotrader.nodes.journal import journal_rejection

        metrics = {
            "strength": 0.3,
            "mean_score": 0.2,
            "dispersion": 0.15,
            "skip_threshold": 0.5,
            "confusion_threshold": 0.05,
        }
        state = _base_state(consensus_metrics=metrics)
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "daily_loss", "reason": "exceeded"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("consensus_metrics") == metrics

    async def test_journal_rejection_passes_verdict_source(self):
        """journal_rejection passes verdict_source from state to build_commit."""
        from cryptotrader.nodes.journal import journal_rejection

        state = _base_state()
        state["data"]["verdict"]["verdict_source"] = "hold_all_mock"
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "volatility", "reason": "too high"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("verdict_source") == "hold_all_mock"

    async def test_journal_rejection_passes_experience_memory(self):
        """journal_rejection passes experience_memory from state to build_commit."""
        from cryptotrader.nodes.journal import journal_rejection

        exp_mem = {"forbidden_zones": ["high volatility bearish"]}
        state = _base_state(experience_memory=exp_mem)
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "check", "reason": "r"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("experience_memory") == exp_mem

    async def test_journal_rejection_passes_node_trace(self):
        """journal_rejection passes node_trace from state to build_commit."""
        from cryptotrader.nodes.journal import journal_rejection

        trace = [{"node": "debate_gate", "duration_ms": 10, "summary": "skipped"}]
        state = _base_state(node_trace=trace)
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "check", "reason": "r"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("node_trace") == trace

    async def test_journal_rejection_passes_debate_skip_reason(self):
        """journal_rejection passes debate_skip_reason from state to build_commit."""
        from cryptotrader.nodes.journal import journal_rejection

        state = _base_state(debate_skip_reason="confusion")
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "check", "reason": "r"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("debate_skip_reason") == "confusion"

    async def test_journal_rejection_uses_defaults_for_missing_fields(self):
        """journal_rejection uses safe defaults when observability fields are absent."""
        from cryptotrader.nodes.journal import journal_rejection

        state = _base_state()
        state["data"]["risk_gate"] = {"passed": False, "rejected_by": "check", "reason": "r"}
        spy, captured = self._make_build_commit_spy()

        with (
            patch("cryptotrader.journal.commit.build_commit", side_effect=spy),
            patch("cryptotrader.journal.store.JournalStore") as mock_store,
            patch("cryptotrader.tracing.get_trace_id", return_value="trace-2"),
            patch("cryptotrader.nodes.verdict._get_notifier") as mock_notifier,
        ):
            mock_store.return_value.commit = AsyncMock()
            mock_notifier.return_value.notify = AsyncMock()
            await journal_rejection(state)

        assert captured.get("consensus_metrics") is None
        assert captured.get("verdict_source") == "ai"
        assert captured.get("experience_memory") == {}
        assert captured.get("node_trace") == []
        assert captured.get("debate_skip_reason") == ""
