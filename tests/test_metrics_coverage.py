"""Tests for metrics.py — MetricsCollector and get_metrics_collector singleton."""

from __future__ import annotations

from cryptotrader.metrics import MetricsCollector, get_metrics_collector


class TestMetricsCollector:
    def test_inc_llm_calls(self):
        mc = MetricsCollector()
        mc.inc_llm_calls(model="gpt-4o", node="agents")

    def test_inc_debate_skipped(self):
        mc = MetricsCollector()
        mc.inc_debate_skipped()

    def test_inc_verdict(self):
        mc = MetricsCollector()
        mc.inc_verdict(action="long")

    def test_inc_risk_rejected(self):
        mc = MetricsCollector()
        mc.inc_risk_rejected(check_name="daily_loss")

    def test_inc_trade_executed(self):
        mc = MetricsCollector()
        mc.inc_trade_executed(engine="paper", side="buy")

    def test_observe_execution_latency(self):
        mc = MetricsCollector()
        mc.observe_execution_latency(engine="paper", ms=150.0)

    def test_observe_pipeline_duration(self):
        mc = MetricsCollector()
        mc.observe_pipeline_duration(ms=5000.0)


class TestGetMetricsCollector:
    def test_singleton(self):
        mc1 = get_metrics_collector()
        mc2 = get_metrics_collector()
        assert mc1 is mc2

    def test_returns_metrics_collector(self):
        mc = get_metrics_collector()
        assert isinstance(mc, MetricsCollector)
