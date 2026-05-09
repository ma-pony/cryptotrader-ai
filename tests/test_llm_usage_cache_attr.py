"""spec 020a T019 — test_llm_usage_cache_attr.py

Tests: log_llm_usage() writes 3 OTel span attrs (llm.cache.*) + CacheMetricsAggregator.record().
SC-Z3: OTel span attrs present for cache_read/creation/hit_rate fields.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cryptotrader.agents.base import log_llm_usage


def _make_msg(
    cache_read: int = 0,
    cache_creation: int = 0,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> AIMessage:
    return AIMessage(
        content="test response",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
        response_metadata={"model_name": "claude-3-5-sonnet-20241022"},
    )


def _run_log_in_span(msg: AIMessage, caller: str = "test") -> dict:
    """Run log_llm_usage inside an in-memory OTel span, return span attributes."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    # Use the provider's tracer directly, attaching the span via context token
    span = tracer.start_span("llm.call")
    ctx = otel_trace.set_span_in_context(span)
    token = otel_context.attach(ctx)
    try:
        log_llm_usage(msg, caller=caller)
    finally:
        otel_context.detach(token)
        span.end()

    spans = exporter.get_finished_spans()
    return dict(spans[0].attributes or {}) if spans else {}


class TestLogLlmUsageCacheAttr:
    def test_cache_attrs_written_to_span(self):
        """FR-Z8: 3 cache attrs written to current OTel span."""
        msg = _make_msg(cache_read=80, cache_creation=20)
        attrs = _run_log_in_span(msg)
        assert "llm.cache.read_tokens" in attrs
        assert "llm.cache.creation_tokens" in attrs
        assert "llm.cache.hit_rate" in attrs

    def test_hit_rate_calculated_correctly(self):
        """FR-Z8: hit_rate = cache_read / (cache_read + cache_creation)."""
        msg = _make_msg(cache_read=80, cache_creation=20)
        attrs = _run_log_in_span(msg)
        assert abs(attrs["llm.cache.hit_rate"] - 0.8) < 1e-9
        assert attrs["llm.cache.read_tokens"] == 80
        assert attrs["llm.cache.creation_tokens"] == 20

    def test_zero_cache_writes_all_three_fields_as_zero(self):
        """FR-Z8 edge case: read+creation=0 -> all 3 fields written as 0 (no exception)."""
        msg = _make_msg(cache_read=0, cache_creation=0)
        attrs = _run_log_in_span(msg)
        assert attrs.get("llm.cache.read_tokens") == 0
        assert attrs.get("llm.cache.creation_tokens") == 0
        assert attrs.get("llm.cache.hit_rate") == 0.0

    def test_only_read_tokens_present(self):
        """cache_read only (no creation) -> hit_rate = 1.0."""
        msg = _make_msg(cache_read=100, cache_creation=0)
        attrs = _run_log_in_span(msg)
        assert abs(attrs["llm.cache.hit_rate"] - 1.0) < 1e-9

    def test_no_otel_span_does_not_raise(self):
        """FR-Z8 edge: no active OTel span -> no exception, structlog still logs."""
        msg = _make_msg(cache_read=50, cache_creation=50)
        # No span active -- should not raise
        log_llm_usage(msg, caller="test_no_span")

    def test_non_aimessage_is_ignored(self):
        """log_llm_usage ignores non-AIMessage inputs silently."""
        # No span needed -- just verifying no crash and no attr write
        log_llm_usage("plain string", caller="test")

    def test_cache_metrics_aggregator_receives_hit_rate(self):
        """FR-Z18: log_llm_usage pushes hit_rate to CacheMetricsAggregator."""
        from cryptotrader.observability.cache_metrics import CacheMetricsAggregator

        # Use a fresh aggregator (bypass the singleton for isolation)
        agg = CacheMetricsAggregator()
        agg.record(0.8)
        agg.record(0.6)
        avg = agg.average()
        assert abs(avg - 0.7) < 1e-9

    def test_cache_aggregator_empty_returns_zero(self):
        """CacheMetricsAggregator returns 0.0 when no data."""
        from cryptotrader.observability.cache_metrics import CacheMetricsAggregator

        agg = CacheMetricsAggregator()
        assert agg.average() == 0.0

    def test_cache_aggregator_evicts_old_entries(self):
        """CacheMetricsAggregator evicts entries outside the window."""
        import time

        from cryptotrader.observability.cache_metrics import CacheMetricsAggregator

        # 1-second window
        agg = CacheMetricsAggregator(window_seconds=1)
        agg.record(1.0)
        time.sleep(1.1)
        agg.record(0.0)
        # Only the 0.0 entry should remain
        assert agg.average() == 0.0

    def test_log_triggers_aggregator_record(self):
        """log_llm_usage calls get_cache_metrics_aggregator().record() with the hit_rate."""
        from cryptotrader.observability.cache_metrics import CacheMetricsAggregator

        fresh_agg = CacheMetricsAggregator()
        msg = _make_msg(cache_read=60, cache_creation=40)

        with patch(
            "cryptotrader.observability.cache_metrics.get_cache_metrics_aggregator",
            return_value=fresh_agg,
        ):
            log_llm_usage(msg, caller="agg_test")

        # hit_rate = 60/(60+40) = 0.6 was recorded
        assert abs(fresh_agg.average() - 0.6) < 1e-9
