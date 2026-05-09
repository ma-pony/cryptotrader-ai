"""spec 020a T032 — test_e2e_trilogy_ops.py

E2E gate: simulates a single trading cycle with 4 mocked agent LLM calls, each
emitting cache usage.  Asserts that:
  - OTel trace contains ≥ 4 agent "llm.call" spans
  - Each span carries all 3 llm.cache.* attributes
  - At least 1 span has cache_read_tokens > 0 (cache retrieval hit)

SC-Z8 / FR-Z8.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cryptotrader.agents.base import log_llm_usage

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_response(
    cache_read: int,
    cache_creation: int,
    input_tokens: int = 500,
    output_tokens: int = 100,
) -> AIMessage:
    return AIMessage(
        content="agent analysis output",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
        response_metadata={"model_name": "claude-3-5-sonnet-20241022"},
    )


def _simulate_cycle(agent_responses: list[AIMessage]) -> list[dict]:
    """
    Run each agent response through log_llm_usage inside its own OTel span.
    Returns the list of span attribute dicts.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("e2e_test")

    agent_names = ["tech_agent", "news_agent", "onchain_agent", "risk_agent"]

    for name, response in zip(agent_names[: len(agent_responses)], agent_responses, strict=False):
        span = tracer.start_span(f"llm.call.{name}")
        ctx = otel_trace.set_span_in_context(span)
        token = otel_context.attach(ctx)
        try:
            log_llm_usage(response, caller=name)
        finally:
            otel_context.detach(token)
            span.end()

    return [dict(s.attributes or {}) for s in exporter.get_finished_spans()]


# ── test cases ────────────────────────────────────────────────────────────────


class TestE2ETrilogyOps:
    def _make_cycle_responses(self) -> list[AIMessage]:
        """4 agent responses: first 3 have cache hits, last one creates cache."""
        return [
            _make_response(cache_read=400, cache_creation=0),  # tech  — full hit
            _make_response(cache_read=350, cache_creation=50),  # news  — partial hit
            _make_response(cache_read=300, cache_creation=100),  # onchain — partial
            _make_response(cache_read=0, cache_creation=500),  # risk  — creation
        ]

    def test_at_least_four_agent_spans(self):
        """SC-Z8(a): OTel trace contains ≥ 4 agent LLM spans."""
        spans = _simulate_cycle(self._make_cycle_responses())
        assert len(spans) >= 4, f"Expected ≥4 spans, got {len(spans)}"

    def test_all_spans_have_three_cache_attrs(self):
        """SC-Z8(b): every agent span carries all 3 llm.cache.* attributes."""
        spans = _simulate_cycle(self._make_cycle_responses())
        for i, attrs in enumerate(spans):
            assert "llm.cache.read_tokens" in attrs, f"span {i} missing read_tokens"
            assert "llm.cache.creation_tokens" in attrs, f"span {i} missing creation_tokens"
            assert "llm.cache.hit_rate" in attrs, f"span {i} missing hit_rate"

    def test_at_least_one_cache_retrieval_hit(self):
        """SC-Z8(c): ≥1 span has cache_read_tokens > 0 (retrieval hit)."""
        spans = _simulate_cycle(self._make_cycle_responses())
        hits = [s for s in spans if s.get("llm.cache.read_tokens", 0) > 0]
        assert len(hits) >= 1, "Expected at least 1 span with cache_read_tokens > 0"

    def test_hit_rate_values_are_in_range(self):
        """All hit_rate values are in [0.0, 1.0]."""
        spans = _simulate_cycle(self._make_cycle_responses())
        for i, attrs in enumerate(spans):
            hit_rate = attrs.get("llm.cache.hit_rate", -1.0)
            assert 0.0 <= hit_rate <= 1.0, f"span {i} hit_rate {hit_rate!r} out of [0, 1]"

    def test_full_cache_hit_rate_is_one(self):
        """tech_agent with cache_read=400, creation=0 -> hit_rate = 1.0."""
        spans = _simulate_cycle(self._make_cycle_responses())
        tech_span = spans[0]
        assert abs(tech_span["llm.cache.hit_rate"] - 1.0) < 1e-9

    def test_cache_creation_only_hit_rate_is_zero(self):
        """risk_agent with cache_read=0, creation=500 -> hit_rate = 0.0."""
        spans = _simulate_cycle(self._make_cycle_responses())
        risk_span = spans[3]
        assert risk_span["llm.cache.hit_rate"] == 0.0

    def test_partial_hit_rate_calculation(self):
        """news_agent: cache_read=350, creation=50 -> hit_rate = 350/400 = 0.875."""
        spans = _simulate_cycle(self._make_cycle_responses())
        news_span = spans[1]
        assert abs(news_span["llm.cache.hit_rate"] - 0.875) < 1e-9
