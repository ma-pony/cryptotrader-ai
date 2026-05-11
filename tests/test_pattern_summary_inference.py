"""Tests for spec 021 pattern_summary_inference enrichment."""

from __future__ import annotations

import json

from cryptotrader.learning.evolution.pattern_summary_inference import (
    _default_body,
    _default_description,
    _extract_json,
    infer_pattern_summary,
)


def _fake_llm_factory(payload: dict | str):
    """Return a callable that mimics ChatOpenAI.invoke — returns an object with .content."""

    class _Resp:
        def __init__(self, body: str) -> None:
            self.content = body

    body = json.dumps(payload) if isinstance(payload, dict) else payload

    def _llm(_prompt: str):
        return _Resp(body)

    return _llm


def test_default_template_when_no_excerpts():
    """Empty agent_analyses_snippet → fall back to template, no LLM call."""
    cases = [{"cycle_id": "a1", "pnl": 1.0, "regime_tags": [], "agent_analyses_snippet": ""}]
    desc, body = infer_pattern_summary("tech", "sma_breakdown_short", cases)
    assert desc == _default_description("tech", "sma_breakdown_short", 1)
    assert body == _default_body("sma_breakdown_short", 1, ["a1"])


def test_llm_success_produces_rich_description_and_body():
    """LLM returns valid JSON → description + body reflect it."""
    cases = [
        {
            "cycle_id": "a1",
            "pnl": -10.0,
            "regime_tags": ["low_vol"],
            "agent_analyses_snippet": "Price broke below SMA20 with MACD diverging.",
        },
        {
            "cycle_id": "a2",
            "pnl": -5.0,
            "regime_tags": ["low_vol"],
            "agent_analyses_snippet": "Same pattern, weak bounce volume.",
        },
    ]
    payload = {
        "description": "Short on SMA20 breakdown with weak bounce.",
        "trigger": "Price closes below SMA20 + negative MACD histogram.",
        "regime": "Low-vol trending-down markets.",
        "failure": "Strong reclaim of SMA20 with volume.",
    }
    desc, body = infer_pattern_summary(
        "tech",
        "sma_breakdown_short",
        cases,
        llm_callable=_fake_llm_factory(payload),
    )
    assert desc == "Short on SMA20 breakdown with weak bounce."
    assert "## Trigger" in body
    assert "Price closes below SMA20" in body
    assert "## Applicable regime" in body
    assert "## Invalidation" in body
    assert "Source cycles (first 5): ['a1', 'a2']" in body


def test_llm_returns_fenced_json():
    """LLM wraps JSON in ```json fence → extractor strips fence."""
    cases = [
        {
            "cycle_id": "c1",
            "pnl": 0.0,
            "regime_tags": [],
            "agent_analyses_snippet": "some text",
        }
    ]
    fenced = "```json\n" + json.dumps({"description": "X", "trigger": "Y"}) + "\n```"
    desc, body = infer_pattern_summary(
        "chain", "test_pattern", cases, llm_callable=_fake_llm_factory(fenced),
    )
    assert desc == "X"
    assert "Y" in body


def test_llm_garbage_falls_back_to_template():
    """LLM returns non-JSON → retry → still bad → fall back to template."""
    cases = [
        {
            "cycle_id": "c1",
            "pnl": 0.0,
            "regime_tags": [],
            "agent_analyses_snippet": "agent said this thing",
        }
    ]
    desc, body = infer_pattern_summary(
        "tech",
        "noise_pattern",
        cases,
        llm_callable=_fake_llm_factory("totally not json"),
    )
    assert desc == _default_description("tech", "noise_pattern", 1)
    assert body == _default_body("noise_pattern", 1, ["c1"])


def test_description_capped_at_280_chars():
    """A verbose LLM description gets truncated."""
    cases = [
        {
            "cycle_id": "c1",
            "pnl": 0.0,
            "regime_tags": [],
            "agent_analyses_snippet": "text",
        }
    ]
    payload = {"description": "x" * 500, "trigger": "t", "regime": "r", "failure": "f"}
    desc, _ = infer_pattern_summary(
        "tech", "verbose", cases, llm_callable=_fake_llm_factory(payload)
    )
    assert len(desc) <= 280
    assert desc.endswith("...")


def test_extract_json_handles_bare_object():
    """Plain JSON object (no fence) parses."""
    raw = 'leading prose {"description":"hi"} trailing'
    assert _extract_json(raw) == {"description": "hi"}


def test_extract_json_returns_none_on_empty():
    assert _extract_json("") is None
    assert _extract_json("no braces here") is None
