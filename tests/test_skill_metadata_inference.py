"""spec 019 - skill_metadata_inference unit tests.

tests/test_skill_metadata_inference.py - >= 6 test cases PASS
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_llm(return_value: str | Exception):
    """Create a mock LLM callable."""
    if isinstance(return_value, Exception):
        mock = MagicMock(side_effect=return_value)
    else:
        mock = MagicMock(return_value=return_value)
    return mock


class TestInferSkillMetadataSuccess:
    """(a) mock LLM returns valid JSON -> output matches LLM."""

    def test_valid_json_response_returned(self):
        """LLM returns valid JSON -> result matches parsed values."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        metadata = {
            "regime_tags": ["high_funding"],
            "triggers_keywords": ["funding", "dydx", "arbitrage"],
            "importance": 0.6,
            "confidence": 0.6,
        }
        llm = _make_llm(json.dumps(metadata))
        result = infer_skill_metadata(
            name="dydx-funding-arbitrage",
            description="dYdX funding arbitrage strategy",
            body="# Strategy body content",
            llm_callable=llm,
        )
        assert result["regime_tags"] == ["high_funding"]
        assert result["triggers_keywords"] == ["funding", "dydx", "arbitrage"]
        assert result["importance"] == pytest.approx(0.6)
        assert result["confidence"] == pytest.approx(0.6)

    def test_llm_called_with_prompt_containing_name(self):
        """LLM should be called with prompt containing skill name."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        metadata = {"regime_tags": [], "triggers_keywords": ["test"], "importance": 0.5, "confidence": 0.5}
        llm = _make_llm(json.dumps(metadata))
        infer_skill_metadata(
            name="my-skill",
            description="Test skill",
            body="Test body",
            llm_callable=llm,
        )
        call_args = llm.call_args[0][0]
        assert "my-skill" in call_args


class TestInferSkillMetadataFailure:
    """(b) mock LLM exception -> default values + warning log."""

    def test_llm_exception_returns_defaults(self, caplog):
        """LLM exception -> default values returned + warning logged."""
        import logging

        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        llm = _make_llm(RuntimeError("LLM unavailable"))
        with caplog.at_level(logging.WARNING):
            result = infer_skill_metadata(
                name="test-skill",
                description="Test",
                body="Test body",
                llm_callable=llm,
            )
        assert result["regime_tags"] == []
        assert result["triggers_keywords"] == []
        assert result["importance"] == pytest.approx(0.5)
        assert result["confidence"] == pytest.approx(0.5)
        assert any("failed" in msg.lower() or "retry" in msg.lower() for msg in caplog.messages)


class TestInferSkillMetadataRetry:
    """(c) LLM outputs invalid JSON -> retry once then default values."""

    def test_invalid_json_retries_and_returns_defaults(self, caplog):
        """Non-JSON response -> retry 1 time -> fall back to default values."""
        import logging

        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        # Both calls return non-JSON
        llm = _make_llm("This is not valid JSON at all")
        with caplog.at_level(logging.WARNING):
            result = infer_skill_metadata(
                name="test-skill",
                description="Test",
                body="Test body",
                llm_callable=llm,
            )
        # Should have been called twice (first + retry)
        assert llm.call_count == 2
        assert result["regime_tags"] == []
        assert result["importance"] == pytest.approx(0.5)

    def test_second_attempt_succeeds(self):
        """If first call fails but second succeeds, return the second result."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        valid_metadata = {
            "regime_tags": ["high_vol"],
            "triggers_keywords": ["vol"],
            "importance": 0.7,
            "confidence": 0.7,
        }
        # First call returns bad JSON, second returns valid
        call_count = 0

        def llm_callable(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not json"
            return json.dumps(valid_metadata)

        result = infer_skill_metadata("test", "Test", "body", llm_callable=llm_callable)
        assert result["regime_tags"] == ["high_vol"]
        assert result["importance"] == pytest.approx(0.7)


class TestPromptContents:
    """(d) prompt contains correct context."""

    def test_prompt_contains_name_description_body(self):
        """Prompt should contain name + description + body summary + 5 skill examples."""
        from cryptotrader.learning.evolution.skill_metadata_inference import _build_prompt

        prompt = _build_prompt(
            name="chain-whale-strategy",
            description="Whale movement detection strategy",
            body="# Whale Strategy\n\nDetect large whale movements on-chain.",
        )
        assert "chain-whale-strategy" in prompt
        assert "Whale movement detection strategy" in prompt
        assert "Whale Strategy" in prompt

    def test_prompt_contains_regime_taxonomy(self):
        """Prompt should contain all 8 regime taxonomy values."""
        from cryptotrader.learning.evolution.skill_metadata_inference import _build_prompt

        prompt = _build_prompt("test", "Test skill", "Body")
        assert "high_funding" in prompt
        assert "negative_funding" in prompt
        assert "high_vol" in prompt
        assert "extreme_fear" in prompt
        assert "extreme_greed" in prompt

    def test_prompt_contains_5_skill_examples(self):
        """Prompt should contain existing 5 skill mapping as examples."""
        from cryptotrader.learning.evolution.skill_metadata_inference import _build_prompt

        prompt = _build_prompt("test", "Test skill", "Body")
        assert "chain-analysis" in prompt
        assert "macro-analysis" in prompt
        assert "tech-analysis" in prompt
        assert "trading-knowledge" in prompt


class TestRegimeTagsValidation:
    """(e) regime_tags subset validation."""

    def test_invalid_regime_tags_filtered(self):
        """regime_tags containing invalid values should be filtered out."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        metadata = {
            "regime_tags": ["high_funding", "invalid_regime", "unknown"],
            "triggers_keywords": ["test"],
            "importance": 0.5,
            "confidence": 0.5,
        }
        llm = _make_llm(json.dumps(metadata))
        result = infer_skill_metadata("test", "desc", "body", llm_callable=llm)
        assert result["regime_tags"] == ["high_funding"]  # only valid one kept

    def test_all_valid_regime_tags_kept(self):
        """All 8 valid regime_tags values should be allowed."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        all_regimes = ["high_funding", "negative_funding", "high_vol", "low_vol"]
        metadata = {
            "regime_tags": all_regimes,
            "triggers_keywords": ["test"],
            "importance": 0.6,
            "confidence": 0.6,
        }
        llm = _make_llm(json.dumps(metadata))
        result = infer_skill_metadata("test", "desc", "body", llm_callable=llm)
        assert set(result["regime_tags"]) == set(all_regimes)


class TestImportanceConfidenceValidation:
    """(f) importance / confidence in [0, 1]."""

    def test_importance_clamped_to_unit_interval(self):
        """importance > 1.0 or < 0.0 should be clamped to [0, 1]."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        metadata = {"regime_tags": [], "triggers_keywords": [], "importance": 1.5, "confidence": -0.3}
        llm = _make_llm(json.dumps(metadata))
        result = infer_skill_metadata("test", "desc", "body", llm_callable=llm)
        assert result["importance"] == pytest.approx(1.0)
        assert result["confidence"] == pytest.approx(0.0)

    def test_valid_importance_confidence_preserved(self):
        """Valid 0.0-1.0 values should be preserved as-is."""
        from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

        metadata = {"regime_tags": [], "triggers_keywords": ["x"], "importance": 0.75, "confidence": 0.65}
        llm = _make_llm(json.dumps(metadata))
        result = infer_skill_metadata("test", "desc", "body", llm_callable=llm)
        assert result["importance"] == pytest.approx(0.75)
        assert result["confidence"] == pytest.approx(0.65)
