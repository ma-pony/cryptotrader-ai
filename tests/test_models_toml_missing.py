"""Tests for graceful degradation when models.toml is missing or unavailable."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cryptotrader.config import AppConfig, LLMConfig, ModelConfig


class TestModelsTomlMissing:
    def test_create_llm_with_role_falls_back_when_no_manifest(self):
        from cryptotrader.agents.base import create_llm
        from cryptotrader.llm.registry import reset_manifest_cache

        reset_manifest_cache()
        cfg = AppConfig()
        cfg.models = ModelConfig(analysis="gpt-4o-mini", fallback="gpt-4o-mini", models_path="/nonexistent/path")
        cfg.llm = LLMConfig()

        with (
            patch("cryptotrader.config.load_config", return_value=cfg),
            patch("cryptotrader.agents.base.ChatOpenAI", return_value=MagicMock()) as mock_cls,
        ):
            result = create_llm(role="analysis")

        assert result is not None
        mock_cls.assert_called()
        reset_manifest_cache()

    def test_create_llm_without_role_works_normally(self):
        from cryptotrader.agents.base import create_llm

        cfg = AppConfig()
        cfg.models = ModelConfig(analysis="gpt-4o-mini", fallback="gpt-4o-mini")
        cfg.llm = LLMConfig()

        with (
            patch("cryptotrader.config.load_config", return_value=cfg),
            patch("cryptotrader.agents.base.ChatOpenAI", return_value=MagicMock()) as mock_cls,
        ):
            result = create_llm(model="gpt-4o-mini")

        assert result is not None
        mock_cls.assert_called_once()
