"""Tests for prompt caching — cache_control breakpoints and log_llm_usage."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cryptotrader.agents.base import log_llm_usage
from cryptotrader.llm.prompt_cache import (
    apply_cache_control,
    is_anthropic_model,
    should_cache,
)


class TestPromptCacheLogging:
    def test_anthropic_cache_read_tokens(self):
        msg = AIMessage(content="hello")
        msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
        }
        msg.response_metadata = {"model_name": "claude-sonnet-4-20250514"}

        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage(msg, caller="test_agent")
            mock_log.info.assert_called_once()
            kwargs = mock_log.info.call_args[1]
            assert kwargs["prompt_cache_hit"] is True
            assert kwargs["cache_read_input_tokens"] == 80

    def test_openai_cached_tokens_via_details(self):
        msg = AIMessage(content="hello")
        msg.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 100,
            "input_token_details": {"cached": 150},
        }
        msg.response_metadata = {"model_name": "gpt-4o"}

        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage(msg, caller="test")
            kwargs = mock_log.info.call_args[1]
            assert kwargs["prompt_cache_hit"] is True
            assert kwargs["cache_read_input_tokens"] == 150

    def test_no_cache_hit(self):
        msg = AIMessage(content="hello")
        msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
        }
        msg.response_metadata = {"model_name": "gpt-4o"}

        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage(msg, caller="test")
            kwargs = mock_log.info.call_args[1]
            assert kwargs["prompt_cache_hit"] is False
            assert kwargs["cache_read_input_tokens"] == 0

    def test_non_ai_message_skipped(self):
        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage("not an AIMessage", caller="test")
            mock_log.info.assert_not_called()

    def test_no_usage_metadata_skipped(self):
        msg = AIMessage(content="hello")
        msg.usage_metadata = None

        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage(msg, caller="test")
            mock_log.info.assert_not_called()

    def test_anthropic_cache_takes_priority(self):
        msg = AIMessage(content="hello")
        msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "input_token_details": {"cached": 40},
        }
        msg.response_metadata = {"model_name": "claude-sonnet-4-20250514"}

        with patch("cryptotrader.agents.base._structlog") as mock_log:
            log_llm_usage(msg, caller="test")
            kwargs = mock_log.info.call_args[1]
            assert kwargs["cache_read_input_tokens"] == 80


class TestIsAnthropicModel:
    def test_claude_model(self):
        assert is_anthropic_model("claude-3.5-sonnet") is True

    def test_claude_opus(self):
        assert is_anthropic_model("claude-3-opus-20240229") is True

    def test_openai_model(self):
        assert is_anthropic_model("gpt-4o") is False

    def test_gemini_model(self):
        assert is_anthropic_model("gemini-2.0-flash") is False

    def test_deepseek_model(self):
        assert is_anthropic_model("deepseek-chat") is False

    def test_empty_string(self):
        assert is_anthropic_model("") is False

    def test_case_insensitive(self):
        assert is_anthropic_model("Claude-3.5-Sonnet") is True


class TestApplyCacheControl:
    def test_transforms_system_message(self):
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
        ]
        result = apply_cache_control(msgs)
        assert len(result) == 2
        sys_msg = result[0]
        assert isinstance(sys_msg, SystemMessage)
        assert isinstance(sys_msg.content, list)
        assert len(sys_msg.content) == 1
        block = sys_msg.content[0]
        assert block["type"] == "text"
        assert block["text"] == "You are a helpful assistant."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_human_message_unchanged(self):
        msgs = [
            SystemMessage(content="System"),
            HumanMessage(content="User input"),
        ]
        result = apply_cache_control(msgs)
        assert result[1].content == "User input"

    def test_already_transformed_left_alone(self):
        already = SystemMessage(content=[{"type": "text", "text": "Cached", "cache_control": {"type": "ephemeral"}}])
        msgs = [already, HumanMessage(content="Hello")]
        result = apply_cache_control(msgs)
        assert result[0] is already

    def test_multiple_system_messages(self):
        msgs = [
            SystemMessage(content="First system"),
            SystemMessage(content="Second system"),
            HumanMessage(content="Query"),
        ]
        result = apply_cache_control(msgs)
        for i in range(2):
            assert isinstance(result[i].content, list)
            assert result[i].content[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_list(self):
        assert apply_cache_control([]) == []


class TestShouldCache:
    def test_enabled_with_claude_model(self):
        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.llm.prompt_caching = True
            assert should_cache(model="claude-3.5-sonnet") is True

    def test_disabled_even_with_claude(self):
        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.llm.prompt_caching = False
            assert should_cache(model="claude-3.5-sonnet") is False

    def test_non_anthropic_model_returns_false(self):
        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.llm.prompt_caching = True
            assert should_cache(model="gpt-4o") is False

    def test_empty_model_and_role(self):
        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.llm.prompt_caching = True
            assert should_cache() is False
