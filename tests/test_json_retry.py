"""Tests for JSON parse retry — fence stripping, LLM repair, fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from cryptotrader.llm.json_retry import (
    _strip_markdown_fences,
    _try_parse,
    extract_json_with_retry,
)


class TestStripMarkdownFences:
    def test_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_no_fence(self):
        text = '{"key": "value"}'
        assert _strip_markdown_fences(text) == '{"key": "value"}'

    def test_surrounding_text(self):
        text = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
        assert _strip_markdown_fences(text) == '{"a": 1}'


class TestTryParse:
    def test_valid_json(self):
        assert _try_parse('{"direction": "bullish"}') == {"direction": "bullish"}

    def test_fenced_json(self):
        text = '```json\n{"direction": "bearish"}\n```'
        assert _try_parse(text) == {"direction": "bearish"}

    def test_invalid_returns_none(self):
        assert _try_parse("not json at all") is None

    def test_array_returns_none(self):
        assert _try_parse("[1, 2, 3]") is None

    def test_embedded_json(self):
        text = 'Some text {"direction": "neutral", "confidence": 0.5} more text'
        result = _try_parse(text)
        assert result is not None
        assert result["direction"] == "neutral"


class TestExtractJsonWithRetry:
    async def test_direct_parse_success(self):
        result = await extract_json_with_retry('{"direction": "bullish", "confidence": 0.8}')
        assert result["direction"] == "bullish"

    async def test_fenced_parse_success(self):
        text = '```json\n{"action": "long"}\n```'
        result = await extract_json_with_retry(text)
        assert result["action"] == "long"

    async def test_no_llm_returns_empty_on_failure(self):
        result = await extract_json_with_retry("totally invalid", llm=None)
        assert result == {}

    async def test_llm_retry_succeeds(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"fixed": true}'))

        result = await extract_json_with_retry(
            "broken json {{",
            llm=mock_llm,
            schema_hint="fixed",
            max_retries=2,
        )
        assert result == {"fixed": True}
        mock_llm.ainvoke.assert_called_once()

    async def test_llm_retry_exhausted(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="still not json"))

        result = await extract_json_with_retry(
            "broken",
            llm=mock_llm,
            max_retries=2,
        )
        assert result == {}
        assert mock_llm.ainvoke.call_count == 2

    async def test_max_retries_zero(self):
        result = await extract_json_with_retry("broken", llm=MagicMock(), max_retries=0)
        assert result == {}

    async def test_llm_exception_during_retry(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))

        result = await extract_json_with_retry(
            "broken",
            llm=mock_llm,
            max_retries=1,
        )
        assert result == {}
