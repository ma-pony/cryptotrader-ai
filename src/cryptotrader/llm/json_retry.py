"""JSON parse retry — strip markdown fences, re-ask LLM with schema hint."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


@dataclass
class JsonParseRetryContext:
    raw_text: str
    error_msg: str
    schema_hint: str
    attempt: int


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences, returning inner content."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _try_parse(text: str) -> dict | None:
    """Try parsing text as JSON, returning None on failure."""
    try:
        stripped = _strip_markdown_fences(text)
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        from cryptotrader.debate.verdict import _extract_json

        return _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        pass

    return None


async def extract_json_with_retry(
    text: str,
    llm: BaseChatModel | None = None,
    schema_hint: str = "",
    max_retries: int = 5,
    original_messages: list[BaseMessage] | None = None,
) -> dict:
    """Extract JSON from text, retrying with LLM if parsing fails.

    1. Try _strip_markdown_fences + json.loads / _extract_json (no retry count)
    2. If llm provided and max_retries > 0: ask LLM to fix the JSON
    3. Final fallback: return empty dict and log warning
    """
    result = _try_parse(text)
    if result is not None:
        return result

    if llm is None or max_retries == 0:
        _slog.warning("json_parse_exhausted", schema_hint=schema_hint[:100], text_preview=text[:200])
        return {}

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    for attempt in range(1, max_retries + 1):
        fix_prompt = _build_fix_prompt(text, schema_hint, attempt)
        try:
            msgs: list[Any] = [
                SystemMessage(content="You are a JSON repair assistant. Output ONLY valid JSON."),
                HumanMessage(content=fix_prompt),
            ]
            resp = await llm.ainvoke(msgs)
            resp_text = resp.content if isinstance(resp, AIMessage) else str(resp)
            result = _try_parse(resp_text)
            if result is not None:
                _slog.info("json_parse_retry_success", attempt=attempt)
                return result
        except Exception:
            logger.debug("JSON retry attempt %d failed", attempt, exc_info=True)

    _slog.warning("json_parse_exhausted", schema_hint=schema_hint[:100], attempts=max_retries)
    return {}


def _build_fix_prompt(raw_text: str, schema_hint: str, attempt: int) -> str:
    """Build a fix prompt without including raw LLM output content."""
    parts = [f"The previous response was not valid JSON (attempt {attempt})."]
    if schema_hint:
        parts.append(f"Expected schema fields: {schema_hint}")
    parts.append("Please output ONLY a valid JSON object with the required fields.")
    return "\n".join(parts)
