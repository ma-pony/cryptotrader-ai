"""Prompt caching for Anthropic Claude models.

Adds ``cache_control`` breakpoints to static system prompts, enabling
~90% input-token discount when the same prefix is sent within 5 minutes.
Transparent no-op for non-Anthropic providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


def is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


def _is_role_anthropic(role: str) -> bool:
    try:
        from cryptotrader.llm.registry import load_manifest

        manifest = load_manifest()
        if manifest is None:
            return False
        role_cfg = manifest.get_role(role)
        if role_cfg is None or not role_cfg.provider_chain:
            return False
        entry = manifest.get_provider(role_cfg.provider_chain[0])
        return entry is not None and entry.provider_type == "anthropic"
    except Exception:
        return False


def should_cache(model: str = "", role: str = "") -> bool:
    """Return True when prompt caching should be applied."""
    from cryptotrader.config import load_config

    cfg = load_config()
    if not cfg.llm.prompt_caching:
        return False
    if model and is_anthropic_model(model):
        return True
    return bool(role and _is_role_anthropic(role))


def apply_cache_control(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Add ``cache_control`` to every ``SystemMessage`` with string content.

    Transforms ``SystemMessage(content="...")`` into content-block format::

        SystemMessage(content=[{"type": "text", "text": "...",
                                "cache_control": {"type": "ephemeral"}}])

    ``HumanMessage`` and already-transformed messages are left unchanged.
    """
    from langchain_core.messages import SystemMessage

    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
            result.append(
                SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": msg.content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                )
            )
        else:
            result.append(msg)
    return result
