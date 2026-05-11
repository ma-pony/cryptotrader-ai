"""DeepSeek / 思维链模型兼容层（reasoning_content round-trip）。

LangChain `ChatOpenAI` 在 `_convert_dict_to_message` 不抽取 `reasoning_content`，
`_convert_message_to_dict` 也不回传 — 详见 langchain_openai/chat_models/base.py:8
的官方告示。

对于走 DeepSeek-v4 / GLM-4 / vLLM thinking 等思维链模型 + 多轮 tool-calling
路径（ToolAgent / `langchain.agents.create_agent`）：
  1. 第 1 轮模型返回 `reasoning_content` + `tool_calls`
  2. 工具执行后 LangChain 把 history 回传到模型
  3. AIMessage 序列化丢失 reasoning_content
  4. 上游 API 检查到 history 缺字段 → 400 BadRequest
     `"The reasoning_content in the thinking mode must be passed back to the API."`

修复：monkey-patch `langchain_openai.chat_models.base` 的两个函数：
  - 入站 (`_convert_dict_to_message`)：把 reasoning_content 收进 additional_kwargs
  - 出站 (`_convert_message_to_dict`)：把 additional_kwargs.reasoning_content 写回 dict

幂等：模块加载即生效；调用 `apply_patch()` 也安全（标记位防重）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai.chat_models import base as _lc_oai_base

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_cryptotrader_reasoning_patch_applied"


def _wrap_convert_dict_to_message(original):
    """入站：收集 reasoning_content → additional_kwargs。"""

    def wrapped(_dict: Mapping[str, Any]) -> BaseMessage:
        msg = original(_dict)
        if isinstance(msg, AIMessage) and "reasoning_content" in _dict:
            rc = _dict.get("reasoning_content")
            if rc:
                msg.additional_kwargs["reasoning_content"] = rc
        return msg

    return wrapped


def _wrap_convert_message_to_dict(original):
    """出站：把 additional_kwargs.reasoning_content 写回 message dict。"""

    def wrapped(message: BaseMessage, api: str = "chat/completions") -> dict[str, Any]:
        out = original(message, api=api)
        if isinstance(message, AIMessage):
            rc = message.additional_kwargs.get("reasoning_content")
            if rc:
                out["reasoning_content"] = rc
        return out

    return wrapped


def apply_patch() -> None:
    """Monkey-patch langchain_openai 的序列化 / 反序列化以保留 reasoning_content。

    幂等：重复调用安全。
    """
    if getattr(_lc_oai_base, _PATCH_FLAG, False):
        return

    _lc_oai_base._convert_dict_to_message = _wrap_convert_dict_to_message(  # type: ignore[attr-defined]
        _lc_oai_base._convert_dict_to_message
    )
    _lc_oai_base._convert_message_to_dict = _wrap_convert_message_to_dict(  # type: ignore[attr-defined]
        _lc_oai_base._convert_message_to_dict
    )
    setattr(_lc_oai_base, _PATCH_FLAG, True)
    logger.debug("Applied langchain_openai reasoning_content round-trip patch")


# 模块加载即应用，确保任何 `from langchain_openai import ChatOpenAI` 之前生效。
apply_patch()
