"""Tests for spec 021 reasoning_content round-trip compat patch."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai.chat_models import base as _lc_oai_base

import cryptotrader.llm.reasoning_compat  # noqa: F401  (applies patch)


def test_patch_idempotent():
    """重复调用 apply_patch 不应重复包装。"""
    from cryptotrader.llm import reasoning_compat

    before = _lc_oai_base._convert_message_to_dict
    reasoning_compat.apply_patch()
    reasoning_compat.apply_patch()
    after = _lc_oai_base._convert_message_to_dict
    assert before is after, "重复 apply_patch 应保持函数引用不变"


def test_incoming_reasoning_content_preserved():
    """入站：响应 dict 含 reasoning_content → AIMessage.additional_kwargs 应有该字段。"""
    raw = {
        "role": "assistant",
        "content": "answer text",
        "reasoning_content": "the model is thinking step by step about X",
    }
    msg = _lc_oai_base._convert_dict_to_message(raw)
    assert isinstance(msg, AIMessage)
    assert msg.content == "answer text"
    assert msg.additional_kwargs.get("reasoning_content") == "the model is thinking step by step about X"


def test_incoming_no_reasoning_unchanged():
    """入站：没有 reasoning_content 的响应不应注入该 key。"""
    raw = {"role": "assistant", "content": "plain answer"}
    msg = _lc_oai_base._convert_dict_to_message(raw)
    assert isinstance(msg, AIMessage)
    assert "reasoning_content" not in msg.additional_kwargs


def test_outgoing_reasoning_content_round_trips():
    """出站：AIMessage.additional_kwargs.reasoning_content → outgoing dict 应保留。"""
    ai = AIMessage(
        content="answer text",
        additional_kwargs={"reasoning_content": "previous thought chain ..."},
    )
    out = _lc_oai_base._convert_message_to_dict(ai)
    assert out["role"] == "assistant"
    assert out["content"] == "answer text"
    assert out["reasoning_content"] == "previous thought chain ..."


def test_outgoing_no_reasoning_unchanged():
    """出站：没有 reasoning_content 的 AIMessage 不应出现该 key。"""
    ai = AIMessage(content="plain answer")
    out = _lc_oai_base._convert_message_to_dict(ai)
    assert "reasoning_content" not in out


def test_outgoing_human_message_untouched():
    """出站：HumanMessage 与 reasoning_content 无关，不应被影响。"""
    h = HumanMessage(content="hello")
    out = _lc_oai_base._convert_message_to_dict(h)
    assert out["role"] == "user"
    assert "reasoning_content" not in out


def test_round_trip_full_path():
    """端到端：dict → message → dict 保留 reasoning_content。"""
    src = {
        "role": "assistant",
        "content": "final",
        "reasoning_content": "chain-of-thought block",
    }
    msg = _lc_oai_base._convert_dict_to_message(src)
    out = _lc_oai_base._convert_message_to_dict(msg)
    assert out["reasoning_content"] == "chain-of-thought block"
