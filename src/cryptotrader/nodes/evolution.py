"""spec 018 — evaluate_node：cycle 末段 FSM + IVE 评估节点。

FR-Z22: 在 risk_gate 之后、journal 节点之前触发 evaluate_node。
FR-Z30: 写 OpenTelemetry 6 attribute（memory.evolution.* 命名空间）。
FR-Z22: 异常 catch + log warning + return {}（不修改 state）。
"""

from __future__ import annotations

import logging
import time

from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)


@node_logger()
async def evaluate_node(state: ArenaState) -> dict:
    """cycle 末段评估节点：触发 FSM 状态转换 + IVE failure 分类。

    从 nodes/agents.py 取 module-level _memory_provider（EvolvingMemoryProvider）。
    异常时返回 {}，不修改 state，不阻塞 cycle。
    """
    t0 = time.monotonic()
    try:
        from cryptotrader.nodes.agents import _memory_provider

        if _memory_provider is None:
            logger.debug("evaluate_node: _memory_provider not initialized, skipping")
            return {}

        # FSM 状态转换
        transitions = _memory_provider.evaluate_all_rules()

        # IVE failure 分类
        classifications = _memory_provider.classify_pending_cases()

        duration_ms = (time.monotonic() - t0) * 1000

        # FR-Z30: 写 OpenTelemetry 6 attribute
        _write_telemetry(transitions, classifications, duration_ms)

        logger.debug(
            "evaluate_node: transitions=%d, classifications=%d, duration_ms=%.1f",
            len(transitions),
            len(classifications),
            duration_ms,
        )
        return {}

    except Exception:
        logger.warning("evaluate_node: unexpected error, returning {}", exc_info=True)
        return {}


def _write_telemetry(transitions: list, classifications: list, duration_ms: float) -> None:
    """写 OpenTelemetry span attributes（FR-Z30 6 attribute，memory.evolution.* 命名空间）。"""
    try:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
        except ImportError:
            return
        if span is None or not span.is_recording():
            return

        # FR-Z30 要求的 6 个 telemetry 属性（memory.evolution.* 命名空间）
        # list[dict] 类型序列化为 JSON 字符串（OTel span attribute 不支持 list[dict] 原生类型）
        import json

        fsm_transitions = [
            {
                "rule_id": getattr(t, "rule_id", ""),
                "agent_id": getattr(t, "agent_id", ""),
                "old_state": getattr(t, "old_state", ""),
                "new_state": getattr(t, "new_state", ""),
            }
            for t in transitions
        ]
        ive_classifications = [
            {
                "case_id": getattr(c, "case_id", ""),
                "failure_type": getattr(c, "failure_type", "noise"),
                "agent_id": "",  # CaseRecord does not carry agent_id directly
            }
            for c in classifications
        ]
        archived_rules = [getattr(t, "rule_id", "") for t in transitions if getattr(t, "new_state", "") == "archived"]

        span.set_attribute("memory.evolution.fsm_transitions", json.dumps(fsm_transitions))
        span.set_attribute("memory.evolution.ive_classifications", json.dumps(ive_classifications))
        span.set_attribute("memory.evolution.archived_rules", json.dumps(archived_rules))
        span.set_attribute("memory.evolution.duration_ms", round(duration_ms, 2))
        span.set_attribute("memory.evolution.ive_llm_calls", len(classifications))
        # ive_llm_tokens: not tracked at this layer; set 0 as placeholder (spec 020 will wire LLM token counter)
        span.set_attribute("memory.evolution.ive_llm_tokens", 0)

    except Exception:
        logger.debug("_write_telemetry failed (non-blocking)", exc_info=True)
