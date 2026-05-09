"""spec 018 — evaluate_node：cycle 末段 FSM + IVE 评估节点。

FR-Z29: 在 risk_gate 之后、journal 节点之前触发 evaluate_node。
FR-Z30: 写 OpenTelemetry 6 attribute。
FR-Z31: 异常 catch + log warning + return {}（不修改 state）。
"""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)


@node_logger()
async def evaluate_node(state: ArenaState) -> dict:
    """cycle 末段评估节点：触发 FSM 状态转换 + IVE failure 分类。

    从 nodes/agents.py 取 module-level _memory_provider（EvolvingMemoryProvider）。
    异常时返回 {}，不修改 state，不阻塞 cycle。
    """
    try:
        from cryptotrader.nodes.agents import _memory_provider

        if _memory_provider is None:
            logger.debug("evaluate_node: _memory_provider not initialized, skipping")
            return {}

        # FSM 状态转换
        transitions = _memory_provider.evaluate_all_rules()

        # IVE failure 分类
        classifications = _memory_provider.classify_pending_cases()

        # FR-Z30: 写 OpenTelemetry 6 attribute
        _write_telemetry(transitions, classifications)

        logger.debug(
            "evaluate_node: transitions=%d, classifications=%d",
            len(transitions),
            len(classifications),
        )
        return {}

    except Exception:
        logger.warning("evaluate_node: unexpected error, returning {}", exc_info=True)
        return {}


def _write_telemetry(transitions: list, classifications: list) -> None:
    """写 OpenTelemetry span attributes（FR-Z30 6 attribute）。"""
    try:
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
        except ImportError:
            return
        if span is None or not span.is_recording():
            return

        # 6 个 telemetry 属性
        span.set_attribute("memory.evaluate.transitions_total", len(transitions))
        span.set_attribute("memory.evaluate.classifications_total", len(classifications))

        fundamental = sum(1 for c in classifications if getattr(c, "failure_type", "") == "fundamental")
        implementation = sum(1 for c in classifications if getattr(c, "failure_type", "") == "implementation")
        noise = sum(1 for c in classifications if getattr(c, "failure_type", "") == "noise")
        archived = sum(1 for t in transitions if getattr(t, "new_state", "") == "archived")

        span.set_attribute("memory.evaluate.fundamental_failures", fundamental)
        span.set_attribute("memory.evaluate.implementation_failures", implementation)
        span.set_attribute("memory.evaluate.noise_classifications", noise)
        span.set_attribute("memory.evaluate.rules_archived", archived)

    except Exception:
        logger.debug("_write_telemetry failed (non-blocking)", exc_info=True)
