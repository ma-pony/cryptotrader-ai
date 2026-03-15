"""可观测性指标模块 - MetricsCollector 单例, 基于 prometheus-client。"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# -- 指标定义 --

_ct_llm_calls_total = Counter(
    "ct_llm_calls_total",
    "LLM 调用总次数",
    ["model", "node"],
)

_ct_debate_skipped_total = Counter(
    "ct_debate_skipped_total",
    "辩论环节被跳过的总次数",
)

_ct_verdict_total = Counter(
    "ct_verdict_total",
    "裁决结果总次数",
    ["action"],
)

_ct_risk_rejected_total = Counter(
    "ct_risk_rejected_total",
    "风控拒绝总次数",
    ["check_name"],
)

_ct_trade_executed_total = Counter(
    "ct_trade_executed_total",
    "交易执行总次数",
    ["engine", "side"],
)

_ct_execution_latency_ms = Histogram(
    "ct_execution_latency_ms",
    "交易执行延迟(毫秒)",
    ["engine"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
)

_ct_pipeline_duration_ms = Histogram(
    "ct_pipeline_duration_ms",
    "完整流水线执行时长(毫秒)",
    buckets=[500, 1000, 2000, 5000, 10000, 30000, 60000],
)


# -- MetricsCollector --


class MetricsCollector:
    """系统核心指标收集器, 封装 prometheus-client 计数器与直方图。"""

    def inc_llm_calls(self, *, model: str, node: str) -> None:
        """递增 LLM 调用计数器。"""
        try:
            _ct_llm_calls_total.labels(model=model, node=node).inc()
        except Exception:
            logger.warning("递增 ct_llm_calls_total 失败", exc_info=True)

    def inc_debate_skipped(self) -> None:
        """递增辩论跳过计数器。"""
        try:
            _ct_debate_skipped_total.inc()
        except Exception:
            logger.warning("递增 ct_debate_skipped_total 失败", exc_info=True)

    def inc_verdict(self, *, action: str) -> None:
        """递增裁决计数器。"""
        try:
            _ct_verdict_total.labels(action=action).inc()
        except Exception:
            logger.warning("递增 ct_verdict_total 失败", exc_info=True)

    def inc_risk_rejected(self, *, check_name: str) -> None:
        """递增风控拒绝计数器。"""
        try:
            _ct_risk_rejected_total.labels(check_name=check_name).inc()
        except Exception:
            logger.warning("递增 ct_risk_rejected_total 失败", exc_info=True)

    def inc_trade_executed(self, *, engine: str, side: str) -> None:
        """递增交易执行计数器。"""
        try:
            _ct_trade_executed_total.labels(engine=engine, side=side).inc()
        except Exception:
            logger.warning("递增 ct_trade_executed_total 失败", exc_info=True)

    def observe_execution_latency(self, *, engine: str, ms: float) -> None:
        """记录交易执行延迟观测值。"""
        try:
            _ct_execution_latency_ms.labels(engine=engine).observe(ms)
        except Exception:
            logger.warning("记录 ct_execution_latency_ms 失败", exc_info=True)

    def observe_pipeline_duration(self, *, ms: float) -> None:
        """记录流水线执行时长观测值。"""
        try:
            _ct_pipeline_duration_ms.observe(ms)
        except Exception:
            logger.warning("记录 ct_pipeline_duration_ms 失败", exc_info=True)


# -- 模块级单例 --

_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """返回 MetricsCollector 单例, 线程安全(GIL 保证)。"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
