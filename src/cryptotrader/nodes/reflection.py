"""Reflection 节点 — 薄包装 learning/memory.py 的反思流程为 graph 节点。

FR-008: 按 every_n_cycles 周期性触发 distill_patterns()。
FR-012: 反思失败不阻塞下一个 trading cycle。
"""

from __future__ import annotations

import logging

from cryptotrader.state import ArenaState
from cryptotrader.tracing import node_logger

logger = logging.getLogger(__name__)


@node_logger()
async def run_reflection(state: ArenaState) -> dict:
    """Graph 节点：触发 memory 蒸馏（FR-008）。

    捕获所有异常后 logger.exception，返回 unchanged state（FR-012）。
    """
    try:
        from cryptotrader.config import load_config
        from cryptotrader.learning.memory import distill_patterns

        config = load_config()
        exp_cfg = config.experience
        cycle_count = state.get("metadata", {}).get("cycle_count", 0)

        if not exp_cfg.enabled:
            logger.debug("Reflection skipped: experience not enabled")
            return {}

        if cycle_count <= 0 or cycle_count % exp_cfg.every_n_cycles != 0:
            logger.debug(
                "Reflection skipped: cycle_count=%d, every_n_cycles=%d",
                cycle_count,
                exp_cfg.every_n_cycles,
            )
            return {}

        logger.info("Running reflection at cycle %d", cycle_count)
        run = distill_patterns(cycles_window=exp_cfg.lookback_commits)
        logger.info(
            "Reflection complete: cases=%d patterns_updated=%d archived=%d",
            run.cases_processed,
            run.patterns_updated,
            run.patterns_archived,
        )
    except Exception:
        # FR-012: 不阻塞下一个 cycle
        logger.exception("Reflection node failed (non-blocking)")

    return {}
