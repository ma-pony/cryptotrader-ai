"""spec 018 — EvolvingMemoryProvider（FR-Z8 / FR-Z9 / FR-Z13 / FR-Z16）。

替换 spec 017a DefaultMemoryProvider；实现 MemoryProvider Protocol。
读 patterns/ 目录 → FSM 过滤 → Pareto 排序 → 返回 markdown 记忆字符串。

I/O helpers（_load_pattern_from_path / _save_pattern_to_path / _load_case_from_path 等）
位于 _io.py，此处 re-export 以保持向后兼容。
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cryptotrader.learning.evolution._io import (
    _extract_agent_analyses,
    _load_case_from_path,
    _load_pattern_from_path,
    _replace_section,
    _save_pattern_to_path,
    _time_decay,
    write_ive_classification,
)

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import CaseRecord, PatternRecord
    from cryptotrader.learning.evolution.fsm import Transition
    from cryptotrader.learning.evolution.ive import FailureClassification

logger = logging.getLogger(__name__)

# Re-export for callers that import from provider (backward compat)
__all__ = [
    "EvolvingMemoryProvider",
    "_extract_agent_analyses",
    "_load_case_from_path",
    "_load_pattern_from_path",
    "_replace_section",
    "_save_pattern_to_path",
    "_time_decay",
]

# 有效 agent IDs（与 spec 014 _constants 一致）
_VALID_AGENTS = ("tech", "chain", "news", "macro")


# ── EvolvingMemoryProvider ─────────────────────────────────────────────────────


class EvolvingMemoryProvider:
    """进化版记忆提供者；实现 spec 017a MemoryProvider Protocol。

    FR-Z8: get_recent_memory 6 步流程（过滤 → Pareto → 评分排序 → top-k → 更新访问计数 → 渲染）
    FR-Z9: 全局容错——内部任一步骤异常 → catch + log warning + return ""
    FR-Z13: evaluate_all_rules() 驱动 FSM 状态转换
    FR-Z16: classify_pending_cases() 对未分类 case 运行 IVE LLM
    """

    def __init__(
        self,
        memory_root: Path | None = None,
        top_k_rules: int = 5,
        top_n_cases: int = 5,
    ) -> None:
        self._root = memory_root or Path("agent_memory")
        self._top_k = top_k_rules
        self._top_n = top_n_cases

    # ── 实现 Protocol ──────────────────────────────────────────────────────────

    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str:
        """FR-Z8: 6 步流程返回格式化 markdown 记忆字符串。

        步骤：
        1. 从 <memory_root>/<agent_id>/patterns/*.md 读所有非 archived/deprecated 规则
        2. 调 pareto.rank_rules(rules) 排序
        3. 二次排序：importance × log(1 + access_count) × time_decay(last_accessed_at)
        4. 取 top-k；写入 access_count / last_accessed_at 回文件
        5. 从 <memory_root>/cases/*.md 读最近 N case（按 timestamp 倒序）
        6. 渲染 markdown
        """
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.set_attribute("memory.provider.type", "EvolvingMemoryProvider")
        except (ImportError, Exception):
            pass

        try:
            effective_k = min(k, self._top_k)
            patterns = self._load_active_patterns(agent_id)
            if not patterns:
                cases = self._load_recent_cases(self._top_n)
                if not cases:
                    return "暂无历史记忆"
                return self._render_memory([], cases)

            # Step 2: Pareto ranking
            from cryptotrader.learning.evolution.pareto import rank_rules

            ranked = rank_rules(patterns)

            # Step 3: secondary sort by importance × log(1+access_count) × time_decay
            now = datetime.now(UTC)
            scored = []
            for rule in ranked:
                decay = _time_decay(rule.last_accessed_at, now)
                score = rule.importance * math.log1p(rule.access_count) * decay
                scored.append((score, rule))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_rules = [r for _, r in scored[:effective_k]]

            # Step 4: update access_count / last_accessed_at
            for rule in top_rules:
                self._update_access(rule)

            # Step 5: load recent cases
            cases = self._load_recent_cases(self._top_n)

            # Step 6: render markdown
            return self._render_memory(top_rules, cases)

        except Exception:
            logger.warning(
                "EvolvingMemoryProvider.get_recent_memory failed for agent=%s, returning ''",
                agent_id,
                exc_info=True,
            )
            return ""

    # ── evaluate_all_rules ─────────────────────────────────────────────────────

    def evaluate_all_rules(self) -> list[Transition]:
        """FR-Z13: 对所有 rule 运行 FSM 状态转换，写回文件，归档 archived 状态 rule。

        Returns list[Transition]（含 rule_id / agent_id / old_state / new_state / triggered_by）。
        """
        from cryptotrader.learning.evolution.fsm import build_transition, evaluate_transitions

        transitions: list[Transition] = []

        for agent_id in _VALID_AGENTS:
            pattern_dir = self._root / agent_id / "patterns"
            if not pattern_dir.exists():
                continue
            for path in sorted(pattern_dir.glob("*.md")):
                try:
                    rule = _load_pattern_from_path(path)
                    if rule is None:
                        continue
                    new_rule = evaluate_transitions(rule)
                    if new_rule is None:
                        continue  # 无转换
                    t = build_transition(rule, new_rule)
                    transitions.append(t)

                    if new_rule.maturity == "archived":
                        self._archive_rule(agent_id, new_rule, path)
                    else:
                        _save_pattern_to_path(new_rule, path)

                except Exception:
                    logger.warning("evaluate_all_rules: error on %s, skipping", path, exc_info=True)

        return transitions

    # ── classify_pending_cases ─────────────────────────────────────────────────

    async def classify_pending_cases(self) -> list[FailureClassification]:
        """FR-Z16: 对 ive_classification==None 的 case 运行 IVE LLM 分类。

        spec 020a FR-Z10: made async to await classify_case (no longer blocks event loop).
        返回 list[FailureClassification]。
        """
        from cryptotrader.learning.evolution.ive import classify_case

        classifications: list[FailureClassification] = []
        cases_dir = self._root / "cases"
        if not cases_dir.exists():
            return []

        for path in sorted(cases_dir.glob("*.md")):
            try:
                case = _load_case_from_path(path)
                if case is None:
                    continue
                if case.ive_classification is not None:
                    continue  # 已分类，跳过

                # spec 020a FR-Z10: await async classify_case
                classification = await classify_case(case)
                classifications.append(classification)

                # 写回 IVE Classification 段
                self._write_ive_classification(path, classification)

                # 更新 fundamental_failure_streak
                if classification.failure_type == "fundamental":
                    self._increment_streak(case)
                else:
                    self._reset_streak(case)

            except Exception:
                logger.warning("classify_pending_cases: error on %s, skipping", path, exc_info=True)

        return classifications

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _load_active_patterns(self, agent_id: str) -> list[PatternRecord]:
        """加载 agent 目录下所有非 archived/deprecated 状态的规则文件。"""
        pattern_dir = self._root / agent_id / "patterns"
        if not pattern_dir.exists():
            return []
        rules = []
        for path in sorted(pattern_dir.glob("*.md")):
            rule = _load_pattern_from_path(path)
            if rule is None:
                continue
            if rule.maturity in ("archived", "deprecated"):
                continue
            rules.append(rule)
        return rules

    def _load_recent_cases(self, n: int) -> list[CaseRecord]:
        """加载最近 N 个 case（按 timestamp 倒序）。"""
        cases_dir = self._root / "cases"
        if not cases_dir.exists():
            return []
        loaded = []
        for path in sorted(cases_dir.glob("*.md")):
            case = _load_case_from_path(path)
            if case is not None:
                loaded.append(case)
        loaded.sort(key=lambda c: c.timestamp, reverse=True)
        return loaded[:n]

    def _update_access(self, rule: PatternRecord) -> None:
        """更新 access_count / last_accessed_at 并写回文件。"""
        try:
            from dataclasses import replace

            now = datetime.now(UTC)
            updated = replace(
                rule,
                access_count=rule.access_count + 1,
                last_accessed_at=now,
            )
            path = rule.file_path
            if path and path.exists():
                _save_pattern_to_path(updated, path)
        except Exception:
            logger.warning("_update_access failed for rule %s", rule.name, exc_info=True)

    def _archive_rule(self, agent_id: str, rule: PatternRecord, source_path: Path) -> None:
        """将 archived 状态的 rule 文件移到 .archived/ 目录。"""
        archived_dir = self._root / agent_id / "patterns" / ".archived"
        archived_dir.mkdir(parents=True, exist_ok=True)
        dest = archived_dir / source_path.name
        # 先写目标文件，再删源文件（伪原子移动）
        _save_pattern_to_path(rule, dest)
        try:
            source_path.unlink()
        except Exception:
            logger.warning("_archive_rule: failed to remove source %s", source_path, exc_info=True)

    def _write_ive_classification(self, case_path: Path, fc: FailureClassification) -> None:
        """将 IVE Classification 写回 case 文件的 ## IVE Classification 段。"""
        write_ive_classification(case_path, fc)

    def _increment_streak(self, case: CaseRecord) -> None:
        """fundamental failure → 对 applied_patterns 中的 rule 增加 fundamental_failure_streak。"""
        try:
            for pattern_ref in case.applied_patterns:
                self._update_rule_streak(pattern_ref, delta=1)
        except Exception:
            logger.warning("_increment_streak failed", exc_info=True)

    def _reset_streak(self, case: CaseRecord) -> None:
        """非 fundamental failure → 重置 applied_patterns 中 rule 的 streak。"""
        try:
            for pattern_ref in case.applied_patterns:
                self._update_rule_streak(pattern_ref, delta=None)
        except Exception:
            logger.warning("_reset_streak failed", exc_info=True)

    def _update_rule_streak(self, pattern_ref: str, delta: int | None) -> None:
        """更新单条 rule 的 fundamental_failure_streak。

        delta=1 → +1；delta=None → reset to 0。
        pattern_ref 格式: "<agent>::<pattern_name>"
        """
        from dataclasses import replace

        if "::" not in pattern_ref:
            return
        agent, name = pattern_ref.split("::", 1)
        path = self._root / agent / "patterns" / f"{name}.md"
        rule = _load_pattern_from_path(path)
        if rule is None:
            return
        new_streak = (rule.fundamental_failure_streak + 1) if delta == 1 else 0
        updated = replace(rule, fundamental_failure_streak=new_streak)
        _save_pattern_to_path(updated, path)

    def _render_memory(
        self,
        rules: list[PatternRecord],
        cases: list[CaseRecord],
    ) -> str:
        """渲染 markdown 记忆字符串（### Patterns + ### Cases）。"""
        parts = []
        if rules:
            rule_lines = []
            for r in rules:
                rule_lines.append(f"- **{r.name}** ({r.maturity}): {r.description}")
            parts.append("### Patterns\n" + "\n".join(rule_lines))
        if cases:
            case_lines = []
            for c in cases:
                pnl_str = f"{c.final_pnl:.2f}" if c.final_pnl is not None else "N/A"
                case_lines.append(f"- [{c.cycle_id}] {c.pair} {c.verdict_action} PnL={pnl_str}")
            parts.append("### Cases\n" + "\n".join(case_lines))
        return "\n\n".join(parts) if parts else "暂无历史记忆"
