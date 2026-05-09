"""spec 018 — EvolvingMemoryProvider（FR-Z8 / FR-Z9 / FR-Z13 / FR-Z16）。

替换 spec 017a DefaultMemoryProvider；实现 MemoryProvider Protocol。
读 patterns/ 目录 → FSM 过滤 → Pareto 排序 → 返回 markdown 记忆字符串。
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import CaseRecord, PatternRecord
    from cryptotrader.learning.evolution.fsm import Transition
    from cryptotrader.learning.evolution.ive import FailureClassification

logger = logging.getLogger(__name__)

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

    def classify_pending_cases(self) -> list[FailureClassification]:
        """FR-Z16: 对 ive_classification==None 的 case 运行 IVE LLM 分类。

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

                classification = classify_case(case)
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
        try:
            from cryptotrader.agents.skills._frontmatter import parse_frontmatter, render_frontmatter
            from cryptotrader.agents.skills._io import atomic_write

            content = case_path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content, path=case_path)
            fm["ive_classification"] = {
                "failure_type": fc.failure_type,
                "confidence": fc.confidence,
                "reasoning": fc.reasoning,
            }

            # 替换或追加 ## IVE Classification 段
            ive_section = (
                "\n## IVE Classification\n\n"
                f"- **Failure Type**: {fc.failure_type}\n"
                f"- **Confidence**: {fc.confidence:.2f}\n"
                f"- **Reasoning**: {fc.reasoning}\n"
            )
            if "## IVE Classification" in body:
                # 替换已有段
                body = _replace_section(body, "IVE Classification", ive_section)
            else:
                body = body.rstrip() + "\n" + ive_section

            new_content = render_frontmatter(fm) + body
            atomic_write(case_path, new_content)
        except Exception:
            logger.warning("_write_ive_classification failed for %s", case_path, exc_info=True)

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


# ── 模块级辅助函数 ─────────────────────────────────────────────────────────────


def _time_decay(last_accessed: datetime, now: datetime) -> float:
    """计算时间衰减因子：exp(-days/30)，days 为距 last_accessed 的天数。"""
    try:
        delta = now - last_accessed
        days = max(0.0, delta.total_seconds() / 86400)
        return math.exp(-days / 30)
    except Exception:
        return 1.0


def _load_pattern_from_path(path: Path) -> PatternRecord | None:
    """从文件加载 PatternRecord，包含 spec 018 新增字段。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack

        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)
        pt = fm.get("pnl_track", {})
        pnl_track = PnLTrack(
            cases=pt.get("cases", 0),
            wins=pt.get("wins", 0),
            win_rate=pt.get("win_rate", 0.0),
            avg_pnl=pt.get("avg_pnl", 0.0),
            last_active=pt.get("last_active", ""),
        )
        # spec 018 新字段
        last_accessed_raw = fm.get("last_accessed_at")
        last_modified_raw = fm.get("last_modified_at")
        now = datetime.now(UTC)

        def _parse_dt(raw: str | None) -> datetime:
            if not raw:
                return now
            try:
                dt = datetime.fromisoformat(str(raw))
                # 确保 timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except (ValueError, TypeError):
                return now

        return PatternRecord(
            name=fm.get("name", path.stem),
            agent=fm.get("agent", ""),
            description=fm.get("description", ""),
            body=body,
            regime_tags=fm.get("regime_tags", []),
            pnl_track=pnl_track,
            maturity=fm.get("maturity", "observed"),
            source_cycles=fm.get("source_cycles", []),
            created=_parse_dt(fm.get("created")),
            file_path=path,
            manually_edited=fm.get("manually_edited", False),
            version=fm.get("version", 1),
            importance=float(fm.get("importance", 0.5)),
            access_count=int(fm.get("access_count", 0)),
            last_accessed_at=_parse_dt(last_accessed_raw),
            last_modified_at=_parse_dt(last_modified_raw),
            fundamental_failure_streak=int(fm.get("fundamental_failure_streak", 0)),
        )
    except Exception:
        logger.warning("_load_pattern_from_path failed for %s", path, exc_info=True)
        return None


def _save_pattern_to_path(pattern: PatternRecord, path: Path) -> None:
    """保存 PatternRecord（含 spec 018 新字段）到文件（原子写）。"""
    from cryptotrader.agents.skills._frontmatter import render_frontmatter
    from cryptotrader.agents.skills._io import atomic_write

    fm: dict = {
        "name": pattern.name,
        "agent": pattern.agent,
        "description": pattern.description,
        "maturity": pattern.maturity,
        "manually_edited": pattern.manually_edited,
        "regime_tags": pattern.regime_tags,
        "pnl_track": {
            "cases": pattern.pnl_track.cases,
            "wins": pattern.pnl_track.wins,
            "win_rate": round(pattern.pnl_track.win_rate, 4),
            "avg_pnl": round(pattern.pnl_track.avg_pnl, 4),
            "last_active": pattern.pnl_track.last_active,
        },
        "source_cycles": pattern.source_cycles,
        "created": pattern.created.isoformat() if isinstance(pattern.created, datetime) else str(pattern.created),
        "version": pattern.version,
        # spec 018 新字段
        "importance": pattern.importance,
        "access_count": pattern.access_count,
        "last_accessed_at": pattern.last_accessed_at.isoformat()
        if isinstance(pattern.last_accessed_at, datetime)
        else str(pattern.last_accessed_at),
        "last_modified_at": pattern.last_modified_at.isoformat()
        if isinstance(pattern.last_modified_at, datetime)
        else str(pattern.last_modified_at),
        "fundamental_failure_streak": pattern.fundamental_failure_streak,
    }
    content = render_frontmatter(fm) + pattern.body
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, content)


def _load_case_from_path(path: Path) -> CaseRecord | None:
    """从文件加载 CaseRecord，包含 spec 018 新增字段。"""
    try:
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter
        from cryptotrader.agents.skills.schema import CaseRecord

        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)

        raw_ts = fm.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(raw_ts)) if raw_ts else datetime.now(UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = datetime.now(UTC)

        # Parse agent_analyses from body sections
        agent_analyses = _extract_agent_analyses(body)

        return CaseRecord(
            cycle_id=fm.get("cycle_id", path.stem),
            timestamp=ts,
            pair=fm.get("pair", ""),
            verdict_action=fm.get("verdict_action", "hold"),
            final_pnl=fm.get("final_pnl"),
            agent_analyses=agent_analyses,
            applied_patterns=fm.get("applied_patterns", []),
            risk_gate_passed=fm.get("risk_gate_passed", True),
            file_path=path,
            trade_execution=fm.get("trade_execution"),
            causal_chain=fm.get("causal_chain"),
            ive_classification=fm.get("ive_classification"),
        )
    except Exception:
        logger.warning("_load_case_from_path failed for %s", path, exc_info=True)
        return None


def _extract_agent_analyses(body: str) -> dict[str, str]:
    """从 case markdown body 提取 agent analyses（### AgentId 段落）。"""
    result: dict[str, str] = {}
    current_agent: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("### ") and not line.startswith("#### "):
            if current_agent is not None:
                result[current_agent.lower()] = "\n".join(current_lines).strip()
            current_agent = line[4:].strip().lower()
            current_lines = []
        elif current_agent is not None and not line.startswith("## "):
            current_lines.append(line)
        elif line.startswith("## ") and current_agent is not None:
            result[current_agent.lower()] = "\n".join(current_lines).strip()
            current_agent = None
            current_lines = []
    if current_agent is not None:
        result[current_agent.lower()] = "\n".join(current_lines).strip()
    return result


def _replace_section(body: str, section_title: str, new_section: str) -> str:
    """替换 body 中 ## <section_title> 段落为 new_section。"""
    import re

    pattern = re.compile(
        rf"^## {re.escape(section_title)}\n.*?(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    replaced, count = pattern.subn(new_section, body)
    if count == 0:
        return body + "\n" + new_section
    return replaced
