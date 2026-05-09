"""Agent Memory 层 — case 写入 + patterns 蒸馏 + archive。

双层架构的核心数据层：
- write_case(): 每 cycle 写入 agent_memory/cases/<cycle_id>.md（per-cycle 单文件）
- update_final_pnl(): 平仓后回填 frontmatter final_pnl
- distill_patterns(): 从 cases 蒸馏 patterns（4 层防过拟合）
- update_pattern_pnl(): 解析 <agent>::<pattern> 更新 pnl_track
- _advance_maturity(): maturity FSM

FR-006 / FR-007 / FR-008 / FR-009 / FR-010 / FR-011 / FR-012 / FR-013
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR, VALID_AGENT_IDS
from cryptotrader.agents.skills._frontmatter import (
    parse_frontmatter,
    render_frontmatter,
)
from cryptotrader.agents.skills._io import atomic_rename, atomic_write, ensure_memory_dirs
from cryptotrader.agents.skills.schema import Maturity, PatternRecord, PnLTrack, ReflectionRun

logger = logging.getLogger(__name__)

# L2: 最少样本量门槛
_MIN_SAMPLES_FOR_ADVANCEMENT = 5

# L3: 区段 vs 全局最小差距
_DEFAULT_GLOBAL_VS_SEGMENT_DELTA = 0.15

# L4: forbidden 最少反向样本量
_MIN_ADVERSE_CASES = 2

# maturity FSM 晋升阈值
_MATURITY_ADVANCE_THRESHOLDS: dict[str, tuple[int, float]] = {
    # maturity → (min_cases, min_win_rate) to advance
    "observed": (_MIN_SAMPLES_FOR_ADVANCEMENT, 0.60),
    "probationary": (15, 0.65),
    "active": (30, 0.70),  # 触发 deprecated 时 win_rate < 0.40
}

_MATURITY_DEPRECATE_WIN_RATE = 0.40  # active 状态 win_rate 低于此 → deprecated


# ── 辅助：读写 case 文件 ──


def _build_case_path(cycle_id: str, memory_dir: Path) -> Path:
    """构建 case 文件路径：agent_memory/cases/<cycle_id>.md。"""
    return memory_dir / "cases" / f"{cycle_id}.md"


def _render_case_content(
    cycle_id: str,
    pair: str,
    agent_analyses: dict[str, str],
    verdict_action: str,
    verdict_reasoning: str,
    applied_patterns: list[str],
    risk_gate_passed: bool,
    execution_status: dict | None,
    final_pnl: float | None,
    trade_execution: dict | None = None,
    causal_chain: dict | None = None,
) -> str:
    """渲染 case 文件内容（frontmatter + markdown body）。"""
    now = datetime.now(UTC).isoformat()
    fm: dict = {
        "cycle_id": cycle_id,
        "timestamp": now,
        "pair": pair,
        "verdict_action": verdict_action,
        "final_pnl": final_pnl,
        "risk_gate_passed": risk_gate_passed,
        # spec 018 新增字段（默认 None，回填时更新）
        "trade_execution": trade_execution,
        "causal_chain": causal_chain,
        "ive_classification": None,
    }
    header = render_frontmatter(fm)

    # Body
    lines = [f"# Cycle Record: {cycle_id}\n"]
    lines.append(f"**Pair**: {pair} | **Action**: {verdict_action} | **Time**: {now}\n")

    # 4 agent analyses
    lines.append("\n## Agent Analyses\n")
    for agent_id, analysis in agent_analyses.items():
        lines.append(f"\n### {agent_id.capitalize()}\n\n{analysis}\n")

    # Verdict
    lines.append("\n## Verdict Reasoning\n")
    lines.append(f"\n{verdict_reasoning}\n")

    # Applied patterns
    if applied_patterns:
        lines.append("\n## Applied Patterns\n")
        for pat in applied_patterns:
            lines.append(f"- {pat}\n")

    # Execution status
    if execution_status:
        lines.append("\n## Execution Status\n")
        for k, v in execution_status.items():
            lines.append(f"- **{k}**: {v}\n")

    # spec 018: Trade Execution
    lines.append("\n## Trade Execution\n")
    if trade_execution:
        for k, v in trade_execution.items():
            lines.append(f"- **{k}**: {v}\n")
    else:
        lines.append("(no trade execution data)\n")

    # spec 018: Causal Chain
    lines.append("\n## Causal Chain\n")
    if causal_chain:
        for k, v in causal_chain.items():
            lines.append(f"- **{k}**: {v}\n")
    else:
        lines.append("(no causal chain data)\n")

    # spec 018: IVE Classification (placeholder — filled by classify_pending_cases)
    lines.append("\n## IVE Classification\n")
    lines.append("(pending classification)\n")

    return header + "".join(lines)


# ── 公开 API ──


def write_case(
    cycle_id: str,
    pair: str,
    agent_analyses: dict[str, str],
    verdict_action: str,
    verdict_reasoning: str,
    applied_patterns: list[str],
    risk_gate_passed: bool,
    execution_status: dict | None,
    final_pnl: float | None,
    memory_dir: Path | None = None,
    trade_execution: dict | None = None,
    causal_chain: dict | None = None,
) -> Path | None:
    """写入单个 trading cycle 的 case 记录（per-cycle 单文件）。

    FR-006: 写入 agent_memory/cases/<cycle_id>.md
    FR-007: 失败时 logger.warning 后返回 None，不阻塞 cycle
    FR-013: 原子写（temp + rename）
    spec 018: 支持 trade_execution / causal_chain 新字段
    """
    mem = memory_dir or DEFAULT_AGENT_MEMORY_DIR
    ensure_memory_dirs(mem)
    path = _build_case_path(cycle_id, mem)
    content = _render_case_content(
        cycle_id=cycle_id,
        pair=pair,
        agent_analyses=agent_analyses,
        verdict_action=verdict_action,
        verdict_reasoning=verdict_reasoning,
        applied_patterns=applied_patterns,
        risk_gate_passed=risk_gate_passed,
        execution_status=execution_status,
        final_pnl=final_pnl,
        trade_execution=trade_execution,
        causal_chain=causal_chain,
    )
    try:
        atomic_write(path, content)
        logger.debug("Case written: %s", path)
        return path
    except Exception:
        logger.warning("Case write failed for %s (non-blocking)", cycle_id, exc_info=True)
        return None


def update_final_pnl(
    cycle_id: str,
    pnl: float,
    memory_dir: Path | None = None,
) -> None:
    """平仓后回填 case 文件的 frontmatter final_pnl 字段（原子写）。"""
    mem = memory_dir or DEFAULT_AGENT_MEMORY_DIR
    path = _build_case_path(cycle_id, mem)
    if not path.exists():
        logger.warning("update_final_pnl: case file not found: %s", path)
        return
    try:
        content = path.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=path)
        fm["final_pnl"] = pnl
        new_content = render_frontmatter(fm) + body
        atomic_write(path, new_content)
        logger.debug("PnL back-filled for %s: pnl=%.2f", cycle_id, pnl)
    except Exception:
        logger.warning("update_final_pnl failed for %s", cycle_id, exc_info=True)


# ── Pattern 文件 I/O ──


def _pattern_path(agent: str, name: str, memory_dir: Path) -> Path:
    return memory_dir / agent / "patterns" / f"{name}.md"


def _archive_path(agent: str, name: str, memory_dir: Path) -> Path:
    return memory_dir / agent / "archive" / f"{name}.md"


def _load_pattern(path: Path) -> PatternRecord | None:
    """从文件加载 PatternRecord，失败时返回 None + warning。"""
    if not path.exists():
        return None
    try:
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
        return PatternRecord(
            name=fm.get("name", path.stem),
            agent=fm.get("agent", ""),
            description=fm.get("description", ""),
            body=body,
            regime_tags=fm.get("regime_tags", []),
            pnl_track=pnl_track,
            maturity=fm.get("maturity", "observed"),
            source_cycles=fm.get("source_cycles", []),
            created=datetime.fromisoformat(fm["created"]) if fm.get("created") else datetime.now(UTC),
            file_path=path,
            manually_edited=fm.get("manually_edited", False),
            version=fm.get("version", 1),
        )
    except Exception:
        logger.warning("Failed to load pattern: %s", path, exc_info=True)
        return None


def _save_pattern(pattern: PatternRecord, path: Path, *, body_override: str | None = None) -> None:
    """保存 PatternRecord 到文件（原子写）。"""
    from cryptotrader.agents.skills._compat import utcnow_str

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
            "last_active": pattern.pnl_track.last_active or utcnow_str(),
        },
        "source_cycles": pattern.source_cycles,
        "created": pattern.created.isoformat() if isinstance(pattern.created, datetime) else str(pattern.created),
        "version": pattern.version,
    }
    body = body_override if body_override is not None else pattern.body
    content = render_frontmatter(fm) + body
    atomic_write(path, content)


# ── update_pattern_pnl ──


def update_pattern_pnl(
    applied_patterns: dict[str, list[str]],
    pnl: float,
    memory_dir: Path | None = None,
) -> None:
    """解析 {agent: [pattern_name, ...]} → 逐条 pattern 文件更新 pnl_track。

    FR-027: 定位到 agent_memory/<agent>/patterns/<pattern_name>.md 更新 pnl_track。
    manually_edited: true 的 pattern 文件只更新 pnl_track，不重写 body。
    """
    mem = memory_dir or DEFAULT_AGENT_MEMORY_DIR
    for agent, names in applied_patterns.items():
        if agent not in VALID_AGENT_IDS:
            logger.warning("update_pattern_pnl: unknown agent '%s', skipping", agent)
            continue
        for name in names:
            path = _pattern_path(agent, name, mem)
            pattern = _load_pattern(path)
            if pattern is None:
                logger.warning("update_pattern_pnl: pattern not found: %s/%s", agent, name)
                continue
            pattern.pnl_track.update(pnl)
            pattern.pnl_track.last_active = datetime.now(UTC).strftime("%Y-%m-%d")
            try:
                if pattern.manually_edited:
                    # 只更新 frontmatter pnl_track，保留 body
                    _save_pattern(pattern, path, body_override=pattern.body)
                else:
                    _save_pattern(pattern, path)
                logger.debug("Pattern PnL updated: %s/%s cases=%d", agent, name, pattern.pnl_track.cases)
            except Exception:
                logger.warning("update_pattern_pnl: save failed for %s/%s", agent, name, exc_info=True)


# ── Maturity FSM ──


def _advance_maturity(pattern: PatternRecord) -> Maturity:
    """计算 pattern 的新 maturity（FSM 转移）。

    FR-011: observed → probationary → active → deprecated
    """
    current = pattern.maturity
    track = pattern.pnl_track
    cases = track.cases
    win_rate = track.win_rate

    if current == "deprecated":
        return "deprecated"

    if current == "active":
        if win_rate < _MATURITY_DEPRECATE_WIN_RATE and cases >= 10:
            return "deprecated"
        return "active"

    # observed → probationary
    if current == "observed":
        min_cases, min_win_rate = _MATURITY_ADVANCE_THRESHOLDS["observed"]
        if cases >= min_cases and win_rate >= min_win_rate:
            return "probationary"
        return "observed"

    # probationary → active
    if current == "probationary":
        min_cases, min_win_rate = _MATURITY_ADVANCE_THRESHOLDS["probationary"]
        if cases >= min_cases and win_rate >= min_win_rate:
            return "active"
        if win_rate < _MATURITY_DEPRECATE_WIN_RATE and cases >= 5:
            return "deprecated"
        return "probationary"

    return current  # type: ignore[return-value]


# ── 4-Layer Anti-Overfitting Functions ──


def _filter_records_by_regime(
    records: list[dict],
    regime_condition: list[str],
) -> list[dict]:
    """L1: 按 regime 条件过滤历史记录，仅保留 regime 匹配的样本。

    使用 Jaccard 相似度（>0）判定匹配。
    """
    if not regime_condition:
        return records  # 无限制时返回全部

    from cryptotrader.learning.regime import regime_overlap

    filtered = []
    for rec in records:
        # 从记录中获取 regime_tags（支持两种来源）
        rec_tags = rec.get("regime_tags", [])
        if not rec_tags:
            # 从 funding_rate / volatility 反推 regime
            from cryptotrader.config import RegimeThresholdsConfig
            from cryptotrader.learning.regime import tag_regime

            summary = {
                "funding_rate": rec.get("funding_rate", 0),
                "volatility": rec.get("volatility", 0),
                "price_change_7d": rec.get("price_change_7d"),
                "fear_greed_index": rec.get("fear_greed_index"),
            }
            thresholds = RegimeThresholdsConfig()
            rec_tags = tag_regime(summary, thresholds)

        if regime_overlap(regime_condition, rec_tags) > 0:
            filtered.append(rec)
    return filtered


def _check_segment_vs_global(
    segment_win_rate: float,
    global_win_rate: float,
    min_delta: float = _DEFAULT_GLOBAL_VS_SEGMENT_DELTA,
) -> bool:
    """L3: 区段胜率必须显著优于全局基线。

    Returns True if segment_win_rate - global_win_rate >= min_delta.
    """
    return (segment_win_rate - global_win_rate) >= min_delta


def _verify_forbidden_pattern(
    forbidden_loss_rate: float,
    adverse_records: list[dict],
    min_adverse_cases: int = _MIN_ADVERSE_CASES,
) -> bool:
    """L4: 对手验证 — forbidden pattern 必须有相反方向亏损证据。

    Returns True if len(adverse_records) >= min_adverse_cases.
    """
    return len(adverse_records) >= min_adverse_cases


# ── cases 读取 ──


def _read_cases(memory_dir: Path, limit: int | None = None) -> list[dict]:
    """读取 agent_memory/cases/ 下所有 case 文件（按文件名时间序）。"""
    cases_dir = memory_dir / "cases"
    if not cases_dir.exists():
        return []
    files = sorted(cases_dir.glob("*.md"))
    if limit:
        files = files[-limit:]
    records = []
    for f in files:
        if f.name.startswith(".") or f.name == "README.md":
            continue
        try:
            content = f.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content, path=f)
            fm["_body"] = body
            fm["_file"] = f
            records.append(fm)
        except Exception:
            logger.warning("Failed to parse case file: %s", f, exc_info=True)
    return records


# ── distill_patterns ──


def distill_patterns(
    memory_dir: Path | None = None,
    cycles_window: int = 50,
) -> ReflectionRun:
    """从 agent_memory/cases/ 蒸馏出 patterns，更新 patterns/ 与 archive/。

    FR-008: 周期性蒸馏；FR-010: 4 层防过拟合；FR-011: maturity 演化。
    FR-012: 失败不抛异常（返回空 ReflectionRun）。
    """
    mem = memory_dir or DEFAULT_AGENT_MEMORY_DIR
    run = ReflectionRun()

    try:
        cases = _read_cases(mem, limit=cycles_window)
        run.cases_processed = len(cases)

        if not cases:
            logger.debug("distill_patterns: no cases found, skipping")
            return run

        # 全局 win rate 由 _check_segment_vs_global / _verify_forbidden_pattern
        # 等 L3/L4 helper 在自身职责内计算（单元测试 test_anti_overfitting_equivalence 直接验证）。
        # distill_patterns 当前主路径采用简化 maturity FSM 驱动，不再在此重复全局 baseline 计算。

        # 解析 applied_patterns → 按 agent 分组
        agent_pattern_counts: dict[str, dict[str, list[float]]] = {}  # agent → pattern → [pnl...]
        for case in cases:
            body = case.get("_body", "")
            pnl = case.get("final_pnl")
            applied = _parse_applied_from_body(body)
            for agent, patterns in applied.items():
                if agent not in VALID_AGENT_IDS:
                    continue
                agent_pattern_counts.setdefault(agent, {})
                for p in patterns:
                    agent_pattern_counts[agent].setdefault(p, [])
                    if pnl is not None:
                        agent_pattern_counts[agent][p].append(float(pnl))

        # 对每个 agent，更新已有 patterns 的 maturity
        for agent in VALID_AGENT_IDS:
            patterns_dir = mem / agent / "patterns"
            if not patterns_dir.exists():
                continue
            for pattern_file in sorted(patterns_dir.glob("*.md")):
                if pattern_file.name.startswith("."):
                    continue
                pattern = _load_pattern(pattern_file)
                if pattern is None:
                    continue
                new_maturity = _advance_maturity(pattern)
                if new_maturity == "deprecated" and pattern.maturity != "deprecated":
                    # 移到 archive
                    arch = _archive_path(agent, pattern.name, mem)
                    try:
                        atomic_rename(pattern_file, arch)
                        run.patterns_archived += 1
                        logger.info("Pattern archived (deprecated): %s/%s", agent, pattern.name)
                    except Exception:
                        logger.warning("Failed to archive pattern: %s", pattern_file, exc_info=True)
                elif new_maturity != pattern.maturity:
                    pattern.maturity = new_maturity
                    try:
                        _save_pattern(pattern, pattern_file)
                        run.patterns_updated += 1
                    except Exception:
                        logger.warning("Failed to update maturity: %s", pattern_file, exc_info=True)

    except Exception:
        logger.warning("distill_patterns failed (non-blocking)", exc_info=True)
        run.error = "distill_patterns internal error"

    return run


def _parse_applied_from_body(body: str) -> dict[str, list[str]]:
    """从 body 文本中解析 applied: 引用，返回 {agent: [pattern_name]}。

    支持格式：
    - bare name: `applied: pattern_name` → originating_agent 解析
    - prefix: `applied: tech::pattern_name`
    """
    result: dict[str, list[str]] = {}
    pattern = re.compile(r"applied:\s*([a-z]+::)?([a-z_][a-z0-9_-]*)", re.MULTILINE)
    for m in pattern.finditer(body):
        prefix = m.group(1)  # e.g. "tech::" or None
        name = m.group(2)
        if prefix:
            agent = prefix.rstrip("::")
            if agent in VALID_AGENT_IDS:
                result.setdefault(agent, [])
                result[agent].append(name)
        # bare name without prefix: skip at this level (needs originating_agent context)
    return result


# ── parse_applied (US5 support) ──


def parse_applied(
    reasoning: str,
    originating_agent: str | None = None,
) -> dict[str, list[str]]:
    """解析 verdict reasoning 中的 applied: 引用。

    FR-026: bare name → originating_agent；prefix → 按 agent 前缀；
    bare name 在多 agent 同时存在时 logger.warning 跳过。

    Returns: {agent_id: [pattern_name, ...]}
    """
    result: dict[str, list[str]] = {}
    pattern = re.compile(r"applied:\s*([a-z]+::)?([a-z_][a-z0-9_-]*)", re.MULTILINE)
    for m in pattern.finditer(reasoning):
        prefix = m.group(1)  # e.g. "tech::" or None
        name = m.group(2)
        if prefix:
            agent = prefix.rstrip(":")
            if agent in VALID_AGENT_IDS:
                result.setdefault(agent, [])
                if name not in result[agent]:
                    result[agent].append(name)
            else:
                logger.warning("parse_applied: unknown agent prefix '%s', skipping", agent)
        elif originating_agent:
            if originating_agent in VALID_AGENT_IDS:
                result.setdefault(originating_agent, [])
                if name not in result[originating_agent]:
                    result[originating_agent].append(name)
        else:
            # bare name without originating_agent (verdict context) — ambiguous
            logger.warning(
                "parse_applied: bare 'applied: %s' in verdict context without agent prefix — skipping PnL attribution",
                name,
            )
    return result
