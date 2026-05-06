"""SKILL.md 整理（Curation）— 手工 + LLM 触发。

FR-015: SKILL.md 不在每 cycle 自动更新，是 curation 独立流程产物。
FR-016: CLI `arena skills curate <name> [--llm]` 触发整理。
FR-017: manually_edited: true 整体跳过；含 AUTO-DISTILLED-PATTERNS 标记时仅替换该区段。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR, DEFAULT_AGENT_SKILLS_DIR, VALID_AGENT_IDS
from cryptotrader.agents.skills._frontmatter import (
    parse_frontmatter,
    render_frontmatter,
)
from cryptotrader.agents.skills._io import atomic_write

logger = logging.getLogger(__name__)

_AUTO_DISTILLED_START = "<!-- AUTO-DISTILLED-PATTERNS -->"
_AUTO_DISTILLED_END = "<!-- END-AUTO-DISTILLED-PATTERNS -->"


def _get_agent_for_skill(skill_name: str, skills_dir: Path) -> str | None:
    """从 SKILL.md frontmatter scope 字段推断对应的 agent_id。"""
    skill_file = skills_dir / skill_name / "SKILL.md"
    if not skill_file.exists():
        return None
    try:
        content = skill_file.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(content, path=skill_file)
        scope = fm.get("scope", "")
        if scope.startswith("agent:"):
            agent = scope.split(":", 1)[1]
            return agent if agent in VALID_AGENT_IDS else None
    except Exception:
        logger.warning("Failed to read scope for skill '%s'", skill_name, exc_info=True)
    return None


def _load_active_patterns(agent: str, memory_dir: Path) -> list[dict]:
    """读取 agent_memory/<agent>/patterns/ 下所有 active 状态的 pattern 文件。"""
    patterns_dir = memory_dir / agent / "patterns"
    if not patterns_dir.exists():
        return []
    result = []
    for f in sorted(patterns_dir.glob("*.md")):
        if f.name.startswith("."):
            continue
        try:
            content = f.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content, path=f)
            if fm.get("maturity") == "active":
                result.append(
                    {"name": fm.get("name", f.stem), "description": fm.get("description", ""), "body": body, "fm": fm}
                )
        except Exception:
            logger.warning("Failed to load pattern %s", f, exc_info=True)
    return result


def _build_auto_distilled_section(patterns: list[dict]) -> str:
    """构建 AUTO-DISTILLED-PATTERNS 区段内容。"""
    if not patterns:
        return "(No active patterns yet)\n"
    lines = []
    for p in patterns:
        name = p["name"]
        desc = p["description"] or ""
        fm = p.get("fm", {})
        wr = fm.get("pnl_track", {}).get("win_rate", 0)
        cases = fm.get("pnl_track", {}).get("cases", 0)
        lines.append(f"- **{name}**: {desc} (win_rate={wr:.0%}, cases={cases})")
    return "\n".join(lines) + "\n"


def _replace_auto_distilled_section(body: str, new_section_content: str) -> str:
    """替换 body 中的 AUTO-DISTILLED-PATTERNS 区段。"""
    pattern = re.compile(
        rf"{re.escape(_AUTO_DISTILLED_START)}.*?{re.escape(_AUTO_DISTILLED_END)}",
        re.DOTALL,
    )
    replacement = f"{_AUTO_DISTILLED_START}\n{new_section_content}{_AUTO_DISTILLED_END}"
    if pattern.search(body):
        return pattern.sub(replacement, body)
    # 如果没有标记，追加到末尾
    return body + f"\n{replacement}\n"


def curate_skill(
    skill_name: str,
    *,
    use_llm: bool = False,
    skills_dir: Path | None = None,
    memory_dir: Path | None = None,
) -> Path | None:
    """整理指定 SKILL.md 并输出 .draft 文件。

    FR-016: 读取 active patterns + 当前 SKILL.md → 输出 SKILL.md.draft。
    FR-017: manually_edited: true → 整体跳过（返回 None）。
    FR-017: AUTO-DISTILLED-PATTERNS 标记存在 → 只替换该区段。

    Returns:
        Path to draft file, or None if skipped.
    """
    s_dir = skills_dir or DEFAULT_AGENT_SKILLS_DIR
    m_dir = memory_dir or DEFAULT_AGENT_MEMORY_DIR

    skill_file = s_dir / skill_name / "SKILL.md"
    if not skill_file.exists():
        logger.warning("curate_skill: SKILL.md not found for '%s'", skill_name)
        return None

    # 读取当前 SKILL.md
    try:
        content = skill_file.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content, path=skill_file)
    except Exception:
        logger.warning("curate_skill: failed to parse SKILL.md for '%s'", skill_name, exc_info=True)
        return None

    # FR-017: manually_edited → skip
    if fm.get("manually_edited", False):
        logger.info("curate_skill: '%s' is manually_edited, skipping", skill_name)
        return None

    # 加载 active patterns
    agent = _get_agent_for_skill(skill_name, s_dir)
    patterns: list[dict] = []
    if agent:
        patterns = _load_active_patterns(agent, m_dir)
    elif fm.get("scope") == "shared":
        # shared skill: 跨 4 agent 收集 active patterns
        for a in VALID_AGENT_IDS:
            patterns.extend(_load_active_patterns(a, m_dir))

    new_section = _build_auto_distilled_section(patterns)

    if use_llm:
        # LLM 整理路径（简化实现：调用 LLM 输出新 body 草稿）
        new_body = _curate_with_llm(skill_name, body, patterns, fm)
    else:
        # 非 LLM：只替换 AUTO-DISTILLED 区段
        new_body = _replace_auto_distilled_section(body, new_section)

    # 写入 draft 文件
    draft_path = skill_file.with_suffix(".md.draft")
    new_fm = dict(fm)
    new_content = render_frontmatter(new_fm) + new_body
    try:
        atomic_write(draft_path, new_content)
        logger.info("curate_skill: draft written to %s (%d patterns)", draft_path, len(patterns))
        return draft_path
    except Exception:
        logger.warning("curate_skill: failed to write draft for '%s'", skill_name, exc_info=True)
        return None


def _curate_with_llm(
    skill_name: str,
    current_body: str,
    patterns: list[dict],
    fm: dict,
) -> str:
    """LLM 辅助整理 SKILL.md body（简化实现）。

    本期输出基于当前 body + active patterns 摘要的草稿，
    详细 prompt 优化留 follow-up。
    """
    pattern_summary = _build_auto_distilled_section(patterns)
    # 替换 AUTO-DISTILLED 区段（LLM 路径也保留手工内容）
    return _replace_auto_distilled_section(current_body, pattern_summary)
