"""Skill 提议 — 分析 active patterns → 输出新 SKILL.md draft。

FR-016a: `arena skills propose-new [--scope <shared|agent:<id>>]`
         分析 active patterns 找共同 regime/theme 子集，
         输出到 stdout 或 agent_skills/<proposed-name>/SKILL.md.draft。
         不自动创建 skill 文件 — 需用户 review + manual save。
"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR, DEFAULT_AGENT_SKILLS_DIR, VALID_AGENT_IDS
from cryptotrader.agents.skills._frontmatter import parse_frontmatter, render_frontmatter
from cryptotrader.agents.skills._io import atomic_write

logger = logging.getLogger(__name__)


def _load_active_patterns_for_agent(agent: str, memory_dir: Path) -> list[dict]:
    """读取 agent_memory/<agent>/patterns/ 下所有 active 状态的 patterns。"""
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
                    {
                        "agent": agent,
                        "name": fm.get("name", f.stem),
                        "description": fm.get("description", ""),
                        "regime_tags": fm.get("regime_tags", []),
                        "body": body,
                        "fm": fm,
                    }
                )
        except Exception:
            logger.warning("Failed to load pattern %s", f, exc_info=True)
    return result


def _find_common_regime_subset(patterns: list[dict]) -> list[str]:
    """找出 patterns 中最常出现的 regime tags（共性主题）。"""
    from collections import Counter

    tag_counter: Counter = Counter()
    for p in patterns:
        for tag in p.get("regime_tags", []):
            tag_counter[tag] += 1

    if not tag_counter:
        return []
    # 返回出现次数 > 总数/3 的 tags
    threshold = max(1, len(patterns) // 3)
    return [tag for tag, count in tag_counter.items() if count >= threshold]


def _generate_proposed_name(scope: str, common_tags: list[str]) -> str:
    """生成提议的 skill 名称。"""
    if scope.startswith("agent:"):
        agent = scope.split(":", 1)[1]
        tag_part = common_tags[0].replace("_", "-") if common_tags else "general"
        return f"{agent}-{tag_part}-strategy"
    tag_part = "-".join(common_tags[:2]).replace("_", "-") if common_tags else "cross-agent"
    return f"shared-{tag_part}-strategy"


def _build_draft_content(
    proposed_name: str,
    scope: str,
    patterns: list[dict],
    common_tags: list[str],
) -> str:
    """构建 draft SKILL.md 内容。"""
    fm = {
        "name": proposed_name,
        "description": f"Proposed skill based on {len(patterns)} active patterns with common regime tags: {', '.join(common_tags) or 'various'}.",
        "scope": scope,
        "version": "1.0",
        "manually_edited": False,
        "_draft": True,
        "_source_patterns": [p["name"] for p in patterns],
        "_common_regime_tags": common_tags,
    }
    header = render_frontmatter(fm)

    lines = [f"# Proposed Skill: {proposed_name}\n\n"]
    lines.append(f"**Scope**: {scope}\n")
    lines.append(f"**Common Regime Tags**: {', '.join(common_tags) or 'none identified'}\n\n")
    lines.append("## Proposed Role\n\n")
    lines.append("(Fill in the agent role and usage rules based on the patterns below)\n\n")
    lines.append("## Source Active Patterns\n\n")

    for p in patterns:
        name = p["name"]
        agent = p.get("agent", "?")
        desc = p.get("description", "")
        regime = ", ".join(p.get("regime_tags", []))
        fm_data = p.get("fm", {})
        wr = fm_data.get("pnl_track", {}).get("win_rate", 0)
        cases = fm_data.get("pnl_track", {}).get("cases", 0)
        lines.append(f"### {agent}::{name}\n\n")
        lines.append(f"- **Description**: {desc}\n")
        lines.append(f"- **Regime**: {regime or 'any'}\n")
        lines.append(f"- **Performance**: win_rate={wr:.0%}, cases={cases}\n\n")

    lines.append("## Usage Rules\n\n")
    lines.append("(To be filled in after review)\n\n")
    lines.append("<!-- AUTO-DISTILLED-PATTERNS -->\n")
    lines.append("(Patterns section — auto-populated by curation)\n")
    lines.append("<!-- END-AUTO-DISTILLED-PATTERNS -->\n")

    return header + "".join(lines)


def propose_new_skill(
    scope: str,
    memory_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """分析 active patterns，提议新 SKILL.md draft。

    FR-016a: 按 scope 过滤：
      - "agent:<id>": 仅分析该 agent 的 patterns
      - "shared": 跨 4 agent 找共性

    Returns:
        Path to .draft file (in output_dir), or None on failure.
    """
    m_dir = memory_dir or DEFAULT_AGENT_MEMORY_DIR
    o_dir = output_dir or DEFAULT_AGENT_SKILLS_DIR

    # 加载 active patterns
    patterns: list[dict] = []
    if scope.startswith("agent:"):
        agent = scope.split(":", 1)[1]
        if agent not in VALID_AGENT_IDS:
            logger.warning("propose_new_skill: unknown agent '%s'", agent)
            return None
        patterns = _load_active_patterns_for_agent(agent, m_dir)
    elif scope == "shared":
        for a in VALID_AGENT_IDS:
            patterns.extend(_load_active_patterns_for_agent(a, m_dir))
    else:
        logger.warning("propose_new_skill: invalid scope '%s'", scope)
        return None

    if not patterns:
        logger.info("propose_new_skill: no active patterns found for scope '%s'", scope)
        # Return a minimal draft even with no patterns
        proposed_name = _generate_proposed_name(scope, [])
    else:
        common_tags = _find_common_regime_subset(patterns)
        proposed_name = _generate_proposed_name(scope, common_tags)

    common_tags = _find_common_regime_subset(patterns)
    draft_content = _build_draft_content(proposed_name, scope, patterns, common_tags)

    # 写入 draft 文件（不覆盖已存在的 SKILL.md）
    draft_path = o_dir / proposed_name / "SKILL.md.draft"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write(draft_path, draft_content)
        logger.info("propose_new_skill: draft written to %s", draft_path)
        return draft_path
    except Exception:
        logger.warning("propose_new_skill: failed to write draft", exc_info=True)
        return None
