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
from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata

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


def _add_metadata_to_frontmatter(draft_content: str, metadata: dict) -> str:
    """把 LLM 推断的 metadata 合并到 draft frontmatter（spec 019 FR-W16）。

    仅在 frontmatter 中不存在对应字段时写入（不覆盖已有值）。
    """
    import yaml

    # 找 frontmatter 分隔符
    if not draft_content.startswith("---"):
        return draft_content
    second = draft_content.find("\n---", 3)
    if second == -1:
        return draft_content

    yaml_str = draft_content[3:second].strip()
    body = draft_content[second + 4 :]

    try:
        fm = yaml.safe_load(yaml_str)
        if not isinstance(fm, dict):
            return draft_content
    except Exception:
        return draft_content

    # 合并 metadata（不覆盖已有字段）
    for key, val in metadata.items():
        if key not in fm:
            fm[key] = val

    # 重新渲染 frontmatter
    from cryptotrader.agents.skills._frontmatter import render_frontmatter

    return render_frontmatter(fm) + body


def _emit_proposal_telemetry(
    proposed_name: str,
    draft_path: Path,
    metadata: dict,
) -> None:
    """写 7 个 OpenTelemetry span attributes（spec 019 FR-W29 + spec 020c P2-3：
    llm_call_failed 局部变量已合并到 metadata["inference_failed"] 单一来源；
    OTel attr 名保留 llm_call_failed 用于后向兼容 dashboard）。"""
    inference_failed = bool(metadata.get("inference_failed", False))
    attrs = {
        "skill.proposal.name": proposed_name,
        "skill.proposal.draft_path": str(draft_path),
        "skill.proposal.llm_inferred_regime_tags": str(metadata.get("regime_tags", [])),
        "skill.proposal.llm_inferred_triggers_keywords": str(metadata.get("triggers_keywords", [])),
        "skill.proposal.llm_inferred_importance": float(metadata.get("importance", 0.5)),
        "skill.proposal.llm_inferred_confidence": float(metadata.get("confidence", 0.5)),
        "skill.proposal.llm_call_failed": inference_failed,
    }
    span_attached = False
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            for key, val in attrs.items():
                if isinstance(val, list | dict):
                    span.set_attribute(key, str(val))
                else:
                    span.set_attribute(key, val)
            span_attached = True
    except Exception:
        pass

    if not span_attached:
        logger.info(
            "skill_proposal name=%s draft_path=%s regime_tags=%s importance=%.2f confidence=%.2f llm_failed=%s",
            proposed_name,
            draft_path,
            metadata.get("regime_tags", []),
            metadata.get("importance", 0.5),
            metadata.get("confidence", 0.5),
            inference_failed,
        )


def propose_new_skill(
    scope: str,
    memory_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """分析 active patterns，提议新 SKILL.md draft。

    FR-016a: 按 scope 过滤：
      - "agent:<id>": 仅分析该 agent 的 patterns
      - "shared": 跨 4 agent 找共性

    spec 019 FR-W16: 创建 .draft 时调 LLM 推断 metadata 写入 frontmatter。

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

    # spec 019 FR-W16: 调 LLM 推断 metadata，合并到 frontmatter
    _default_metadata = {
        "regime_tags": [],
        "triggers_keywords": [],
        "importance": 0.5,
        "confidence": 0.5,
        "inference_failed": True,  # spec 020a FR-Z17: default path = failure
    }
    try:
        description = f"Proposed skill based on {len(patterns)} active patterns with common regime tags: {', '.join(common_tags) or 'various'}."
        metadata = infer_skill_metadata(
            name=proposed_name,
            description=description,
            body=draft_content,
        )
        # spec 020a FR-Z17: inference_failed is set by infer_skill_metadata itself;
        # fall back to False if the key is somehow missing (success path).
        if "inference_failed" not in metadata:
            metadata["inference_failed"] = False
    except Exception:
        logger.warning("propose_new_skill: LLM metadata inference failed", exc_info=True)
        metadata = dict(_default_metadata)  # includes inference_failed: True

    # 把 metadata 合并到 draft frontmatter（access_count=0 / last_accessed_at 由迁移脚本处理）
    metadata["access_count"] = 0
    from datetime import UTC, datetime

    metadata["last_accessed_at"] = datetime.now(UTC).isoformat()
    draft_content = _add_metadata_to_frontmatter(draft_content, metadata)

    # 写入 draft 文件（不覆盖已存在的 SKILL.md）
    draft_path = o_dir / proposed_name / "SKILL.md.draft"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write(draft_path, draft_content)
        logger.info("propose_new_skill: draft written to %s", draft_path)
        # spec 019 FR-W29: 写 7 telemetry attributes
        _emit_proposal_telemetry(proposed_name, draft_path, metadata)
        return draft_path
    except Exception:
        logger.warning("propose_new_skill: failed to write draft", exc_info=True)
        return None
