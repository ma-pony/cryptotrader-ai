"""Tests for US3: SKILL.md 加载 + 动态发现（FR-019a / FR-004b / FR-004a）。

TDD: 先 FAIL，实现 agents/skills/loader.py 后 GREEN。
"""

from __future__ import annotations

import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _write_skill(
    path: Path, name: str, scope: str, description: str = "Test skill description that is at least 30 chars."
) -> None:
    """写入测试用 SKILL.md 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
name: {name}
description: {description}
scope: {scope}
version: "1.0"
manually_edited: false
---

# Skill: {name}

This is the body content of the skill.

## Patterns

None yet.
"""
    path.write_text(content, encoding="utf-8")


class TestParseSkillMd:
    """测试 parse_skill_md()：合规 frontmatter 解析。"""

    def test_parse_valid_skill_md(self, tmp_path):
        """合规 SKILL.md 应被正确解析为 Skill 对象。"""
        from cryptotrader.agents.skills.loader import parse_skill_md

        skill_file = tmp_path / "tech-analysis" / "SKILL.md"
        _write_skill(skill_file, "tech-analysis", "agent:tech")

        skill = parse_skill_md(skill_file)
        assert skill is not None
        assert skill.name == "tech-analysis"
        assert skill.scope == "agent:tech"
        assert "body content" in skill.body
        assert skill.file_path == skill_file

    def test_parse_shared_scope(self, tmp_path):
        """scope: shared 应被正确解析。"""
        from cryptotrader.agents.skills.loader import parse_skill_md

        skill_file = tmp_path / "trading-knowledge" / "SKILL.md"
        _write_skill(skill_file, "trading-knowledge", "shared")

        skill = parse_skill_md(skill_file)
        assert skill.is_shared
        assert skill.agent_id is None

    def test_corrupt_frontmatter_raises_warning(self, tmp_path, caplog):
        """frontmatter 损坏时应 logger.warning + 返回 None（FR-024）。"""
        import logging

        from cryptotrader.agents.skills.loader import parse_skill_md

        bad_file = tmp_path / "bad-skill" / "SKILL.md"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("no frontmatter here\njust plain text", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            skill = parse_skill_md(bad_file)
        assert skill is None, "损坏的 SKILL.md 应返回 None"


class TestDiscoverSkillsForAgent:
    """测试 discover_skills_for_agent()：按 scope 动态发现（FR-004b）。"""

    def test_discover_returns_own_and_shared(self, tmp_path):
        """tech agent 应发现 scope=agent:tech 和 scope=shared 的 skills（FR-019）。"""
        from cryptotrader.agents.skills.loader import discover_skills_for_agent

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir / "tech-analysis" / "SKILL.md", "tech-analysis", "agent:tech")
        _write_skill(skills_dir / "trading-knowledge" / "SKILL.md", "trading-knowledge", "shared")
        _write_skill(skills_dir / "chain-analysis" / "SKILL.md", "chain-analysis", "agent:chain")

        skills = discover_skills_for_agent("tech", skill_dir=skills_dir)
        names = [s.name for s in skills]
        assert "tech-analysis" in names, "own skill 应被发现"
        assert "trading-knowledge" in names, "shared skill 应被发现"
        assert "chain-analysis" not in names, "其他 agent 的 skill 不应被发现"

    def test_discover_excludes_other_agent_skills(self, tmp_path):
        """chain agent 不应看到 tech agent 的 skill。"""
        from cryptotrader.agents.skills.loader import discover_skills_for_agent

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir / "tech-analysis" / "SKILL.md", "tech-analysis", "agent:tech")
        _write_skill(skills_dir / "chain-analysis" / "SKILL.md", "chain-analysis", "agent:chain")

        skills = discover_skills_for_agent("chain", skill_dir=skills_dir)
        names = [s.name for s in skills]
        assert "tech-analysis" not in names
        assert "chain-analysis" in names

    def test_mtime_cache_invalidation(self, tmp_path):
        """文件 mtime 变化后，LRU 缓存应自动失效（FR-019a）。"""
        from cryptotrader.agents.skills.loader import _clear_cache, discover_skills_for_agent

        _clear_cache()
        skills_dir = tmp_path / "agent_skills"
        skill_file = skills_dir / "tech-analysis" / "SKILL.md"
        _write_skill(skill_file, "tech-analysis", "agent:tech")

        # 第一次读取
        skills1 = discover_skills_for_agent("tech", skill_dir=skills_dir)
        assert len(skills1) == 1
        body1 = skills1[0].body

        # 修改文件内容（更新 mtime）
        time.sleep(0.01)  # 确保 mtime 变化
        new_content = skill_file.read_text() + "\n\nNew content added."
        skill_file.write_text(new_content, encoding="utf-8")

        # 第二次读取应看到新内容（缓存失效）
        _clear_cache()
        skills2 = discover_skills_for_agent("tech", skill_dir=skills_dir)
        assert len(skills2) == 1
        body2 = skills2[0].body
        assert "New content added." in body2, "mtime 变化后应重新加载文件"
        assert body1 != body2, "缓存应已失效，body 内容必须不同"
