"""Tests for US4: SkillsInjectionMiddleware 注入（FR-018 / FR-019 / FR-020 / FR-024）。

TDD: 先 FAIL，实现 agents/skills/middleware.py 后 GREEN。
"""

from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _write_skill(skills_dir: Path, name: str, scope: str) -> Path:
    """写入测试用 SKILL.md。"""
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
name: {name}
description: Test description for skill that is at least 30 chars.
scope: {scope}
version: "1.0"
manually_edited: false
---

Body content for {name}.
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestSkillsInjectionMiddleware:
    """测试 SkillsInjectionMiddleware 注入逻辑。"""

    def test_own_and_shared_both_injected(self, tmp_path):
        """tech agent 应同时注入 own (agent:tech) + shared skill（FR-019）。"""
        from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", "agent:tech")
        _write_skill(skills_dir, "trading-knowledge", "shared")

        middleware = SkillsInjectionMiddleware(agent_id="tech", skill_dir=skills_dir)
        system_prompt = middleware.build_system_addendum()

        assert "Body content for tech-analysis" in system_prompt, "own skill body 应被注入"
        assert "Body content for trading-knowledge" in system_prompt, "shared skill body 应被注入"

    def test_missing_own_skill_only_injects_shared(self, tmp_path, caplog):
        """own skill 缺失时应仅注入 shared + logger.warning（FR-024）。"""
        from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

        skills_dir = tmp_path / "agent_skills"
        # 只有 shared，没有 tech-analysis
        _write_skill(skills_dir, "trading-knowledge", "shared")

        with caplog.at_level(logging.WARNING, logger="cryptotrader.agents.skills"):
            middleware = SkillsInjectionMiddleware(agent_id="tech", skill_dir=skills_dir)
            system_prompt = middleware.build_system_addendum()

        assert "Body content for trading-knowledge" in system_prompt, "shared skill 应被注入"
        assert "Body content for tech-analysis" not in system_prompt, "缺失的 own skill 不应出现"

    def test_missing_shared_skill_only_injects_own(self, tmp_path):
        """shared skill 缺失时应仅注入 own skill。"""
        from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", "agent:tech")
        # 没有 trading-knowledge

        middleware = SkillsInjectionMiddleware(agent_id="tech", skill_dir=skills_dir)
        system_prompt = middleware.build_system_addendum()

        assert "Body content for tech-analysis" in system_prompt
        assert "trading-knowledge" not in system_prompt

    def test_both_missing_returns_empty(self, tmp_path):
        """both own + shared 都缺失时应不修改 request（返回空字符串）（FR-024）。"""
        from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

        skills_dir = tmp_path / "empty_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        middleware = SkillsInjectionMiddleware(agent_id="tech", skill_dir=skills_dir)
        system_prompt = middleware.build_system_addendum()
        assert system_prompt == "" or len(system_prompt.strip()) == 0, "无 skill 时应返回空字符串"

    def test_corrupt_frontmatter_skipped_cycle_continues(self, tmp_path, caplog):
        """frontmatter 损坏的 SKILL.md 应跳过 + warning，cycle 不崩（FR-024）。"""
        from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

        skills_dir = tmp_path / "agent_skills"
        # 写一个正常的 shared skill
        _write_skill(skills_dir, "trading-knowledge", "shared")
        # 写一个损坏的 tech skill
        bad = skills_dir / "tech-analysis" / "SKILL.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("no frontmatter here just garbage", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            middleware = SkillsInjectionMiddleware(agent_id="tech", skill_dir=skills_dir)
            system_prompt = middleware.build_system_addendum()

        # shared skill 仍被注入
        assert "Body content for trading-knowledge" in system_prompt
        # 损坏的文件不阻塞
        assert "garbage" not in system_prompt
