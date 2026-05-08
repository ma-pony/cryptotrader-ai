"""Tests for skill injection via DefaultSkillProvider (spec 017b replaces middleware).

Originally tested SkillsInjectionMiddleware (FR-018/FR-019/FR-020/FR-024).
After spec 017b the middleware was deleted; skill injection now happens via
DefaultSkillProvider.get_available_skills() + PromptBuilder._render_skills().
These tests cover the same scope-filter and error-resilience requirements via
the new path.
"""

from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _write_skill(skills_dir: Path, name: str, scope: str) -> Path:
    """Write a test SKILL.md."""
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
    """Scope-filter + error-resilience tests (FR-019 / FR-024) via DefaultSkillProvider."""

    def test_own_and_shared_both_injected(self, tmp_path):
        """tech agent should receive both agent:tech and shared skills (FR-019)."""
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", "agent:tech")
        _write_skill(skills_dir, "trading-knowledge", "shared")

        provider = DefaultSkillProvider(skills_root=skills_dir)
        skills = provider.get_available_skills("tech", snapshot={})
        bodies = " ".join(s.body for s in skills)

        assert "Body content for tech-analysis" in bodies, "own skill body should be injected"
        assert "Body content for trading-knowledge" in bodies, "shared skill body should be injected"

    def test_missing_own_skill_only_injects_shared(self, tmp_path, caplog):
        """When own skill absent, only shared skill injected; no crash (FR-024)."""
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "trading-knowledge", "shared")

        with caplog.at_level(logging.WARNING, logger="cryptotrader.agents.skills"):
            provider = DefaultSkillProvider(skills_root=skills_dir)
            skills = provider.get_available_skills("tech", snapshot={})

        bodies = " ".join(s.body for s in skills)
        assert "Body content for trading-knowledge" in bodies
        assert "Body content for tech-analysis" not in bodies

    def test_missing_shared_skill_only_injects_own(self, tmp_path):
        """When shared skill absent, only own skill injected."""
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", "agent:tech")

        provider = DefaultSkillProvider(skills_root=skills_dir)
        skills = provider.get_available_skills("tech", snapshot={})
        bodies = " ".join(s.body for s in skills)

        assert "Body content for tech-analysis" in bodies
        assert "trading-knowledge" not in bodies

    def test_both_missing_returns_empty(self, tmp_path):
        """When no skills exist, get_available_skills returns [] (FR-024)."""
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        skills_dir = tmp_path / "empty_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        provider = DefaultSkillProvider(skills_root=skills_dir)
        skills = provider.get_available_skills("tech", snapshot={})
        assert skills == []

    def test_corrupt_frontmatter_skipped_cycle_continues(self, tmp_path, caplog):
        """Corrupt SKILL.md is skipped + warning logged; cycle continues (FR-024)."""
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "trading-knowledge", "shared")
        # Write a corrupt tech skill
        bad = skills_dir / "tech-analysis" / "SKILL.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("no frontmatter here just garbage", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            provider = DefaultSkillProvider(skills_root=skills_dir)
            skills = provider.get_available_skills("tech", snapshot={})

        bodies = " ".join(s.body for s in skills)
        assert "Body content for trading-knowledge" in bodies
        assert "garbage" not in bodies
