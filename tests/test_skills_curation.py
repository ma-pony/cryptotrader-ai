"""Tests for US3: SKILL.md 整理（curation）（FR-015 / FR-016 / FR-017）。

TDD: 先 FAIL，实现 learning/curation.py 后 GREEN。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _write_skill(skills_dir: Path, name: str, scope: str, manually_edited: bool = False) -> Path:
    """写入测试用 SKILL.md 文件。"""
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
name: {name}
description: Test skill description that is at least 30 chars.
scope: {scope}
version: "1.0"
manually_edited: {str(manually_edited).lower()}
---

# Skill: {name}

Initial body content.

## Patterns

<!-- AUTO-DISTILLED-PATTERNS -->
Old patterns content.
<!-- END-AUTO-DISTILLED-PATTERNS -->
"""
    path.write_text(content, encoding="utf-8")
    return path


def _make_memory_with_patterns(tmp_path: Path, agent: str, count: int = 3) -> Path:
    """创建带 active patterns 的 memory 目录。"""
    from cryptotrader.agents.skills._frontmatter import render_frontmatter
    from cryptotrader.agents.skills._io import ensure_memory_dirs

    ensure_memory_dirs(tmp_path)
    for i in range(count):
        name = f"pattern_{i:02d}"
        path = tmp_path / agent / "patterns" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = {
            "name": name,
            "agent": agent,
            "description": f"Test pattern {i}",
            "maturity": "active",
            "manually_edited": False,
            "regime_tags": ["high_funding"],
            "pnl_track": {"cases": 10, "wins": 7, "win_rate": 0.7, "avg_pnl": 80.0, "last_active": "2026-05-06"},
            "source_cycles": [],
            "created": "2026-05-01T00:00:00",
            "version": 1,
        }
        content = render_frontmatter(fm) + f"\n## {name}\n\nConditions.\n"
        path.write_text(content, encoding="utf-8")
    return tmp_path


class TestCurateSkill:
    """测试 curate_skill()：整理 SKILL.md draft（FR-016）。"""

    def test_curate_no_llm_generates_draft(self, tmp_path):
        """curate_skill(use_llm=False) 应生成 SKILL.md.draft（FR-016）。"""
        from cryptotrader.learning.curation import curate_skill

        skills_dir = tmp_path / "agent_skills"
        mem_dir = tmp_path / "memory"
        _write_skill(skills_dir, "tech-analysis", "agent:tech")
        _make_memory_with_patterns(mem_dir, "tech", count=3)

        draft_path = curate_skill(
            skill_name="tech-analysis",
            use_llm=False,
            skills_dir=skills_dir,
            memory_dir=mem_dir,
        )
        assert draft_path is not None
        assert draft_path.exists(), "draft 文件应被创建"
        content = draft_path.read_text(encoding="utf-8")
        assert len(content) > 100, "draft 文件不应为空"

    def test_curate_skips_manually_edited(self, tmp_path):
        """manually_edited: true 的 SKILL.md 应被整体跳过（FR-017）。"""
        from cryptotrader.learning.curation import curate_skill

        skills_dir = tmp_path / "agent_skills"
        mem_dir = tmp_path / "memory"
        _write_skill(skills_dir, "tech-analysis", "agent:tech", manually_edited=True)
        _make_memory_with_patterns(mem_dir, "tech", count=3)

        run = curate_skill(
            skill_name="tech-analysis",
            use_llm=False,
            skills_dir=skills_dir,
            memory_dir=mem_dir,
        )
        # 被跳过时应返回 None 或空路径
        assert run is None, "manually_edited 的 SKILL.md 整理应被跳过"

    def test_curate_only_replaces_auto_distilled_section(self, tmp_path):
        """含 AUTO-DISTILLED-PATTERNS 标记时，仅替换该区段（FR-017）。"""
        from cryptotrader.learning.curation import curate_skill

        skills_dir = tmp_path / "agent_skills"
        mem_dir = tmp_path / "memory"
        _write_skill(skills_dir, "tech-analysis", "agent:tech", manually_edited=False)
        _make_memory_with_patterns(mem_dir, "tech", count=2)

        draft_path = curate_skill(
            skill_name="tech-analysis",
            use_llm=False,
            skills_dir=skills_dir,
            memory_dir=mem_dir,
        )
        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        # 原始手工内容（标记外）应保留
        assert "Initial body content." in content, "标记外的手工内容应保留"
        # AUTO-DISTILLED-PATTERNS 区段应被更新
        assert "<!-- AUTO-DISTILLED-PATTERNS -->" in content
