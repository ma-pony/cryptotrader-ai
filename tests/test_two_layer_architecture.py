"""Tests for US1: 双层架构解耦（agent_memory/ gitignored + agent_skills/ git-tracked）。

测试分层边界：
- .gitignore 含 agent_memory/ 条目
- agent_skills/ 含 initial 5 个目录各有 SKILL.md
- SKILL.md frontmatter 合规（name + description + scope）
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# 仓库根目录（相对于 tests/ 向上一级）
REPO_ROOT = Path(__file__).parent.parent


class TestGitignore:
    """测试 .gitignore 包含 agent_memory/ 条目（FR-002）。"""

    def test_gitignore_contains_agent_memory(self):
        """agent_memory/ 应出现在 .gitignore 中。"""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore 文件不存在"
        content = gitignore.read_text(encoding="utf-8")
        assert "agent_memory/" in content, ".gitignore 必须包含 agent_memory/ 条目"


class TestAgentSkillsDirectory:
    """测试 agent_skills/ 结构（FR-001 + FR-004）。"""

    def test_initial_five_skill_dirs_exist(self):
        """agent_skills/ 下必须存在 initial 5 个 skill 目录（FR-004）。"""
        skills_dir = REPO_ROOT / "agent_skills"
        expected = [
            "tech-analysis",
            "chain-analysis",
            "news-analysis",
            "macro-analysis",
            "trading-knowledge",
        ]
        assert skills_dir.exists(), "agent_skills/ 目录不存在"
        for name in expected:
            assert (skills_dir / name).is_dir(), f"agent_skills/{name}/ 目录不存在"

    def test_each_skill_dir_has_skill_md(self):
        """每个 skill 目录都必须有 SKILL.md（FR-014）。"""
        skills_dir = REPO_ROOT / "agent_skills"
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                skill_md = skill_dir / "SKILL.md"
                assert skill_md.exists(), f"{skill_dir.name}/SKILL.md 不存在"

    def test_skill_md_frontmatter_compliant(self):
        """每个 SKILL.md frontmatter 必须包含 name、description、scope（FR-014 + FR-004a）。"""
        from cryptotrader.agents.skills._frontmatter import parse_frontmatter, validate_skill_frontmatter

        skills_dir = REPO_ROOT / "agent_skills"
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding="utf-8")
            data, _ = parse_frontmatter(content, path=skill_md)
            # 不抛异常即表示 frontmatter 合规
            validate_skill_frontmatter(data, path=skill_md)
            assert data["name"], f"{skill_dir.name}/SKILL.md 缺少 name 字段"
            assert data["description"], f"{skill_dir.name}/SKILL.md 缺少 description 字段"
            assert data["scope"], f"{skill_dir.name}/SKILL.md 缺少 scope 字段"

    def test_agent_skills_tracked_by_git(self):
        """agent_skills/*/SKILL.md 应被 git 跟踪（不在 .gitignore 排除列表）。"""
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "agent_skills/"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        # exit code 1 = not ignored (我们期望 agent_skills/ 不被 gitignore)
        assert result.returncode != 0, "agent_skills/ 不应该被 .gitignore 排除"

    def test_agent_memory_not_tracked_by_git(self):
        """agent_memory/ 应被 .gitignore 排除（FR-002）。"""
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "agent_memory/"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        # exit code 0 = IS ignored (我们期望 agent_memory/ 被忽略)
        assert result.returncode == 0, "agent_memory/ 应该被 .gitignore 排除"
