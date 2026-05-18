"""spec 022 T019 - 验证 EvolvingSkillProvider 从 agent_skills/_internal/ 正确加载技能。

同时作为 spec 019 行为不回归的守门测试：两层检索、scope 过滤、access_count 回写
逻辑全部基于 tmp_path 隔离，不受真实磁盘路径影响。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INTERNAL_SKILLS_DIR = REPO_ROOT / "agent_skills" / "_internal"


def _write_skill(
    skills_dir: Path,
    name: str,
    scope: str = "shared",
    regime_tags: list[str] | None = None,
    importance: float = 0.5,
    confidence: float = 0.5,
    access_count: int = 0,
) -> Path:
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    regime_str = str(regime_tags or [])
    la = datetime.now(UTC).isoformat()
    content = f"""---
name: {name}
description: Test skill - internal path migration verification (spec 022).
scope: {scope}
version: "1.0"
manually_edited: false
regime_tags: {regime_str}
triggers_keywords: []
importance: {importance}
confidence: {confidence}
access_count: {access_count}
last_accessed_at: "{la}"
---

Body content for {name}.
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestInternalPathStructure:
    """验证 agent_skills/_internal/ 磁盘结构 (T016 迁移结果)。"""

    def test_internal_dir_exists(self):
        """agent_skills/_internal/ 必须存在。"""
        assert INTERNAL_SKILLS_DIR.exists(), "agent_skills/_internal/ 目录不存在"

    def test_five_skill_dirs_under_internal(self):
        """5 个 skill 目录均已迁移到 _internal/ 下。"""
        expected = [
            "tech-analysis",
            "chain-analysis",
            "news-analysis",
            "macro-analysis",
            "trading-knowledge",
        ]
        for name in expected:
            d = INTERNAL_SKILLS_DIR / name
            assert d.is_dir(), f"agent_skills/_internal/{name}/ 不存在"

    def test_each_internal_skill_has_skill_md(self):
        """每个 _internal/ skill 目录都有 SKILL.md。"""
        for skill_dir in INTERNAL_SKILLS_DIR.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                assert (skill_dir / "SKILL.md").exists(), f"agent_skills/_internal/{skill_dir.name}/SKILL.md 缺失"

    def test_old_top_level_dirs_no_longer_exist(self):
        """迁移后旧的顶层 agent_skills/<name>/ 目录不应存在 (直接删旧不留 fallback)。"""
        old_dirs = [
            "tech-analysis",
            "chain-analysis",
            "news-analysis",
            "macro-analysis",
            "trading-knowledge",
        ]
        top_level = REPO_ROOT / "agent_skills"
        for name in old_dirs:
            old_path = top_level / name
            assert not old_path.exists(), f"旧路径 agent_skills/{name}/ 仍存在，应已迁移到 _internal/"


class TestEvolvingSkillProviderFromInternalPath:
    """验证 EvolvingSkillProvider 默认路径改为 _internal/ 后行为不回归 (spec 019)。"""

    def test_default_skill_root_is_internal(self):
        """EvolvingSkillProvider 默认 skill_root 应指向 agent_skills/_internal。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        provider = EvolvingSkillProvider()
        assert "_internal" in str(provider._skill_root), (
            f"默认 skill_root 未更新到 _internal/，当前值: {provider._skill_root}"
        )

    def test_loads_four_agent_skills_from_internal(self):
        """从真实 _internal/ 加载 4 个 agent-scoped skill (scope: agent:<id>)。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        provider = EvolvingSkillProvider(skill_root=INTERNAL_SKILLS_DIR)
        for agent_id in ("tech", "chain", "news", "macro"):
            result = provider.get_available_skills(f"{agent_id}_agent", {})
            assert len(result) >= 1, f"agent={agent_id} 未从 _internal/ 加载到任何 skill"

    def test_get_skill_by_name_from_internal(self, tmp_path):
        """get_skill_by_name 能从 _internal 结构的目录找到技能。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "_internal"
        _write_skill(skills_dir, "tech-analysis", scope="agent:tech")
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        skill = provider.get_skill_by_name("tech-analysis")
        assert skill is not None
        assert skill.name == "tech-analysis"

    def test_scope_filter_works_in_internal(self, tmp_path):
        """scope 过滤逻辑在 _internal 路径下仍然正确。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "_internal"
        _write_skill(skills_dir, "tech-only", scope="agent:tech")
        _write_skill(skills_dir, "shared-skill", scope="shared")
        provider = EvolvingSkillProvider(skill_root=skills_dir)

        tech_result = provider.get_available_skills("tech", {})
        chain_result = provider.get_available_skills("chain", {})

        tech_names = [s.name for s in tech_result]
        chain_names = [s.name for s in chain_result]

        assert "tech-only" in tech_names
        assert "shared-skill" in tech_names
        assert "tech-only" not in chain_names
        assert "shared-skill" in chain_names

    def test_access_count_written_back_in_internal(self, tmp_path):
        """access_count 回写在 _internal 路径下仍然工作 (spec 019 FR-W9 不回归)。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "_internal"
        skill_path = _write_skill(skills_dir, "shared-skill", scope="shared", access_count=3)
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        provider.get_available_skills("tech", {})
        content = skill_path.read_text(encoding="utf-8")
        assert "access_count: 4" in content

    def test_empty_internal_dir_returns_empty(self, tmp_path):
        """_internal 目录为空时返回空列表 (FR-W9)。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "_internal"
        skills_dir.mkdir()
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_available_skills("tech", {})
        assert result == []

    def test_nonexistent_internal_path_returns_empty(self):
        """路径不存在时 get_available_skills 不抛异常，返回空列表。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        provider = EvolvingSkillProvider(skill_root=Path("/nonexistent/_internal"))
        result = provider.get_available_skills("tech", {})
        assert result == []
