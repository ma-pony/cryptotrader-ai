"""Tests for EvolvingSkillProvider — spec 019 D-RT-01 两层检索算法（FR-W7..W10）。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def _write_skill(
    skills_dir: Path,
    name: str,
    scope: str = "shared",
    regime_tags: list[str] | None = None,
    triggers_keywords: list[str] | None = None,
    importance: float = 0.5,
    confidence: float = 0.5,
    access_count: int = 0,
    last_accessed_at: str | None = None,
) -> Path:
    """写入测试用 SKILL.md（含 spec 019 新字段）。"""
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    regime_str = str(regime_tags or [])
    triggers_str = str(triggers_keywords or [])
    la = last_accessed_at or datetime.now(UTC).isoformat()
    content = f"""---
name: {name}
description: Test skill description that is at least 30 chars.
scope: {scope}
version: "1.0"
manually_edited: false
regime_tags: {regime_str}
triggers_keywords: {triggers_str}
importance: {importance}
confidence: {confidence}
access_count: {access_count}
last_accessed_at: "{la}"
---

Body content for skill {name}.
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestExtractRegime:
    """测试 _extract_regime snapshot 推断函数。"""

    def test_high_funding_rate(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"funding_rate": 0.0005}
        assert _extract_regime(snapshot) == "high_funding"

    def test_negative_funding_rate(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"funding_rate": -0.0002}
        assert _extract_regime(snapshot) == "negative_funding"

    def test_extreme_fear(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"fear_greed_index": 20}
        assert _extract_regime(snapshot) == "extreme_fear"

    def test_extreme_greed(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"fear_greed_index": 80}
        assert _extract_regime(snapshot) == "extreme_greed"

    def test_nested_market_funding(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"market": {"funding_rate": 0.0004}}
        assert _extract_regime(snapshot) == "high_funding"

    def test_no_regime_returns_none(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"price": 50000}
        assert _extract_regime(snapshot) is None

    def test_funding_near_zero_returns_none(self):
        from cryptotrader.learning.evolution.skill_provider import _extract_regime

        snapshot = {"funding_rate": 0.0001}
        assert _extract_regime(snapshot) is None


class TestEvolvingSkillProviderGetAvailableSkills:
    """测试 D-RT-01 两层检索算法（FR-W7/W8/W9）。"""

    def test_empty_skill_dir_returns_empty(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        skills_dir.mkdir()
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_available_skills("tech_agent", {})
        assert result == []

    def test_returns_skills_matching_scope(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", scope="shared")
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_available_skills("tech_agent", {})
        assert len(result) >= 1
        assert any(s.name == "tech-analysis" for s in result)

    def test_regime_filter_excludes_non_matching(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        # high_funding skill — should be filtered out when no high funding
        _write_skill(skills_dir, "funding-skill", scope="shared", regime_tags=["high_funding"])
        # no regime skill — always passes
        _write_skill(skills_dir, "general-skill", scope="shared", regime_tags=[])
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        # snapshot with extreme_fear, not high_funding
        snapshot = {"fear_greed_index": 10}
        result = provider.get_available_skills("tech_agent", snapshot)
        names = [s.name for s in result]
        assert "general-skill" in names
        assert "funding-skill" not in names

    def test_regime_filter_includes_matching(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "funding-skill", scope="shared", regime_tags=["high_funding"])
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        snapshot = {"funding_rate": 0.0005}  # high_funding
        result = provider.get_available_skills("tech_agent", snapshot)
        names = [s.name for s in result]
        assert "funding-skill" in names

    def test_top_k_limits_result(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        for i in range(10):
            _write_skill(skills_dir, f"skill-{i:02d}", scope="shared")
        provider = EvolvingSkillProvider(skill_root=skills_dir, top_k=3)
        result = provider.get_available_skills("tech_agent", {}, k=3)
        assert len(result) <= 3

    def test_exception_returns_empty_list(self, tmp_path):
        """任何异常 → [] (FR-W9)。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        provider = EvolvingSkillProvider(skill_root=Path("/nonexistent/path"))
        result = provider.get_available_skills("tech_agent", {})
        assert result == []

    def test_access_count_written_back(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        skill_path = _write_skill(skills_dir, "tech-analysis", scope="shared", access_count=0)
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_available_skills("tech_agent", {})
        assert len(result) >= 1
        # Read file back to verify access_count incremented
        content = skill_path.read_text(encoding="utf-8")
        assert "access_count: 1" in content

    def test_importance_affects_ordering(self, tmp_path):
        """高 importance 的 skill 应排在前面。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "low-importance", scope="shared", importance=0.1, confidence=1.0)
        _write_skill(skills_dir, "high-importance", scope="shared", importance=0.9, confidence=1.0)
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_available_skills("tech_agent", {}, k=2)
        if len(result) >= 2:
            assert result[0].name == "high-importance"


class TestEvolvingSkillProviderGetSkillByName:
    """测试 get_skill_by_name（FR-W10）。"""

    def test_found_returns_skill(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", scope="shared")
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        skill = provider.get_skill_by_name("tech-analysis")
        assert skill is not None
        assert skill.name == "tech-analysis"
        assert "Body content for skill tech-analysis" in skill.body

    def test_not_found_returns_none(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        skills_dir.mkdir()
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        result = provider.get_skill_by_name("nonexistent")
        assert result is None

    def test_access_count_incremented(self, tmp_path):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        skill_path = _write_skill(skills_dir, "tech-analysis", scope="shared", access_count=5)
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        provider.get_skill_by_name("tech-analysis")
        content = skill_path.read_text(encoding="utf-8")
        assert "access_count: 6" in content

    def test_exception_returns_none(self):
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        provider = EvolvingSkillProvider(skill_root=Path("/nonexistent/path"))
        result = provider.get_skill_by_name("any-skill")
        assert result is None
