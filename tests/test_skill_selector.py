"""Tests for SkillSelector — regime matching and token budget."""

from __future__ import annotations

import logging

from cryptotrader.agents.skill_loader import SkillLoader
from cryptotrader.agents.skill_selector import SkillSelector
from cryptotrader.config import AgentConfig


def _make_loader(tmp_path, skills: dict[str, str]) -> SkillLoader:
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir(exist_ok=True)
    for name, content in skills.items():
        (skill_dir / f"{name}.md").write_text(content)
    return SkillLoader(search_paths=[skill_dir])


class TestSkillSelector:
    def test_default_skills_loaded(self, tmp_path):
        loader = _make_loader(tmp_path, {"momentum": "# Momentum Strategy"})
        cfg = AgentConfig(agent_id="test", skills=["momentum"])
        selector = SkillSelector()
        result = selector.select(cfg, [], loader)
        assert len(result) == 1
        assert "Momentum" in result[0]

    def test_regime_tag_match_appends_extra(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            {
                "base_skill": "base content",
                "bull_momentum": "bull content",
            },
        )
        cfg = AgentConfig(
            agent_id="test",
            skills=["base_skill"],
            regime_skills={"trending_up": ["bull_momentum"]},
        )
        selector = SkillSelector()
        result = selector.select(cfg, ["trending_up"], loader)
        assert len(result) == 2
        assert "base content" in result[0]
        assert "bull content" in result[1]

    def test_no_regime_match_only_default(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            {
                "base_skill": "base content",
                "bull_momentum": "bull content",
            },
        )
        cfg = AgentConfig(
            agent_id="test",
            skills=["base_skill"],
            regime_skills={"trending_up": ["bull_momentum"]},
        )
        selector = SkillSelector()
        result = selector.select(cfg, ["high_volatility"], loader)
        assert len(result) == 1
        assert "base content" in result[0]

    def test_multi_tag_match_deduplicates(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            {
                "shared_skill": "shared content",
                "vol_skill": "vol content",
            },
        )
        cfg = AgentConfig(
            agent_id="test",
            skills=["shared_skill"],
            regime_skills={
                "trending_up": ["shared_skill", "vol_skill"],
                "high_volatility": ["vol_skill"],
            },
        )
        selector = SkillSelector()
        result = selector.select(cfg, ["trending_up", "high_volatility"], loader)
        assert len(result) == 2

    def test_token_budget_truncation(self, tmp_path, caplog):
        loader = _make_loader(
            tmp_path,
            {
                "big_skill": "x" * 5000,
                "small_skill": "y" * 100,
            },
        )
        cfg = AgentConfig(agent_id="test", skills=["big_skill", "small_skill"])
        selector = SkillSelector()
        with caplog.at_level(logging.WARNING):
            result = selector.select(cfg, [], loader, token_budget_chars=4000)
        assert len(result) == 1
        assert len(result[0]) == 4000
        assert "budget exceeded" in caplog.text

    def test_empty_config_returns_empty(self, tmp_path):
        loader = _make_loader(tmp_path, {})
        cfg = AgentConfig(agent_id="test")
        selector = SkillSelector()
        result = selector.select(cfg, [], loader)
        assert result == []
