"""Tests for SkillLoader — file-system Skill loading."""

from __future__ import annotations

import logging

from cryptotrader.agents.skill_loader import SkillLoader


class TestSkillLoader:
    def test_load_existing_skill(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "momentum.md").write_text("# Momentum\nBuy low, sell high.")
        loader = SkillLoader(search_paths=[skill_dir])
        content = loader.load("momentum")
        assert "Momentum" in content
        assert "Buy low" in content

    def test_load_missing_skill_returns_empty(self, tmp_path, caplog):
        loader = SkillLoader(search_paths=[tmp_path])
        with caplog.at_level(logging.WARNING):
            result = loader.load("nonexistent")
        assert result == ""
        assert "not found" in caplog.text

    def test_project_dir_priority_over_home(self, tmp_path):
        proj_dir = tmp_path / "project_skills"
        home_dir = tmp_path / "home_skills"
        proj_dir.mkdir()
        home_dir.mkdir()
        (proj_dir / "strat.md").write_text("project version")
        (home_dir / "strat.md").write_text("home version")
        loader = SkillLoader(search_paths=[proj_dir, home_dir])
        content = loader.load("strat")
        assert "project version" in content

    def test_sanitize_applied(self, tmp_path):
        skill_dir = tmp_path / "skills"
        skill_dir.mkdir()
        (skill_dir / "test.md").write_text("Normal content here")
        loader = SkillLoader(search_paths=[skill_dir])
        content = loader.load("test")
        assert content == "Normal content here"
