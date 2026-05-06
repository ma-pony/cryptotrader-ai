"""Tests for US4: load_skill tool I/O（FR-021 / FR-022 / FR-023 / FR-025）。

TDD: 先 FAIL，实现 agents/skills/tool.py 后 GREEN。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _write_skill(skills_dir: Path, name: str, scope: str = "shared") -> Path:
    """写入测试用 SKILL.md。"""
    path = skills_dir / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
name: {name}
description: Test skill description that is at least 30 chars.
scope: {scope}
version: "1.0"
manually_edited: false
---

Body content for skill {name}.
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadSkillFunction:
    """测试 load_skill Python 函数接口（FR-022 双接口）。"""

    def test_skill_exists_returns_body(self, tmp_path):
        """skill 存在时应返回包含 body 的 dict（FR-022）。"""
        from cryptotrader.agents.skills.tool import load_skill

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "trading-knowledge")

        result = load_skill("trading-knowledge", skill_dir=skills_dir)
        assert isinstance(result, dict)
        assert result.get("name") == "trading-knowledge"
        assert "Body content for skill trading-knowledge" in result.get("body", "")

    def test_skill_not_found_returns_error(self, tmp_path):
        """不存在的 skill 返回 skill_not_found error（FR-025 + load_skill.contract.md）。"""
        from cryptotrader.agents.skills.tool import load_skill

        skills_dir = tmp_path / "agent_skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        result = load_skill("nonexistent", skill_dir=skills_dir)
        assert result.get("error") == "skill_not_found"
        assert result.get("name") == "nonexistent"

    def test_corrupt_file_returns_error(self, tmp_path):
        """frontmatter 损坏时返回 corrupt_file error。"""
        from cryptotrader.agents.skills.tool import load_skill

        skills_dir = tmp_path / "agent_skills"
        bad = skills_dir / "bad-skill" / "SKILL.md"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("no frontmatter here at all", encoding="utf-8")

        result = load_skill("bad-skill", skill_dir=skills_dir)
        assert result.get("error") in ("corrupt_file", "skill_not_found"), (
            f"损坏文件应返回 corrupt_file 或 skill_not_found，实际: {result}"
        )

    def test_rate_limit_per_cycle_on_11th_call(self, tmp_path):
        """同一 trace_id 第 11 次调用应返回 rate_limit_per_cycle（FR-025）。"""
        from cryptotrader.agents.skills.tool import _reset_call_counter, load_skill

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "trading-knowledge")

        trace_id = "test-trace-001"
        _reset_call_counter(trace_id)

        # 前 10 次应成功
        for i in range(10):
            result = load_skill("trading-knowledge", skill_dir=skills_dir, trace_id=trace_id)
            assert result.get("error") != "rate_limit_per_cycle", f"第 {i + 1} 次不应限流"

        # 第 11 次应被限流
        result = load_skill("trading-knowledge", skill_dir=skills_dir, trace_id=trace_id)
        assert result.get("error") == "rate_limit_per_cycle", "第 11 次应返回 rate_limit_per_cycle"
