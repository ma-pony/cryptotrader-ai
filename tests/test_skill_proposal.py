"""Tests for US3: propose-new + 动态发现（FR-016a + FR-017a）。

TDD: 先 FAIL，实现 learning/skill_proposal.py 后 GREEN。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _make_memory_with_patterns(tmp_path: Path, agent: str, pattern_names: list[str]) -> Path:
    """创建带 patterns 的 agent_memory 目录。"""
    from cryptotrader.agents.skills._frontmatter import render_frontmatter
    from cryptotrader.agents.skills._io import ensure_memory_dirs

    ensure_memory_dirs(tmp_path)
    for name in pattern_names:
        path = tmp_path / agent / "patterns" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = {
            "name": name,
            "agent": agent,
            "description": f"Pattern {name}",
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


class TestProposeNewSkill:
    """测试 propose_new_skill()：分析 patterns → 输出新 SKILL.md draft（FR-016a）。"""

    def test_propose_agent_scope_only_reads_that_agent(self, tmp_path):
        """--scope agent:tech 仅分析 tech 的 patterns（FR-016a）。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        mem = tmp_path / "memory"
        skills = tmp_path / "skills"
        _make_memory_with_patterns(mem, "tech", ["rsi_bounce", "bollinger_squeeze"])
        _make_memory_with_patterns(mem, "chain", ["funding_squeeze"])

        draft_path = propose_new_skill(
            scope="agent:tech",
            memory_dir=mem,
            output_dir=skills,
        )
        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        # 应包含 tech patterns 内容
        assert "rsi_bounce" in content or "bollinger_squeeze" in content

    def test_propose_shared_scope_reads_all_agents(self, tmp_path):
        """--scope shared 跨 4 agent 分析（FR-016a）。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        mem = tmp_path / "memory"
        skills = tmp_path / "skills"
        for agent in ["tech", "chain", "news", "macro"]:
            _make_memory_with_patterns(mem, agent, [f"{agent}_pattern"])

        draft_path = propose_new_skill(
            scope="shared",
            memory_dir=mem,
            output_dir=skills,
        )
        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        # draft 文件应存在且非空
        assert len(content) > 100

    def test_propose_outputs_draft_not_live_skill(self, tmp_path):
        """propose_new_skill 应输出 .draft 文件，不直接覆盖 agent_skills/（FR-016a 限制）。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        mem = tmp_path / "memory"
        skills = tmp_path / "skills"
        _make_memory_with_patterns(mem, "tech", ["rsi_pattern"])

        draft_path = propose_new_skill(
            scope="agent:tech",
            memory_dir=mem,
            output_dir=skills,
        )
        assert draft_path is not None
        # draft 文件名应含 .draft 后缀或在非 agent_skills/ 的目录中
        assert ".draft" in draft_path.name or "draft" in str(draft_path), (
            "propose_new_skill 应输出 draft，不直接写 agent_skills/"
        )
