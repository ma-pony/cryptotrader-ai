"""Tests for spec 019 FR-W16/W29: propose_new_skill LLM metadata inference + telemetry。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _write_pattern(memory_dir: Path, agent: str, name: str, regime_tags: list[str] | None = None) -> None:
    """写入 active pattern 文件。"""
    patterns_dir = memory_dir / agent / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    tags = regime_tags or ["high_funding"]
    path = patterns_dir / f"{name}.md"
    content = f"""---
name: {name}
description: Test pattern description that is at least 30 chars.
maturity: active
regime_tags: {tags}
pnl_track:
  win_rate: 0.65
  cases: 20
---

Pattern body for {name}.
"""
    path.write_text(content, encoding="utf-8")


class TestProposeNewSkillMetadataInference:
    """测试 propose_new_skill 调用 LLM metadata inference（FR-W16）。"""

    def test_llm_metadata_merged_into_draft_frontmatter(self, tmp_path):
        """LLM 返回 metadata 后应合并进 draft frontmatter。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        _write_pattern(memory_dir, "tech", "pattern-alpha", ["high_funding"])

        mock_metadata = {
            "regime_tags": ["high_funding"],
            "triggers_keywords": ["funding", "long"],
            "importance": 0.75,
            "confidence": 0.8,
        }
        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            return_value=mock_metadata,
        ):
            draft_path = propose_new_skill(
                scope="agent:tech",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        assert "importance: 0.75" in content
        assert "confidence: 0.8" in content

    def test_llm_failure_falls_back_to_defaults(self, tmp_path):
        """LLM 调用失败时应使用默认 metadata，draft 仍成功写入。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        _write_pattern(memory_dir, "tech", "pattern-beta")

        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            draft_path = propose_new_skill(
                scope="agent:tech",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        # 默认值应写入 frontmatter
        assert "importance: 0.5" in content

    def test_draft_written_even_with_no_patterns(self, tmp_path):
        """无 active patterns 时仍应生成 draft 文件。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        (memory_dir / "tech" / "patterns").mkdir(parents=True, exist_ok=True)

        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            return_value={
                "regime_tags": [],
                "triggers_keywords": [],
                "importance": 0.5,
                "confidence": 0.5,
            },
        ):
            draft_path = propose_new_skill(
                scope="agent:tech",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        assert draft_path.name == "SKILL.md.draft"

    def test_access_count_zero_in_draft(self, tmp_path):
        """draft 中 access_count 应为 0（新 skill 尚未使用）。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        _write_pattern(memory_dir, "tech", "pattern-gamma")

        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            return_value={
                "regime_tags": ["high_funding"],
                "triggers_keywords": ["funding"],
                "importance": 0.6,
                "confidence": 0.7,
            },
        ):
            draft_path = propose_new_skill(
                scope="agent:tech",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        assert "access_count: 0" in content

    def test_last_accessed_at_present_in_draft(self, tmp_path):
        """draft frontmatter 中应有 last_accessed_at 字段。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        _write_pattern(memory_dir, "tech", "pattern-delta")

        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            return_value={
                "regime_tags": [],
                "triggers_keywords": [],
                "importance": 0.5,
                "confidence": 0.5,
            },
        ):
            draft_path = propose_new_skill(
                scope="agent:tech",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        content = draft_path.read_text(encoding="utf-8")
        assert "last_accessed_at" in content

    def test_shared_scope_aggregates_patterns(self, tmp_path):
        """shared scope 应跨所有 agents 汇总 patterns。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"
        for agent in ["tech", "chain"]:
            _write_pattern(memory_dir, agent, f"pattern-{agent}-1", ["high_funding"])

        with patch(
            "cryptotrader.learning.skill_proposal.infer_skill_metadata",
            return_value={
                "regime_tags": ["high_funding"],
                "triggers_keywords": ["funding"],
                "importance": 0.7,
                "confidence": 0.75,
            },
        ) as mock_infer:
            draft_path = propose_new_skill(
                scope="shared",
                memory_dir=memory_dir,
                output_dir=output_dir,
            )

        assert draft_path is not None
        # infer_skill_metadata 应被调用一次
        mock_infer.assert_called_once()
        content = draft_path.read_text(encoding="utf-8")
        # shared scope 生成名称以 shared- 开头
        assert "shared-" in str(draft_path)
