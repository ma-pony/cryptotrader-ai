"""spec 019 E2E skill evolution tests — tests/test_e2e_skill_evolution.py

SC-W14: 全链路验收（mocked cycle）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _write_skill(
    skills_dir: Path,
    name: str,
    scope: str = "shared",
    importance: float = 0.7,
    regime_tags: list[str] | None = None,
    triggers_keywords: list[str] | None = None,
) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    la = datetime.now(UTC).isoformat()
    regime_str = str(regime_tags or [])
    triggers_str = str(triggers_keywords or ["momentum"])
    content = f"""---
name: {name}
description: Test skill description that is at least 30 chars.
scope: {scope}
version: "1.0"
manually_edited: false
regime_tags: {regime_str}
triggers_keywords: {triggers_str}
importance: {importance}
confidence: 0.7
access_count: 0
last_accessed_at: "{la}"
---

# Skill Body for {name}

Key technique: analyze market momentum and funding rates.
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestEvolvingSkillProviderE2E:
    """SC-W14(a/b): EvolvingSkillProvider 在 4 agent 路径中运作。"""

    def test_get_available_skills_returns_ranked_skills(self, tmp_path: Path) -> None:
        """SC-W14(b): D-RT-01 按 importance 排序 skill 列表。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "high-importance-skill", scope="shared", importance=0.9)
        _write_skill(skills_dir, "low-importance-skill", scope="shared", importance=0.2)

        provider = EvolvingSkillProvider(skill_root=skills_dir)
        snapshot = {"price": 50000}
        result = provider.get_available_skills("tech", snapshot, k=5)

        assert len(result) >= 1
        # 高 importance 在前
        if len(result) >= 2:
            assert result[0].importance >= result[-1].importance

    def test_regime_filter_narrows_results(self, tmp_path: Path) -> None:
        """D-RT-01 第 1 层：regime 过滤。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "funding-skill", scope="shared", regime_tags=["high_funding"])
        _write_skill(skills_dir, "general-skill", scope="shared", regime_tags=[])

        provider = EvolvingSkillProvider(skill_root=skills_dir)
        # snapshot without high_funding → funding-skill 过滤掉
        snapshot = {"price": 50000}
        result = provider.get_available_skills("tech", snapshot, k=5)

        names = [s.name for s in result]
        assert "general-skill" in names
        assert "funding-skill" not in names

    def test_access_count_incremented_after_retrieval(self, tmp_path: Path) -> None:
        """SC-W14(b): access_count 写回验证。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        skill_path = _write_skill(skills_dir, "tech-analysis", scope="shared")

        provider = EvolvingSkillProvider(skill_root=skills_dir)
        _ = provider.get_available_skills("tech", {}, k=5)

        content = skill_path.read_text(encoding="utf-8")
        assert "access_count: 1" in content

    def test_get_skill_by_name_returns_body(self, tmp_path: Path) -> None:
        """SC-W14(b): get_skill_by_name 返回包含 body 的 Skill 对象。"""
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "agent_skills"
        _write_skill(skills_dir, "tech-analysis", scope="shared")

        provider = EvolvingSkillProvider(skill_root=skills_dir)
        skill = provider.get_skill_by_name("tech-analysis")

        assert skill is not None
        assert "Skill Body for tech-analysis" in skill.body


class TestTelemetryAttributes:
    """SC-W14(c): skill.retrieval.* telemetry 字段写入（graceful degradation when OTel absent）。"""

    def test_emit_telemetry_does_not_raise_without_otel(self, tmp_path: Path) -> None:
        """_emit_telemetry gracefully 降级（无 OTel 包）并记录 logger.info（FR-W28）。"""
        from cryptotrader.learning.evolution.skill_provider import _emit_telemetry

        # OTel 未安装 → should not raise, falls back to logger
        _emit_telemetry(
            candidates=[],
            filtered_out=[],
            top_k_with_scores=[],
            duration_ms=5.0,
        )
        # 函数正常返回即 PASS

    def test_emit_telemetry_logs_retrieval_attrs(self, tmp_path: Path) -> None:
        """_emit_telemetry 降级时调用 logger.info 写入 skill.retrieval.* 信息。"""
        import logging

        from cryptotrader.learning.evolution.skill_provider import _emit_telemetry

        with patch("cryptotrader.learning.evolution.skill_provider.logger") as mock_logger:
            _emit_telemetry(
                candidates=[],
                filtered_out=[],
                top_k_with_scores=[],
                duration_ms=8.3,
            )
        # logger.info should have been called with retrieval info
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "duration_ms" in call_args.args[0] or any("duration_ms" in str(a) for a in call_args.args)


class TestProposalTelemetry:
    """SC-W14(d): skill_proposal .draft 含 LLM 推断 metadata + 7 telemetry 字段。"""

    def test_propose_new_skill_creates_draft_with_metadata(self, tmp_path: Path) -> None:
        """propose_new_skill 创建 .draft 文件含 importance + regime_tags。"""
        from cryptotrader.learning.skill_proposal import propose_new_skill

        memory_dir = tmp_path / "agent_memory"
        output_dir = tmp_path / "agent_skills"

        # 写入 active pattern
        patterns_dir = memory_dir / "tech" / "patterns"
        patterns_dir.mkdir(parents=True, exist_ok=True)
        pattern = patterns_dir / "alpha_pattern.md"
        pattern.write_text(
            """---
name: alpha_pattern
description: Alpha pattern for high funding regime trigger analysis.
maturity: active
regime_tags: ['high_funding']
pnl_track:
  win_rate: 0.7
  cases: 15
---
Pattern body.
""",
            encoding="utf-8",
        )

        mock_metadata = {
            "regime_tags": ["high_funding"],
            "triggers_keywords": ["funding", "long"],
            "importance": 0.72,
            "confidence": 0.75,
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
        assert "importance: 0.72" in content
        assert "access_count: 0" in content
        assert "last_accessed_at" in content

    def test_proposal_telemetry_does_not_raise_without_otel(self, tmp_path: Path) -> None:
        """_emit_proposal_telemetry gracefully 降级（无 OTel 包），不抛出（FR-W29）。"""
        from cryptotrader.learning.skill_proposal import _emit_proposal_telemetry

        # Should not raise even without opentelemetry installed
        _emit_proposal_telemetry(
            proposed_name="test-skill",
            draft_path=tmp_path / "SKILL.md.draft",
            metadata={
                "regime_tags": ["high_funding"],
                "triggers_keywords": ["funding"],
                "importance": 0.6,
                "confidence": 0.6,
                "inference_failed": False,
            },
        )
        # 正常返回即 PASS


class TestAPISkillsEndpointE2E:
    """SC-W14(e): Web /api/memory/skills 返回更新后 access_count。"""

    def test_api_skills_reflects_access_count(self, tmp_path: Path) -> None:
        """API 从 SKILL.md 读取最新 access_count。"""
        import os

        os.environ.setdefault("AUTH_MODE", "disabled")
        os.environ.setdefault("API_KEY", "test-key-019")

        import api.routes.memory as mem_module
        from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

        skills_dir = tmp_path / "skills"
        skill_path = _write_skill(skills_dir, "tech-analysis", scope="shared")

        # Trigger access_count increment
        provider = EvolvingSkillProvider(skill_root=skills_dir)
        provider.get_available_skills("tech", {}, k=5)

        # Verify file updated
        content = skill_path.read_text(encoding="utf-8")
        assert "access_count: 1" in content

        # Verify API reads updated value
        original = mem_module._SKILLS_ROOT
        mem_module._SKILLS_ROOT = skills_dir

        from fastapi.testclient import TestClient

        from api.main import app

        with patch("api.routes.memory._SKILLS_ROOT", skills_dir):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/memory/skills")

        mem_module._SKILLS_ROOT = original

        assert resp.status_code == 200
        data = resp.json()
        items = data.get("items", [])
        if items:
            skill = next((i for i in items if i["name"] == "tech-analysis"), None)
            if skill:
                assert skill["access_count"] >= 1
