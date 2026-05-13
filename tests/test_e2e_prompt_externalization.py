"""E2E tests for prompt externalization (spec 017b, US-Y5 / SC-Y10 / SC-Y13 / SC-Y15 / SC-Y17).

Coverage:
- T051: PromptBuilder telemetry 8 fields on each of 4 agents
- T052: fixture skills — _test_shared loaded by all 4 agents; _test_tech only by tech
- SC-Y13: scope filter correctness via DefaultSkillProvider
- live_steering parameter renders into prompt user-tail
- SC-Y17: PromptBuilder.build() returns (SystemMessage, HumanMessage) for all 4 agents
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURES_SKILLS = Path(__file__).parent / "fixtures" / "skills"
REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "agents"


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_pb(agent_id: str, skills_root: Path | None = None):
    """Build a real PromptBuilder for agent_id, optionally overriding the skills root."""
    from cryptotrader.agents.prompt_builder import (
        DefaultSkillProvider,
        PromptBuilder,
    )

    return PromptBuilder(
        agent_id=agent_id,
        config_dir=CONFIG_DIR,
        skill_provider=DefaultSkillProvider(skills_root=skills_root or REPO_ROOT / "agent_skills"),
    )


def _snapshot_dict(pair: str = "BTC/USDT") -> dict:
    return {
        "pair": pair,
        "timestamp": "2026-01-01T00:00:00Z",
        "ticker": {"last": 50000.0},
        "funding_rate": 0.0001,
        "volatility": 0.02,
        "onchain": {"open_interest": 1000000.0, "exchange_netflow": -500.0, "liquidations_24h": {}},
        "news": {"headlines": ["BTC ETF sees record inflows"]},
        "macro": {"fed_rate": 5.25, "dxy": 104.0},
    }


# ── SC-Y17: PromptBuilder.build() returns correct message types ───────────────


class TestPromptBuilderBuildAllAgents:
    """SC-Y17: All 4 builtin agents produce valid (SystemMessage, HumanMessage) pairs."""

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_build_returns_correct_message_types(self, agent_id):
        from langchain_core.messages import HumanMessage, SystemMessage

        pb = _build_pb(agent_id)
        sys_msg, usr_msg = pb.build(snapshot=_snapshot_dict(), portfolio={})
        assert isinstance(sys_msg, SystemMessage), f"{agent_id}: expected SystemMessage"
        assert isinstance(usr_msg, HumanMessage), f"{agent_id}: expected HumanMessage"
        assert len(sys_msg.content) > 20, f"{agent_id}: system prompt too short"

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_build_system_contains_role_text(self, agent_id):
        """system_prompt section from config/agents/<id>.md appears in SystemMessage."""
        pb = _build_pb(agent_id)
        sys_msg, _ = pb.build(snapshot=_snapshot_dict(), portfolio={})
        role_prefix = pb.config.body_sections["system_prompt"][:30]
        assert role_prefix in sys_msg.content, f"{agent_id}: system_prompt section not in SystemMessage"


# ── T051: telemetry 8 fields ──────────────────────────────────────────────────


class TestPromptBuilderTelemetry:
    """T051: Each agent's build() emits 8 telemetry fields."""

    TELEMETRY_FIELDS = {
        "prompt.builder.agent_id",
        "prompt.builder.sections_included",
        "prompt.builder.dropped_sections",
        "prompt.builder.degraded_sections",
        "prompt.builder.prompt_size_pre",
        "prompt.builder.prompt_size_post",
        "prompt.builder.budget",
        "prompt.builder.duration_ms",
    }

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_telemetry_emitted_via_log(self, agent_id, caplog):
        """When no OTel span is active, telemetry falls through to logger.info."""
        import logging

        pb = _build_pb(agent_id)
        with caplog.at_level(logging.INFO, logger="cryptotrader.agents.prompt_builder"):
            pb.build(snapshot=_snapshot_dict(), portfolio={})

        # At minimum, the logger.info path emits agent_id in the message
        messages = " ".join(r.message for r in caplog.records)
        assert agent_id in messages, f"{agent_id}: agent_id missing from telemetry log"

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_telemetry_attached_to_otel_span(self, agent_id):
        """When an OTel span is active, all 8 telemetry fields are set on it."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        span_attrs: dict = {}
        mock_span.set_attribute.side_effect = lambda k, v: span_attrs.update({k: v})

        # Inject a fake opentelemetry.trace into sys.modules so the
        # try-block inside prompt_builder._emit_telemetry() succeeds even
        # when the real opentelemetry package is not installed.
        fake_trace = types.ModuleType("opentelemetry.trace")
        fake_trace.get_current_span = lambda: mock_span  # type: ignore[attr-defined]
        fake_otel = types.ModuleType("opentelemetry")
        fake_otel.trace = fake_trace  # type: ignore[attr-defined]

        orig_otel = sys.modules.get("opentelemetry")
        orig_trace = sys.modules.get("opentelemetry.trace")
        sys.modules["opentelemetry"] = fake_otel
        sys.modules["opentelemetry.trace"] = fake_trace
        try:
            pb = _build_pb(agent_id)
            pb.build(snapshot=_snapshot_dict(), portfolio={})
        finally:
            if orig_otel is None:
                sys.modules.pop("opentelemetry", None)
            else:
                sys.modules["opentelemetry"] = orig_otel
            if orig_trace is None:
                sys.modules.pop("opentelemetry.trace", None)
            else:
                sys.modules["opentelemetry.trace"] = orig_trace

        for field in self.TELEMETRY_FIELDS:
            assert field in span_attrs, f"{agent_id}: telemetry field '{field}' not set on span"

        assert span_attrs["prompt.builder.agent_id"] == agent_id
        assert isinstance(span_attrs["prompt.builder.prompt_size_pre"], int)
        assert isinstance(span_attrs["prompt.builder.prompt_size_post"], int)
        assert span_attrs["prompt.builder.budget"] > 0
        assert span_attrs["prompt.builder.duration_ms"] >= 0


# ── T052 / SC-Y13: fixture skill scope filter ─────────────────────────────────


class TestFixtureSkillScopeFilter:
    """T052 / SC-Y13: _test_shared loaded by all 4 agents; _test_tech only by tech."""

    def _get_skill_names(self, agent_id: str) -> list[str]:
        from cryptotrader.agents.prompt_builder import DefaultSkillProvider

        provider = DefaultSkillProvider(skills_root=FIXTURES_SKILLS)
        skills = provider.get_available_skills(agent_id, snapshot={})
        return [s.name for s in skills]

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_shared_skill_loaded_by_all_agents(self, agent_id):
        names = self._get_skill_names(agent_id)
        assert "test-shared-skill" in names, f"{agent_id}: shared skill 'test-shared-skill' not loaded"

    def test_tech_skill_only_loaded_by_tech(self):
        tech_names = self._get_skill_names("tech")
        assert "test-tech-skill" in tech_names, "tech-specific skill missing from tech agent"

    @pytest.mark.parametrize("agent_id", ["chain", "news", "macro"])
    def test_tech_skill_not_loaded_by_other_agents(self, agent_id):
        names = self._get_skill_names(agent_id)
        assert "test-tech-skill" not in names, f"{agent_id}: tech-specific skill should NOT be loaded by {agent_id}"

    def test_shared_skill_body_in_prompt(self):
        """test-shared-skill body text appears in the assembled prompt for all agents."""
        for agent_id in ("tech", "chain", "news", "macro"):
            pb = _build_pb(agent_id, skills_root=FIXTURES_SKILLS)
            sys_msg, usr_msg = pb.build(snapshot=_snapshot_dict(), portfolio={})
            full = sys_msg.content + usr_msg.content
            assert "TEST SHARED SKILL" in full, f"{agent_id}: shared skill body missing from assembled prompt"

    def test_tech_skill_body_only_in_tech_prompt(self):
        """test-tech-skill body text appears in tech prompt but not chain/news/macro."""
        tech_pb = _build_pb("tech", skills_root=FIXTURES_SKILLS)
        sys_msg, usr_msg = tech_pb.build(snapshot=_snapshot_dict(), portfolio={})
        tech_full = sys_msg.content + usr_msg.content
        assert "TEST TECH SKILL" in tech_full, "tech-skill body missing from tech prompt"

        for agent_id in ("chain", "news", "macro"):
            pb = _build_pb(agent_id, skills_root=FIXTURES_SKILLS)
            sys_msg, usr_msg = pb.build(snapshot=_snapshot_dict(), portfolio={})
            full = sys_msg.content + usr_msg.content
            assert "TEST TECH SKILL" not in full, f"{agent_id}: tech-skill body should NOT appear in {agent_id} prompt"


# ── live_steering parameter renders into prompt ───────────────────────────────


class TestSteeringInPrompt:
    """build(steering=...) injects live steering text into the user message."""

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_steering_in_prompt_when_provided(self, agent_id):
        pb = _build_pb(agent_id)
        steer = "Focus on funding-rate divergence in the next call."
        sys_msg, usr_msg = pb.build(snapshot=_snapshot_dict(), portfolio={}, steering=steer)
        full = sys_msg.content + usr_msg.content
        assert steer in full, f"{agent_id}: steering text missing from prompt"

    @pytest.mark.parametrize("agent_id", ["tech", "chain", "news", "macro"])
    def test_no_steering_section_when_empty(self, agent_id):
        pb = _build_pb(agent_id)
        sys_msg, usr_msg = pb.build(snapshot=_snapshot_dict(), portfolio={})
        full = sys_msg.content + usr_msg.content
        assert "[用户实时引导]" not in full
