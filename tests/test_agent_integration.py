"""Integration tests — SC-002/SC-003/SC-004/SC-005/SC-010."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.config import AgentConfig, AgentsConfig

# ── shared helpers ──────────────────────────────────────────────────────────


def _fake_pb(agent_id: str = "tech") -> MagicMock:
    """Return a MagicMock satisfying the PromptBuilder interface."""
    pb = MagicMock()
    pb.build.return_value = (
        SystemMessage(content=f"system:{agent_id}"),
        HumanMessage(content="user"),
    )
    return pb


def _real_pb(agent_id: str) -> PromptBuilder:  # type: ignore[name-defined]  # noqa: F821
    """Build a real PromptBuilder pointing at the repo's config/agents/ directory."""
    from cryptotrader.agents.prompt_builder import (
        DefaultMemoryProvider,
        DefaultSkillProvider,
        PromptBuilder,
    )

    repo_root = Path(__file__).parent.parent
    return PromptBuilder(
        agent_id=agent_id,
        config_dir=repo_root / "config" / "agents",
        memory_provider=DefaultMemoryProvider(memory_root=repo_root / "agent_memory"),
        skill_provider=DefaultSkillProvider(skills_root=repo_root / "agent_skills"),
    )


@pytest.fixture
def _reset_config_cache():
    import cryptotrader.config as cfg_mod

    original = cfg_mod._cached_config
    cfg_mod._cached_config = None
    yield
    cfg_mod._cached_config = original


# ── SC-002: TOML config model end-to-end ───────────────────────────────────


class TestSC002ModelConfig:
    """SC-002: TOML config model end-to-end."""

    def test_custom_model_propagates_to_agent(self):
        agents = {
            "news_agent": AgentConfig(agent_id="news_agent", model="gpt-5"),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("news_agent", prompt_builder=_real_pb("news"))
        assert agent.model == "gpt-5"

    def test_custom_timeout_on_agent_config(self):
        agents = {
            "news_agent": AgentConfig(agent_id="news_agent", timeout_seconds=30),
        }
        cfg = AgentsConfig(_agents=agents)
        agent_cfg = cfg.get("news_agent")
        assert agent_cfg is not None
        assert agent_cfg.timeout_seconds == 30


# ── SC-003: enabled=false excludes agent from pipeline ─────────────────────


class TestSC003EnabledFalse:
    """SC-003: enabled=false excludes agent from pipeline."""

    def test_disabled_agent_not_in_list_active(self):
        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
        }
        cfg = AgentsConfig(_agents=agents)
        active_ids = [a.agent_id for a in cfg.list_active()]
        assert "tech_agent" not in active_ids

    def test_disabled_agent_build_raises(self):
        from cryptotrader.config import AgentNotFoundError

        agents = {
            "tech_agent": AgentConfig(agent_id="tech_agent", enabled=False),
        }
        cfg = AgentsConfig(_agents=agents)
        with pytest.raises(AgentNotFoundError):
            cfg.build("tech_agent", prompt_builder=_fake_pb())


# ── SC-004: Skill injection via PromptBuilder ──────────────────────────────


class TestSC004SkillInjection:
    """SC-004: Skills now injected via PromptBuilder._render_skills (spec 017b)."""

    def test_skill_rendered_by_prompt_builder(self, tmp_path):
        """DefaultSkillProvider discovers and renders skill body into prompt."""
        from cryptotrader.agents.prompt_builder import (
            DefaultMemoryProvider,
            DefaultSkillProvider,
            PromptBuilder,
        )

        # Create a shared SKILL.md in a temp skills dir (name must be kebab-case)
        shared_dir = tmp_path / "_shared"
        shared_dir.mkdir(parents=True)
        skill_md = shared_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: momentum-strategy\nscope: shared\ndescription: RSI strategy\n---\n"
            "RSI below 35 and MACD golden cross"
        )

        repo_root = Path(__file__).parent.parent
        pb = PromptBuilder(
            agent_id="tech",
            config_dir=repo_root / "config" / "agents",
            memory_provider=DefaultMemoryProvider(memory_root=repo_root / "agent_memory"),
            skill_provider=DefaultSkillProvider(skills_root=tmp_path),
        )
        sys_msg, usr_msg = pb.build(snapshot={}, portfolio={})
        # skill body appears somewhere in the assembled prompt (sys or user slot)
        full_prompt = sys_msg.content + usr_msg.content
        assert "RSI below 35" in full_prompt


# ── SC-005: Custom agent registered and buildable ──────────────────────────


class TestSC005CustomAgent:
    """SC-005: Custom agent with tools is buildable via AgentsConfig.build()."""

    def test_custom_agent_with_tools_builds(self):
        from cryptotrader.agents.base import ToolAgent

        agents = {
            "whale_agent": AgentConfig(
                agent_id="whale_agent",
                tools=["get_whale_transfers"],
            ),
        }
        cfg = AgentsConfig(_agents=agents)
        agent = cfg.build("whale_agent", prompt_builder=_fake_pb("whale_agent"))
        assert agent.agent_id == "whale_agent"
        assert isinstance(agent, ToolAgent)
        assert len(agent.tools) == 1


# ── SC-010: PromptBuilder config file drives system prompt ─────────────────


class TestSC010PromptBuilderConfig:
    """SC-010: config/agents/<id>.md system_prompt section drives the agent system prompt."""

    def test_config_system_prompt_appears_in_message(self):
        """system_prompt section from config/agents/tech.md ends up in SystemMessage."""
        pb = _real_pb("tech")
        sys_msg, _ = pb.build(snapshot={}, portfolio={})
        # tech.md system_prompt contains the TechAgent role text
        assert len(sys_msg.content) > 50
        # Verify PromptBuilder config loaded successfully
        assert pb.config.agent_id == "tech"

    def test_four_builtin_config_files_loadable(self):
        """All four builtin agent config files are valid and loadable."""
        for agent_id in ("tech", "chain", "news", "macro"):
            pb = _real_pb(agent_id)
            assert pb.config.agent_id == agent_id
            sys_msg, usr_msg = pb.build(snapshot={}, portfolio={})
            assert isinstance(sys_msg.content, str)
            assert isinstance(usr_msg.content, str)
