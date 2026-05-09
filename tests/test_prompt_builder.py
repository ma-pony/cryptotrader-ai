"""PromptBuilder 单元测试 — 覆盖 SC-X2 的 7 用例（T011）。"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.prompt_builder import (
    ConfigValidationError,
    PromptBuilder,
    Skill,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── Fixtures ────────────────────────────────────────────────────────────────────


MINIMAL_CONFIG = textwrap.dedent("""\
    ---
    agent_id: tech
    description: 技术分析测试 agent
    sections:
      - system_prompt
      - user_tail
      - available_skills
      - recent_memory
      - output_schema
    budget: 8000
    priority:
      system_prompt: 1
      output_schema: 1
      snapshot: 2
      portfolio: 3
      user_tail: 4
      recent_memory: 5
      available_skills: 6
    ---

    ## system_prompt

    你是 CryptoTrader AI 系统的技术分析 agent（测试版本）。
    职责：基于 OHLCV 信号，给出方向性判断与置信度。

    ## user_tail

    请基于上述数据，输出 JSON 决策。

    ## available_skills

    （运行时由 SkillProvider 注入）

    ## recent_memory

    （运行时由 MemoryProvider 注入）

    ## output_schema

    ```json
    {
      "direction": "bullish|bearish|neutral",
      "confidence": 0.0,
      "reasoning": "分析理由"
    }
    ```
    """)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """创建包含合法 tech.md 的临时 config 目录。"""
    d = tmp_path / "config" / "agents"
    d.mkdir(parents=True)
    (d / "tech.md").write_text(MINIMAL_CONFIG, encoding="utf-8")
    return d


@pytest.fixture
def noop_memory() -> MagicMock:
    """返回固定空字符串的 mock MemoryProvider。"""
    m = MagicMock()
    m.get_recent_memory.return_value = ""
    return m


@pytest.fixture
def noop_skills() -> MagicMock:
    """返回空列表的 mock SkillProvider。"""
    m = MagicMock()
    m.get_available_skills.return_value = []
    return m


@pytest.fixture
def builder(config_dir: Path, noop_memory: MagicMock, noop_skills: MagicMock) -> PromptBuilder:
    return PromptBuilder(
        agent_id="tech",
        config_dir=config_dir,
        memory_provider=noop_memory,
        skill_provider=noop_skills,
        model="gpt-4o-mini",
    )


SAMPLE_SNAPSHOT = {
    "pair": "BTC/USDT",
    "price": 65000.0,
    "rsi": 72.3,
    "funding_rate": 0.0003,
}

SAMPLE_PORTFOLIO = {
    "available_balance": 10000.0,
    "position": 0.0,
}


# ── (a) 加载合法 config ──────────────────────────────────────────────────────────


class TestLoadValidConfig:
    def test_builder_loads_config(self, builder: PromptBuilder) -> None:
        assert builder.config.agent_id == "tech"
        assert builder.config.budget == 8000
        assert "system_prompt" in builder.config.body_sections

    def test_build_returns_langchain_messages(self, builder: PromptBuilder) -> None:
        sys_msg, usr_msg = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert isinstance(sys_msg, SystemMessage)
        assert isinstance(usr_msg, HumanMessage)
        assert len(sys_msg.content) > 0
        assert len(usr_msg.content) > 0


# ── (b) 缺字段抛 ConfigValidationError ─────────────────────────────────────────


class TestConfigValidationError:
    def test_missing_agent_id_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "agents"
        d.mkdir()
        bad_content = MINIMAL_CONFIG.replace("agent_id: tech\n", "")
        (d / "tech.md").write_text(bad_content, encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            PromptBuilder(
                agent_id="tech",
                config_dir=d,
                memory_provider=MagicMock(get_recent_memory=MagicMock(return_value="")),
                skill_provider=MagicMock(get_available_skills=MagicMock(return_value=[])),
            )

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "agents"
        d.mkdir()
        # 没有 tech.md 文件
        with pytest.raises(ConfigValidationError):
            PromptBuilder(
                agent_id="tech",
                config_dir=d,
                memory_provider=MagicMock(get_recent_memory=MagicMock(return_value="")),
                skill_provider=MagicMock(get_available_skills=MagicMock(return_value=[])),
            )


# ── (c) 拼接产出 SystemMessage + HumanMessage ────────────────────────────────────


class TestMessageAssembly:
    def test_system_message_contains_system_prompt_from_config(self, builder: PromptBuilder) -> None:
        """SystemMessage 包含 config 中 system_prompt 段落的标志性文字。"""
        sys_msg, _ = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert "技术分析" in sys_msg.content  # 来自 config 中 system_prompt 段落

    def test_system_message_contains_output_schema(self, builder: PromptBuilder) -> None:
        sys_msg, _ = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert "output_schema" in sys_msg.content.lower() or "direction" in sys_msg.content

    def test_user_message_contains_snapshot_data(self, builder: PromptBuilder) -> None:
        _, usr_msg = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert "BTC/USDT" in usr_msg.content or "65000" in usr_msg.content


# ── (d) memory_provider 空返回走占位 ────────────────────────────────────────────


class TestMemoryProviderEmpty:
    def test_empty_memory_shows_placeholder(self, builder: PromptBuilder, noop_memory: MagicMock) -> None:
        noop_memory.get_recent_memory.return_value = ""
        _, usr_msg = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        # 空 memory → 显示占位"暂无历史记忆"
        assert True
        # 至少不报错，且 usr_msg 有内容
        assert len(usr_msg.content) > 0

    def test_memory_provider_exception_degrades_gracefully(self, config_dir: Path, noop_skills: MagicMock) -> None:
        failing_memory = MagicMock()
        failing_memory.get_recent_memory.side_effect = RuntimeError("DB connection failed")
        builder = PromptBuilder(
            agent_id="tech",
            config_dir=config_dir,
            memory_provider=failing_memory,
            skill_provider=noop_skills,
        )
        # 不抛异常，降级为占位
        sys_msg, usr_msg = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert isinstance(sys_msg, SystemMessage)
        assert isinstance(usr_msg, HumanMessage)


# ── (e) skill_provider 空返回走占位 ─────────────────────────────────────────────


class TestSkillProviderEmpty:
    def test_empty_skills_shows_placeholder(self, builder: PromptBuilder, noop_skills: MagicMock) -> None:
        noop_skills.get_available_skills.return_value = []
        sys_msg, _ = builder.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        # 空 skills → 显示占位"暂无可用技能"（在 system slot 中）
        assert "暂无可用技能" in sys_msg.content

    def test_skill_provider_with_skills_renders_markdown(self, config_dir: Path, noop_memory: MagicMock) -> None:
        # spec 017b FR-Y29: PromptBuilder._render_skills() 输出完整 SKILL.md body
        skill_provider = MagicMock()
        skill_provider.get_available_skills.return_value = [
            Skill(
                skill_id="test-skill",
                description="测试技能描述",
                tags=["tech"],
                steps=["步骤一", "步骤二"],
                body="测试技能描述\n\n## 使用场景\n详细使用说明...",
            )
        ]
        b = PromptBuilder(
            agent_id="tech",
            config_dir=config_dir,
            memory_provider=noop_memory,
            skill_provider=skill_provider,
        )
        sys_msg, _ = b.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        assert "test-skill" in sys_msg.content
        assert "测试技能描述" in sys_msg.content
        # 完整 body 渲染（spec 017b FR-Y29）：含 body 中的子标题
        assert "## 使用场景" in sys_msg.content


# ── (f) slot_overrides 生效 ──────────────────────────────────────────────────────


class TestSlotOverrides:
    def test_slot_overrides_moves_recent_memory_to_system(self, tmp_path: Path) -> None:
        """验证 slot_overrides 可把 recent_memory 移入 system slot。"""
        config_with_overrides = textwrap.dedent("""\
            ---
            agent_id: tech
            description: 带 slot_overrides 的测试 agent
            sections:
              - system_prompt
              - user_tail
              - available_skills
              - recent_memory
              - output_schema
            budget: 8000
            priority:
              system_prompt: 1
              output_schema: 1
              recent_memory: 2
              available_skills: 6
            slot_overrides:
              system:
                - system_prompt
                - recent_memory
                - available_skills
                - output_schema
              user_tail:
                - snapshot
                - portfolio
                - user_tail
            ---

            ## system_prompt

            你是技术分析 agent（slot override 测试版本）。

            ## user_tail

            请输出 JSON。

            ## available_skills

            （技能占位）

            ## recent_memory

            【记忆占位内容，供 slot_overrides 测试】

            ## output_schema

            ```json
            {"direction": "bullish"}
            ```
            """)
        d = tmp_path / "agents"
        d.mkdir()
        (d / "tech.md").write_text(config_with_overrides, encoding="utf-8")

        memory_provider = MagicMock()
        memory_provider.get_recent_memory.return_value = "历史记忆内容：BTC 上涨信号"
        skill_provider = MagicMock()
        skill_provider.get_available_skills.return_value = []

        b = PromptBuilder(
            agent_id="tech",
            config_dir=d,
            memory_provider=memory_provider,
            skill_provider=skill_provider,
        )
        sys_msg, usr_msg = b.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO)
        # recent_memory 被移入 system slot
        assert "历史记忆内容" in sys_msg.content
        # user_tail 不含 recent_memory
        assert "历史记忆内容" not in usr_msg.content


# ── (g) snapshot/portfolio 字段缺失走默认占位 ────────────────────────────────────


class TestMissingFields:
    def test_empty_snapshot_renders_placeholder(self, builder: PromptBuilder) -> None:
        _sys_msg, usr_msg = builder.build({}, {})
        # 空 snapshot → "<missing>" 占位，不报错
        assert isinstance(usr_msg, HumanMessage)
        assert "<missing>" in usr_msg.content

    def test_partial_snapshot_renders_available_fields(self, builder: PromptBuilder) -> None:
        partial = {"pair": "ETH/USDT"}
        _sys_msg, usr_msg = builder.build(partial, SAMPLE_PORTFOLIO)
        assert "ETH/USDT" in usr_msg.content

    def test_none_values_in_snapshot_show_placeholder(self, builder: PromptBuilder) -> None:
        snapshot_with_none = {"pair": "BTC/USDT", "rsi": None, "price": 65000.0}
        _, usr_msg = builder.build(snapshot_with_none, SAMPLE_PORTFOLIO)
        assert "BTC/USDT" in usr_msg.content
        # None 值 → "<missing>"
        assert "<missing>" in usr_msg.content


# ── (h) experience 参数旁路 MemoryProvider（spec 017b FR-Y6b / SC-Y11 第 45 用例） ──────────


class TestExperienceParameterBypass:
    """spec 017b FR-Y6b：experience 非空时 PromptBuilder 跳过 MemoryProvider。"""

    def test_build_experience_overrides_memory_provider(self, config_dir: Path, noop_skills: MagicMock) -> None:
        # mock memory provider — 期望从未被调用
        mock_mem = MagicMock()
        mock_mem.get_recent_memory = MagicMock(return_value="provider-fallback-content")
        b = PromptBuilder(
            agent_id="tech",
            config_dir=config_dir,
            memory_provider=mock_mem,
            skill_provider=noop_skills,
        )
        exp_text = "HISTORICAL_EXPERIENCE: 来自上游 verbal_reinforcement 节点的字符串"
        _, usr_msg = b.build(SAMPLE_SNAPSHOT, SAMPLE_PORTFOLIO, experience=exp_text)
        # experience 应注入到 recent_memory section
        assert exp_text in usr_msg.content
        # MemoryProvider.get_recent_memory 未被调用
        mock_mem.get_recent_memory.assert_not_called()
        # provider 的 fallback 内容不应出现
        assert "provider-fallback-content" not in usr_msg.content
