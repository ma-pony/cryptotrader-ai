"""ConfigLoader 单元测试 — 覆盖 9 条校验规则（T009）。"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from cryptotrader.agents.prompt_builder import AgentConfig, ConfigLoader, ConfigValidationError

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(tmp_path: Path, name: str, content: str) -> Path:
    """将 content 写入 tmp_path/<name>.md 并返回路径。"""
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


VALID_CONTENT = textwrap.dedent("""\
    ---
    agent_id: myagent
    description: 测试 agent
    sections:
      - system_prompt
      - user_tail
      - available_skills
      - output_schema
    budget: 8000
    priority:
      system_prompt: 1
      output_schema: 1
      user_tail: 4
      available_skills: 6
    ---

    ## system_prompt

    你是测试 agent。

    ## user_tail

    请输出 JSON。

    ## available_skills

    （运行时注入）

    ## output_schema

    ```json
    {"direction": "bullish"}
    ```
    """)


class TestConfigLoaderValidConfig:
    """规则 1+3+4+5+6+7+8：合法 config 能正常加载。"""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "myagent", VALID_CONTENT)
        cfg = ConfigLoader.load(path)
        assert isinstance(cfg, AgentConfig)
        assert cfg.agent_id == "myagent"
        assert cfg.description == "测试 agent"
        assert cfg.budget == 8000
        assert "system_prompt" in cfg.body_sections
        assert "output_schema" in cfg.body_sections
        assert cfg.priority["system_prompt"] == 1

    def test_body_sections_parsed_correctly(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, "myagent", VALID_CONTENT)
        cfg = ConfigLoader.load(path)
        assert "你是测试 agent" in cfg.body_sections["system_prompt"]
        assert "运行时注入" in cfg.body_sections["available_skills"]


class TestConfigLoaderFileErrors:
    """规则 1：文件不可读抛 ConfigValidationError。"""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.md"
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(missing)
        assert "无法读取" in str(exc_info.value)

    def test_no_frontmatter_delimiter_raises(self, tmp_path: Path) -> None:
        content = "agent_id: test\ndescription: no frontmatter\n"
        path = _write_config(tmp_path, "test", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "frontmatter" in str(exc_info.value)


class TestConfigLoaderMissingFields:
    """规则 3：缺少必填字段抛 ConfigValidationError。"""

    @pytest.mark.parametrize("missing_field", ["agent_id", "description", "sections", "budget", "priority"])
    def test_missing_required_field(self, tmp_path: Path, missing_field: str) -> None:
        import yaml

        # 构造合法 frontmatter dict，删除某个必填字段
        fm: dict = {
            "agent_id": "myagent",
            "description": "test",
            "sections": ["system_prompt", "user_tail", "available_skills", "output_schema"],
            "budget": 8000,
            "priority": {"system_prompt": 1, "output_schema": 1},
        }
        fm.pop(missing_field)
        fm_str = yaml.dump(fm)
        body = (
            "\n## system_prompt\n\nrole\n\n## user_tail\n\ntail\n\n"
            "## available_skills\n\nskills\n\n## output_schema\n\nschema\n"
        )
        content = f"---\n{fm_str}---\n{body}"
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert missing_field in str(exc_info.value)


class TestConfigLoaderAgentIdMismatch:
    """规则 4：agent_id 与文件名不匹配抛 ConfigValidationError。"""

    def test_agent_id_mismatch(self, tmp_path: Path) -> None:
        # 文件名是 myagent.md，但 agent_id 写成 otheragent
        content = VALID_CONTENT.replace("agent_id: myagent", "agent_id: otheragent")
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "不匹配" in str(exc_info.value)


class TestConfigLoaderBudget:
    """规则 5：budget ≤ 0 抛 ConfigValidationError。"""

    def test_zero_budget_raises(self, tmp_path: Path) -> None:
        content = VALID_CONTENT.replace("budget: 8000", "budget: 0")
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "budget" in str(exc_info.value)

    def test_negative_budget_raises(self, tmp_path: Path) -> None:
        content = VALID_CONTENT.replace("budget: 8000", "budget: -100")
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "budget" in str(exc_info.value)


class TestConfigLoaderSectionsMissing:
    """规则 6+7：sections 缺核心项 / body 缺段落抛 ConfigValidationError。"""

    def test_sections_missing_core_item(self, tmp_path: Path) -> None:
        # 删除 output_schema 声明
        content = VALID_CONTENT.replace("  - output_schema\n", "")
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "output_schema" in str(exc_info.value)

    def test_body_missing_declared_section(self, tmp_path: Path) -> None:
        # sections 声明了 output_schema，但 body 中删除对应 ## 段
        content = VALID_CONTENT.replace('## output_schema\n\n```json\n{"direction": "bullish"}\n```\n', "")
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "output_schema" in str(exc_info.value)


class TestConfigLoaderPriority:
    """规则 8：priority 引用未声明 section 抛 ConfigValidationError。"""

    def test_priority_references_unknown_section(self, tmp_path: Path) -> None:
        content = VALID_CONTENT.replace(
            "priority:\n  system_prompt: 1\n  output_schema: 1\n  user_tail: 4\n  available_skills: 6",
            "priority:\n  system_prompt: 1\n  output_schema: 1\n  nonexistent_section: 9",
        )
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "nonexistent_section" in str(exc_info.value)


class TestConfigLoaderSlotOverrides:
    """规则 9：slot_overrides 引用错误 / 交集冲突抛 ConfigValidationError。"""

    def test_slot_overrides_references_undeclared_section(self, tmp_path: Path) -> None:
        import yaml

        fm: dict = {
            "agent_id": "myagent",
            "description": "test",
            "sections": ["system_prompt", "user_tail", "available_skills", "output_schema"],
            "budget": 8000,
            "priority": {"system_prompt": 1, "output_schema": 1},
            "slot_overrides": {
                "system": ["system_prompt", "ghost_section"],  # ghost_section 未在 sections 声明
                "user_tail": ["user_tail"],
            },
        }
        fm_str = yaml.dump(fm)
        body = (
            "\n## system_prompt\n\nrole\n\n## user_tail\n\ntail\n\n"
            "## available_skills\n\nskills\n\n## output_schema\n\nschema\n"
        )
        content = f"---\n{fm_str}---\n{body}"
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "ghost_section" in str(exc_info.value)

    def test_slot_overrides_intersection_conflict(self, tmp_path: Path) -> None:
        import yaml

        fm: dict = {
            "agent_id": "myagent",
            "description": "test",
            "sections": ["system_prompt", "user_tail", "available_skills", "output_schema"],
            "budget": 8000,
            "priority": {"system_prompt": 1, "output_schema": 1},
            "slot_overrides": {
                "system": ["system_prompt", "available_skills"],
                "user_tail": ["available_skills", "user_tail"],  # available_skills 在两边
            },
        }
        fm_str = yaml.dump(fm)
        body = (
            "\n## system_prompt\n\nrole\n\n## user_tail\n\ntail\n\n"
            "## available_skills\n\nskills\n\n## output_schema\n\nschema\n"
        )
        content = f"---\n{fm_str}---\n{body}"
        path = _write_config(tmp_path, "myagent", content)
        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader.load(path)
        assert "交集" in str(exc_info.value)
