# Quickstart：Agent Prompt Builder Integration（spec 017b）

本文档展示 spec 017b 落地后的开发者使用入口。

## 落地后目录结构

```
config/agents/
├── tech.md     # NEW (C1)
├── chain.md    # NEW (C1)
├── news.md     # NEW (C1)
└── macro.md    # NEW (C1)

src/cryptotrader/agents/
├── base.py                  # 重构 (C2): 删 ANALYSIS_FRAMEWORK / role_description / _build_prompt
├── prompt_builder.py        # 重构 (C2): _render_skills 完整 body / scope filter / build() 加 experience
├── snapshot_renderer.py     # NEW (C1): render_crypto_snapshot()
├── tech.py / chain.py / news.py / macro.py   # 重构 (C2): 删 ROLE / 改构造器
└── skills/
    └── middleware.py        # 删除 (C2)

src/cryptotrader/
├── config.py                # 重构 (C2): 删 _resolve_role / _resolve_skills / prompt_template
└── security.py              # 修注释 (C2)

src/cryptotrader/nodes/
└── agents.py                # 重构 (C2): module-level singleton + load_skill_tool 注入
```

## 开发者使用场景

### 场景 1：修改某 agent 的 prompt

修改 `config/agents/tech.md` 中 `## system_prompt` 段的内容，重启 API（或 scheduler）即生效。**不需要改 Python 代码**。

```bash
vim config/agents/tech.md
# 修改 system_prompt 段（或 output_schema 段）
arena scheduler restart  # 或 docker compose restart scheduler
```

### 场景 2：观察 prompt 拼接 telemetry

通过现有 OpenTelemetry tracing UI（spec 010 落地）查询 cycle trace，找到对应 agent span，查看 attribute：

```
prompt.builder.agent_id          = "tech"
prompt.builder.sections_included = ["system_prompt", "user_tail", "available_skills", "recent_memory", "output_schema", "snapshot", "portfolio"]
prompt.builder.dropped_sections  = []
prompt.builder.degraded_sections = []
prompt.builder.prompt_size_pre   = 4823
prompt.builder.prompt_size_post  = 4823
prompt.builder.budget            = 8000
prompt.builder.duration_ms       = 12.4
prompt.builder.experience_source = "caller"  # 或 "provider" / "empty"
```

### 场景 3：开发自定义 agent（spec 018+ 用）

```python
from cryptotrader.agents.base import BaseAgent
from cryptotrader.agents.prompt_builder import PromptBuilder

class CustomAgent(BaseAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = ""):
        super().__init__(agent_id="custom", prompt_builder=prompt_builder, model=model)

# 实例化
from cryptotrader.nodes.agents import _get_or_build_pb
pb = _get_or_build_pb("custom", model="...")
agent = CustomAgent(prompt_builder=pb)
```

### 场景 4：本地单元测试 4 agent

```python
import pytest
from pathlib import Path
from cryptotrader.agents.tech import TechAgent
from cryptotrader.agents.prompt_builder import (
    PromptBuilder, DefaultMemoryProvider, DefaultSkillProvider
)

@pytest.fixture()
def tech_agent():
    pb = PromptBuilder(
        agent_id="tech",
        config_dir=Path("config/agents"),
        memory_provider=DefaultMemoryProvider(memory_root=Path("agent_memory")),
        skill_provider=DefaultSkillProvider(skills_root=Path("agent_skills")),
        model="test-model",
    )
    return TechAgent(prompt_builder=pb)

@pytest.mark.asyncio
async def test_tech_agent_uses_config_prompt(tech_agent, mocker, sample_snapshot):
    # mock LLM ainvoke 返回 fixture
    mock_llm = mocker.patch("langchain_openai.ChatOpenAI.ainvoke", ...)
    await tech_agent.analyze(sample_snapshot)
    # 断言 SystemMessage.content 含 config/agents/tech.md 中 system_prompt 段标志性文字
    sys_msg = mock_llm.call_args[0][0][0]
    assert "技术分析" in sys_msg.content  # 或其他 ROLE 标志性词
```

## 验证清单（C3 完成后跑）

```bash
# SC-Y4: ROLE 常量已退役
grep -rn "^ROLE\s*=" src/cryptotrader/agents/
# 期望：返回空

# SC-Y5: 4 个 agent 文件 < 150 行
wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py
# 期望：每个 < 150

# SC-Y6: middleware 已删
ls src/cryptotrader/agents/skills/middleware.py
# 期望：No such file or directory

# SC-Y7: 残留代码彻底清理
grep -rn "ANALYSIS_FRAMEWORK\|role_description\|prompt_template\|_resolve_role\|_resolve_skills\|SkillsInjectionMiddleware" src/cryptotrader/
# 期望：仅 spec 文档 / test 文档命中，src/ .py 文件无命中

# SC-Y1: 4 个 config 文件存在
ls config/agents/*.md

# SC-Y8: snapshot_renderer 单测 PASS
pytest tests/test_snapshot_renderer.py -v
# 期望：≥ 6 用例 PASS

# SC-Y9: 4 agent 单测 PASS
pytest tests/test_tech_agent.py tests/test_chain_agent.py tests/test_news_agent.py tests/test_macro_agent.py -v

# SC-Y10: E2E PASS
pytest tests/test_e2e_prompt_externalization.py -v

# SC-Y11: 017a 基建测试不回归
pytest tests/test_config_loader.py tests/test_token_budget.py tests/test_prompt_builder.py -v
# 期望：≥ 44 用例 PASS（PromptBuilder 加 experience 参数应有 +1 用例 = 45）

# SC-Y12: spec 014 / 015 既有测试不回归
pytest tests/ -x --ignore=tests/test_e2e_prompt_externalization.py 2>&1 | tail -5
```

## 与 spec 018 的衔接

spec 018 启动后，无需改 spec 017b 任何代码，仅替换 Provider 实现：

```python
# spec 018 示意（不在 017b 实现范围）
from cryptotrader.evolution.providers import EvolvingMemoryProvider, EvolvingSkillProvider

# nodes/agents.py 顶层
_memory_provider = EvolvingMemoryProvider(...)   # 替代 DefaultMemoryProvider
_skill_provider = EvolvingSkillProvider(...)     # 替代 DefaultSkillProvider

# PromptBuilder 行为不变，但 Provider 内部用了 GEPA / IDF / FSM 等进化算法
```

## 与 spec 014 verbal_reinforcement 的衔接

spec 014 的 `verbal_reinforcement` 节点输出 experience: str → state，本 spec 通过 BaseAgent.analyze 的 experience 参数 → PromptBuilder.build 的 experience 参数 → recent_memory section 完整流转，**无回归**。
