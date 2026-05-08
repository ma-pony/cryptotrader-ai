# Quickstart：Agent Prompt Externalization

本文档展示 spec 017 落地后的开发者使用入口与常见操作。

## 目录结构（落地后）

```
config/agents/
├── tech.md     # 技术分析 agent prompt 配置
├── chain.md    # 链上分析 agent prompt 配置
├── news.md     # 新闻分析 agent prompt 配置
└── macro.md    # 宏观分析 agent prompt 配置

src/cryptotrader/agents/
├── prompt_builder.py   # NEW: PromptBuilder + Provider Protocol + ConfigLoader + TokenBudgetEnforcer
├── tech.py             # 重构后无 ROLE 字符串
├── chain.py
├── news.py
└── macro.py
```

## 开发者使用场景

### 场景 1：修改 TechAgent 的 prompt

修改 `config/agents/tech.md` 中 `## system_prompt` 段的内容，重启 API（或 scheduler）即可生效。**不需要改 Python 代码、不需要重新打包**。

```bash
vim config/agents/tech.md
# 修改 system_prompt 段
arena scheduler restart  # 或 docker compose restart scheduler
```

### 场景 2：新增一个可丢弃的 section

在 `config/agents/tech.md` 中：

1. 在 `sections` 列表添加新 section 名（如 `historical_volatility`）
2. 在 body 添加 `## historical_volatility` 段
3. 在 `priority` 添加该 section 的优先级数字（如 `7`，表示比 `available_skills` 还低优先级）
4. （可选）在 `slot_overrides.user_tail` 添加该 section 名

重启服务后，PromptBuilder 自动把该 section 加入拼接 + token budget 管理。

### 场景 3：调整 token budget

修改 `config/agents/<name>.md` 的 frontmatter `budget` 字段：

```yaml
budget: 12000   # 从 8000 调到 12000
```

重启服务即可。telemetry 字段 `prompt.builder.budget` 会同步更新。

### 场景 4：观察 prompt 拼接 telemetry

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
```

若 `dropped_sections` 非空，说明 token 超 budget，触发了优先级丢弃；可调高 budget 或精简 prompt。

### 场景 5：本地单元测试 PromptBuilder

```python
from pathlib import Path
from cryptotrader.agents.prompt_builder import (
    PromptBuilder, DefaultMemoryProvider, DefaultSkillProvider
)

# 用 fixture config 测试
config_dir = Path("tests/fixtures/agent_configs")
mem_p = DefaultMemoryProvider(memory_root=Path("tests/fixtures/memory"))
skl_p = DefaultSkillProvider(skills_root=Path("tests/fixtures/skills"))

pb = PromptBuilder("tech", config_dir, mem_p, skl_p, model="test")
sys_msg, usr_msg = pb.build(
    snapshot={"price": 50000, "rsi": 65},
    portfolio={"cash": 10000, "positions": []},
)
assert "技术分析" in sys_msg.content
assert "rsi" in usr_msg.content.lower()
```

### 场景 6：注入 mock Provider 测试 spec 018 进化逻辑

```python
class MockMemoryProvider:
    def get_recent_memory(self, agent_id, snapshot, k=5):
        return "### Patterns\n- 测试 pattern"

mock_pb = PromptBuilder(
    "tech",
    config_dir,
    memory_provider=MockMemoryProvider(),
    skill_provider=DefaultSkillProvider(),
    model="test",
)
sys_msg, usr_msg = mock_pb.build(snapshot={}, portfolio={})
assert "测试 pattern" in usr_msg.content
```

由于 `MemoryProvider` 是 `Protocol`，无需继承基类，鸭子类型即可。

## 常见错误

### `ConfigValidationError: Config 校验失败 [config/agents/tech.md]: 缺少必填字段: budget`

frontmatter 中没写 `budget` 字段。补上即可。

### `ConfigValidationError: ... section 'output_schema' 在 body 中未找到`

frontmatter 的 `sections` 声明了 `output_schema`，但 body 中没有 `## output_schema` 标题段落。补上：

```markdown
## output_schema

```json
{...}
```
```

### Telemetry 字段全部为 0 / 空

可能的原因：
1. PromptBuilder 不在 OpenTelemetry tracing 上下文中调用 → 检查 LangGraph node 是否在 cycle root span 内
2. spec 010 tracing 未启用 → 检查 `config/default.toml` 中 `[telemetry]` section

## 验证清单（T6 完成后跑）

```bash
# SC-X8: ROLE 常量已全部退役
grep -rn "^ROLE\s*=" src/cryptotrader/agents/
# 期望：返回空

# SC-X9: 4 个 agent 文件每个 < 150 行
wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py
# 期望：每个 < 150

# SC-X1: 4 个 config 文件存在
ls config/agents/*.md
# 期望：tech.md / chain.md / news.md / macro.md

# SC-X2 / SC-X3: 单测全 PASS
pytest tests/test_prompt_builder.py tests/test_token_budget.py tests/test_config_loader.py -v

# SC-X5: E2E PASS
pytest tests/test_e2e_prompt_externalization.py -v

# SC-X4: 4 agent 单测 PASS
pytest tests/test_tech_agent.py tests/test_chain_agent.py tests/test_news_agent.py tests/test_macro_agent.py -v
```

## 与 spec 018 的衔接点

spec 018 启动时无需修改本 spec 任何代码，仅替换 Provider 实现：

```python
# spec 018 示意
from cryptotrader.evolution.providers import EvolvingMemoryProvider, EvolvingSkillProvider

mem_p = EvolvingMemoryProvider(...)   # 满足 MemoryProvider Protocol
skl_p = EvolvingSkillProvider(...)    # 满足 SkillProvider Protocol

pb = PromptBuilder("tech", config_dir, mem_p, skl_p, model="claude-3-5-sonnet")
# PromptBuilder 行为完全不变，但 Provider 内部用了 GEPA / IDF / FSM 等进化算法
```
