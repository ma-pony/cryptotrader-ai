# Contract：Agent Config 文件 Schema

**格式**：YAML frontmatter + Markdown body，文件路径 `config/agents/<agent_id>.md`

## Frontmatter Schema

```yaml
---
agent_id: <str>                    # 必填，与文件名一致
description: <str>                 # 必填，一句话描述
sections:                          # 必填，list[str]，body 必须包含的 section 名
  - system_prompt
  - user_tail
  - available_skills
  - recent_memory
  - output_schema
budget: <int>                      # 必填，token 预算上限
priority:                          # 必填，dict[str, int]，section→优先级（数字越小越保留）
  system_prompt: 1                 # 强制保留（即使设大也不丢）
  output_schema: 1                 # 强制保留
  snapshot: 2
  portfolio: 3
  user_tail: 4
  recent_memory: 5
  available_skills: 6
slot_overrides:                    # 可选，覆盖默认 slot 分配
  system:
    - system_prompt
    - available_skills
    - output_schema
  user_tail:
    - recent_memory
    - snapshot
    - portfolio
    - agent_analyses
    - user_tail
---
```

### 字段约束

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `agent_id` | str | ✓ | MUST 等于文件名（不含 `.md`） |
| `description` | str | ✓ | 非空 |
| `sections` | list[str] | ✓ | MUST 至少含 `system_prompt` / `user_tail` / `available_skills` / `recent_memory` / `output_schema` 5 项 |
| `budget` | int | ✓ | > 0 |
| `priority` | dict[str, int] | ✓ | key 必须出现在 `sections`；value 为整数 |
| `slot_overrides.system` | list[str] | ✗ | value 中每个 section MUST 在 `sections` 中 |
| `slot_overrides.user_tail` | list[str] | ✗ | 同上；与 `system` 不能交集（否则启动期校验失败） |

### 默认 slot 分配（无 slot_overrides 时）

- **system**: `system_prompt`, `available_skills`, `output_schema`
- **user_tail**: `recent_memory`, `snapshot`, `portfolio`, `agent_analyses`, `user_tail`

## Body Schema

Markdown body MUST 含以下 section（用 `## <section_name>` 标题切分）：

```markdown
## system_prompt

<role 描述、约束、风格指南；多行 markdown 文本>

## user_tail

<结尾追加给 LLM 的指令，例如"现在请输出 JSON"等；可较短>

## available_skills

<占位段，由 SkillProvider 在运行时注入；config 中可写说明文字或留空 — 运行时会被替换>

## recent_memory

<占位段，由 MemoryProvider 在运行时注入；config 中可写说明文字或留空>

## output_schema

<JSON schema 描述或 example，供 LLM 输出格式参考；本段建议含 enum / 必填字段说明>

## snapshot

<可选模板段；若提供则用 Python f-string 风格占位 {field_name} 渲染 snapshot dict；不提供则降级到 default_format()>

## portfolio

<同 snapshot 模板规则>

## agent_analyses

<可选；仅 verdict-style agent 用；本 spec 4 个 analysis agent 通常不用，可缺省>
```

### 占位 section 说明

`available_skills` / `recent_memory` 在 config 中只是 placeholder 内容（说明文字或空段）。**运行时**这些 section 的内容由 Provider 注入替换：

| Section | 注入来源 | 替换时机 |
|---|---|---|
| `available_skills` | `skill_provider.get_available_skills(...)` | 每次 build() 调用 |
| `recent_memory` | `memory_provider.get_recent_memory(...)` | 每次 build() 调用 |

config 中保留这些 section 是为了：
1. 校验：确保 frontmatter `sections` 声明与 body 一致
2. 占位文档：让人类读者知道这些 section 会被注入
3. 默认占位：当 Provider 返回空时显示该静态文本（可选 fallback）

## 完整示例：`config/agents/tech.md`

```markdown
---
agent_id: tech
description: 技术分析 agent — 关注价格趋势、成交量、动量指标
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

你是 CryptoTrader AI 系统的技术分析 agent。

职责：基于 OHLCV / 成交量 / 资金费率 / 持仓量等技术信号，给出对当前交易对在未来 4-12h 的方向性判断与置信度。

约束：
- 仅基于提供的 snapshot 数据分析，不臆测未提供的数据
- 输出 JSON 严格符合 `output_schema`
- score ∈ [-1, 1]，confidence ∈ [0, 1]
- 若信号矛盾（如 RSI 超买 + MACD 金叉），如实标记 mixed_signals: true

## user_tail

请基于上述 snapshot + memory + skills，输出 JSON 决策。

## available_skills

（运行时由 SkillProvider 注入）

## recent_memory

（运行时由 MemoryProvider 注入）

## output_schema

```json
{
  "score": -0.3,
  "confidence": 0.65,
  "direction": "bearish",
  "key_factors": ["RSI 超买", "5MA 跌破 20MA"],
  "mixed_signals": false,
  "summary": "短期回调风险增加..."
}
```
```

## 校验规则汇总

`ConfigLoader` 在加载时执行以下检查（任一失败抛 `ConfigValidationError`）：

1. 文件可读 → 失败抛 "无法读取 config 文件: {path}"
2. frontmatter 可被 `yaml.safe_load` 解析 → 失败抛 "YAML 解析失败: {err}"
3. frontmatter 必填字段齐全（`agent_id` / `description` / `sections` / `budget` / `priority`）→ 失败抛 "缺少必填字段: {field}"
4. `agent_id` == 文件名（不含扩展名）→ 失败抛 "agent_id ({val}) 与文件名 ({name}) 不匹配"
5. `budget` > 0 → 失败抛 "budget 必须 > 0"
6. `sections` 含 5 个核心 section → 失败抛 "sections 缺少必需项: {missing}"
7. body 中 `## <section_name>` 标题与 `sections` 一一对应 → 失败抛 "section '{name}' 在 body 中未找到"
8. `priority` 中每个 key 都在 `sections` 中 → 失败抛 "priority 引用了未声明的 section: {name}"
9. `slot_overrides`（若有）中引用的 section 都在 `sections` 中，且 system / user_tail 无交集 → 失败抛对应原因
