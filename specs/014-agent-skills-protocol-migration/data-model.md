# Phase 1 Data Model — 014 Agent Skills 协议迁移

## 实体清单

### 1. `Skill` —— 一条经验记录的运行时表示（dataclass）

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `agent` | `Literal["tech", "chain", "news", "macro", "shared"]` | 所属 agent；shared 表示跨 agent 知识 |
| `kind` | `Literal["pattern", "forbidden", "instruction", "knowledge"]` | 类别——决定路径与行为 |
| `name` | `str` | snake_case；与文件名 stem 一致；同 agent 同 kind 内唯一 |
| `description` | `str` | 1 句话摘要；30-500 字符；用于 prompt 注入 |
| `body` | `str` | markdown 正文；用于 `load_skill` tool 返回 |
| `regime_tags` | `list[str]` | 该 skill 适用的 regime；空列表 = 通配 |
| `pnl_track` | `PnLTrack` | PnL 反馈累计 |
| `maturity` | `Literal["observed", "probationary", "active", "deprecated"]` | 演化阶段 |
| `manually_edited` | `bool` | 用户手编 body 后置 True；reflection 仅更新 `pnl_track` |
| `created` | `datetime` | UTC ISO 8601 |
| `source_commits` | `list[str]` | 蒸馏来源的 commit hash（前 16 字符）|
| `version` | `str` | 默认 `"1.0"`；将来 schema 演进时递增 |
| `file_path` | `Path` | 运行时 absolute path（不持久化）|

校验规则（FR-004 / FR-005 / contracts/skill_frontmatter.schema.yaml）：

- `name` 必须匹配 `^[a-z][a-z0-9_]*$`
- `agent` ∈ {tech, chain, news, macro, shared}
- shared 目录下 `kind` ∈ {knowledge}（spec 隐含约定，不放 patterns/forbidden）
- 4 个 analysis agent 目录下 `kind` ∈ {pattern, forbidden, instruction}
- `instruction` kind 同 agent 至多 1 个文件（`instructions.md`）
- `description` 长度 30-500 字符
- `regime_tags` 中每个值必须出现在 `learning/regime.py` 的 `KNOWN_REGIMES` 集合中（除了空数组）
- `pnl_track.cases >= 0`，`win_rate ∈ [0, 1]`

### 2. `PnLTrack` —— 累计 PnL 元数据

```python
@dataclass
class PnLTrack:
    cases: int = 0                 # settled trades that referenced this skill via `applied:`
    win_rate: float = 0.0          # cases where pnl > 0 / cases
    avg_pnl: float = 0.0           # mean pnl across cases (USDT)
    last_active: datetime | None = None    # last update timestamp; audit only
```

更新规则（FR-021）：
- 每次 trading cycle 平仓后，遍历 verdict.reasoning 中的 `applied: <name>` 引用
- 对每个有效引用，定位对应 Skill 文件，原子更新 `pnl_track`：
  - `cases += 1`
  - `win_rate = (old_win_rate * (cases - 1) + (1 if pnl > 0 else 0)) / cases`
  - `avg_pnl = (old_avg_pnl * (cases - 1) + pnl) / cases`
  - `last_active = now()`

### 3. `Maturity` 状态机

```
observed (新蒸馏，cases < L2 阈值=5)
   ↓ cases ≥ 5 + L1+L3 通过
probationary (满足 L1/L3，但还在小样本观察)
   ↓ cases ≥ 15 + win_rate ≥ 0.55 + L3 持续显著
active (生产可用，注入 prompt 默认值)
   ↓ win_rate < 0.40 OR cases ≥ 30 且差距收敛
deprecated (移到 archive/，不再被加载)

（无时间衰减层 L5——纯 PnL 触发）
```

### 4. `AgentSkillSet` —— 单 agent 一次 cycle 的加载结果

```python
@dataclass
class AgentSkillSet:
    agent: str                              # tech | chain | news | macro
    instructions: Skill | None              # 单条；agent 行为约束
    patterns: list[Skill]                   # 已 regime 过滤
    forbidden: list[Skill]                  # 已 regime 过滤
    knowledge: list[Skill]                  # 来自 shared/，不过滤
    regime_tags: list[str]                  # 构造时输入的 regime（snapshot 时刻）

    def render_for_prompt(self) -> str:
        """Render to markdown text appended to system_message."""
```

### 5. `ReflectionRun` —— 单次反思的结构化日志

```python
@dataclass
class ReflectionRun:
    started_at: datetime
    finished_at: datetime | None = None
    commits_window: tuple[str, str] = ("", "")    # (first_hash, last_hash)
    created_skills: list[str] = field(default_factory=list)    # file paths
    updated_skills: list[str] = field(default_factory=list)
    archived_skills: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)            # [{file, error_type, msg}]
```

仅作日志/可观测，不持久化（写到 stdout / structlog）。

### 6. `LoadSkillRequest` / `LoadSkillResponse` —— tool 输入输出

```python
@dataclass
class LoadSkillRequest:
    name: str    # "<name>" or "<agent>::<name>"

# 成功响应
{"name": str, "agent": str, "kind": str, "body": str}

# 错误响应（4 类）
{"error": "skill_not_found", "name": str}
{"error": "ambiguous_name", "candidates": list[str]}
{"error": "corrupt_file", "path": str, "details": str}
{"error": "rate_limit_per_cycle", "limit": int}
```

## 关系图

```
Skill (1 file = 1 Skill)
  ├─ belongs_to → agent (5 个之一)
  ├─ has → PnLTrack
  └─ persisted_as → markdown file (frontmatter + body)

AgentSkillSet (1 per cycle per agent)
  ├─ has → 1 Skill (instructions)
  ├─ has → list[Skill] (patterns; regime-filtered)
  ├─ has → list[Skill] (forbidden; regime-filtered)
  └─ has → list[Skill] (knowledge; from shared/)

ReflectionRun (1 per reflection job execution)
  ├─ created → list[Skill]
  ├─ updated → list[Skill]
  └─ archived → list[Skill]

verdict.reasoning text
  └─ contains "applied: <name>" references
       └─ Reflection updates referenced Skill.pnl_track
```

## 文件系统物化

```
agent_skills/
├── tech/instructions.md                      # 1 个 Skill (kind=instruction)
├── tech/patterns/funding_squeeze_long.md     # 1 个 Skill (kind=pattern)
├── tech/forbidden/chase_low_volume.md        # 1 个 Skill (kind=forbidden)
├── tech/archive/old_macd_pattern.md          # 1 个 Skill (deprecated)
└── shared/funding_rate.md                    # 1 个 Skill (kind=knowledge, agent=shared)
```

每个 .md 文件 = 1 个 Skill。frontmatter 由 PyYAML 解析为 dataclass 字段。

## 不持久化的运行时状态

- `_load_skill_call_count: dict[trace_id, int]`：rate-limit 计数（每 cycle 上限 10 次，FR Edge Case）
- `_skill_set_cache: dict[(agent, regime_hash), AgentSkillSet]`：进程内 LRU；reflection 写入后失效
- `_load_skill_lock: threading.Lock`：tool 调用计数与缓存的并发保护

## 与 spec 实体的对应关系

spec.md "Key Entities" 提到：

| spec | data-model.md |
|---|---|
| Skill | ✅ 同名 |
| AgentSkillSet | ✅ 同名 |
| ReflectionRun | ✅ 同名 |
| AppliedSkillReference | （隐含在 verdict.reasoning text 解析；非独立实体）|
