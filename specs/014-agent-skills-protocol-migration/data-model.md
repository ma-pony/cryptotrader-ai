# Phase 1 Data Model — 014 双层架构 v2

## 实体清单

### 1. `Skill` — 高层能力包（git 跟踪）

```python
@dataclass
class Skill:
    name: str                    # tech-analysis / chain-analysis / news-analysis / macro-analysis / trading-knowledge
    description: str             # frontmatter, 30-500 chars
    body: str                    # markdown content（含 active patterns 摘要 + forbidden 摘要 + agent role）
    manually_edited: bool        # frontmatter, 默认 False
    version: str                 # frontmatter, 默认 "1.0"
    file_path: Path              # 运行时 absolute path
```

- 总数：恒为 5
- 文件位置：`agent_skills/<skill-name>/SKILL.md`
- 协议：Anthropic Skills（与 `.claude/skills/speckit-*` 对齐）

### 2. `PatternRecord` — Memory 层蒸馏数据（gitignored）

```python
@dataclass
class PatternRecord:
    name: str                       # snake_case，文件名
    agent: str                      # tech / chain / news / macro
    description: str                # 一句话摘要 (30-500 chars)
    body: str                       # 完整 markdown（条件、案例、例外）
    regime_tags: list[str]          # 适用 regime；空 = 通配
    pnl_track: PnLTrack
    maturity: Literal["observed", "probationary", "active", "deprecated"]
    manually_edited: bool           # 默认 False
    created: datetime
    source_cycles: list[str]        # 蒸馏来源的 cycle_ids
    version: str                    # 默认 "1.0"
    file_path: Path
```

- 总数：每 agent 几十到上百
- 文件位置：`agent_memory/<agent>/patterns/<name>.md`（active）或 `archive/<name>.md`（deprecated）
- 由 reflection 自动生成或更新；不进 git

### 3. `CaseRecord` — Memory 层原始数据（gitignored，永久保留）

```python
@dataclass
class CaseRecord:
    cycle_id: str                   # commit hash[:16] 或 trace_id
    timestamp: datetime
    pair: str                       # BTC/USDT:USDT 等
    agent: str                      # tech / chain / news / macro
    snapshot_summary: dict          # market data snapshot
    agent_analysis: str             # agent 节点输出的 reasoning + score
    verdict_action: Literal["long", "short", "hold", "close"]
    verdict_reasoning: str          # 含 applied: 引用
    applied_patterns: list[str]     # 从 verdict_reasoning 解析的 pattern names
    risk_gate_passed: bool
    execution_status: dict | None
    final_pnl: float | None         # 平仓后回填；None 表示未平仓
    file_path: Path
```

- 总数：~50/天/agent × N agents × 不限期 = 长期累积
- 文件位置：`agent_memory/<agent>/cases/<YYYY-MM-DD>-cycle-<hash[:8]>.md`
- 永久保留；用户可手工归档

### 4. `PnLTrack` — Pattern 的 PnL 反馈累计

```python
@dataclass
class PnLTrack:
    cases: int = 0
    win_rate: float = 0.0      # 0-1
    avg_pnl: float = 0.0       # USDT
    last_active: datetime | None = None
```

更新规则（spec FR-027）：每 case 平仓后，遍历 `applied_patterns`，对每条 pattern：
- `cases += 1`
- `win_rate = (old * (cases-1) + (1 if pnl > 0 else 0)) / cases`
- `avg_pnl = (old * (cases-1) + pnl) / cases`
- `last_active = now()`

### 5. `Maturity` 状态机

```
observed (新蒸馏，cases < 5)
   ↓ cases ≥ 5 + L1+L3 通过
probationary
   ↓ cases ≥ 15 + win_rate ≥ 0.55 + L3 持续显著
active (生产可用，进入 SKILL.md curation 候选池)
   ↓ win_rate < 0.40 OR cases ≥ 30 且差距收敛
deprecated (移到 archive/，curation 时不引用)
```

无时间衰减层（spec 已 lock-in）。

### 6. `AgentSkillSet` — Middleware 一次加载结果

```python
@dataclass
class AgentSkillSet:
    agent_id: str                       # tech / chain / news / macro
    own_skill: Skill                    # agent 自己的 SKILL.md（如 tech-analysis）
    shared_knowledge: Skill             # trading-knowledge SKILL.md
```

注：双层架构下 middleware 不再加载 patterns（patterns 在 SKILL.md body 里，已 curated）。

### 7. `ReflectionRun` — Memory 层反思日志

```python
@dataclass
class ReflectionRun:
    started_at: datetime
    finished_at: datetime | None
    cycles_window: tuple[str, str]    # (first_cycle_id, last_cycle_id)
    new_patterns: list[str]           # file paths
    updated_patterns: list[str]
    archived_patterns: list[str]
    errors: list[dict]
```

仅日志，不持久化。

### 8. `CurationRun` — SKILL.md 整理日志

```python
@dataclass
class CurationRun:
    started_at: datetime
    skill_name: str                   # tech-analysis 等
    triggered_by: Literal["manual", "llm", "cron"]
    active_patterns_used: list[str]   # 输入的 pattern names
    output_path: Path                 # SKILL.md.draft 或 SKILL.md
    skipped_reason: str | None        # 如 manually_edited=True
```

### 9. `LoadSkillRequest` / `LoadSkillResponse`

```python
class LoadSkillRequest(BaseModel):
    name: str    # 5 个 skill name 之一

# 成功
{"name": "tech-analysis", "body": "..."}

# 错误（4 类）
{"error": "skill_not_found", "name": "..."}
{"error": "corrupt_file", "path": "...", "details": "..."}
{"error": "rate_limit_per_cycle", "limit": 10}
{"error": "skill_dir_missing"}
```

## 关系图

```
agent_memory/<agent>/cases/<cycle_id>.md     [CaseRecord]
                ↓ reflection 周期蒸馏
agent_memory/<agent>/patterns/<name>.md      [PatternRecord]
                ↓ curation 整理 (manual or LLM)
agent_skills/<skill>/SKILL.md                [Skill]
                ↑ middleware.wrap_model_call
                ↑ load_skill(name) tool
            agent prompt
```

## 文件系统物化

```
.gitignore
+ agent_memory/

agent_memory/                            # gitignored
├── tech/
│   ├── cases/2026-05-06-cycle-7ffc.md
│   ├── patterns/funding_squeeze_long.md
│   └── archive/old_macd_pattern.md
├── chain/{cases,patterns,archive}/
├── news/{cases,patterns,archive}/
└── macro/{cases,patterns,archive}/

agent_skills/                            # git tracked, 5 个目录
├── tech-analysis/SKILL.md
├── chain-analysis/SKILL.md
├── news-analysis/SKILL.md
├── macro-analysis/SKILL.md
└── trading-knowledge/SKILL.md
```

## 不持久化运行时状态

- `_load_skill_call_count: dict[trace_id, int]`：rate-limit 计数（≤ 10/cycle）
- `_skill_cache: dict[skill_name, Skill]`：进程内缓存；curation 写入后失效

## 与 spec Key Entities 对应

| spec | data-model.md |
|---|---|
| Skill | ✅ 同名（5 个之一）|
| PatternRecord | ✅ 同名（memory 层数据）|
| CaseRecord | ✅ 同名（memory 层数据）|
| AgentSkillSet | ✅ 同名（middleware 加载结果）|
| ReflectionRun | ✅ 同名 |
| CurationRun | ✅ 同名（新增）|
