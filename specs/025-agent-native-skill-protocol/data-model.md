# Phase 1 Data Model: Agent-Native Skill Protocol Layer

**Date**: 2026-05-18
**Spec**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md)

## 实体清单

本 spec 引入 / 暴露 4 个实体。其中 2 个复用既有 schema，2 个新增。

| 实体 | 来源 | spec 022 角色 |
|---|---|---|
| `PatternRecord` | spec 018 定义 + spec 021 实装 | 通过 `/api/memory/patterns` 暴露给外部 agent |
| `HeartbeatEvent` | **本 spec 新增** | `/api/events/heartbeat` 响应单元 |
| `SkillRecord (external)` | **本 spec 新增** | `/skill/<name>` 响应单元（YAML frontmatter + markdown body） |
| `ExternalSkillFetchEvent` | **本 spec 新增** | observability 内部记录（不暴露给外部 agent，仅 metrics 聚合） |

---

## PatternRecord（复用 spec 018）

**Source**：`src/cryptotrader/learning/memory.py:PatternRecord`（spec 018 定义，spec 021 已实装）

**Schema**（Pydantic v2）：
```python
class PatternRecord(BaseModel):
    agent: Literal["tech", "chain", "news", "macro"]
    slug: str                              # e.g., "sma-breakdown-short"
    applied_text: str                      # human-readable pattern description
    maturity: Literal["observed", "stable", "active", "archived"]
    pnl_track: list[float]                 # 历史 PnL 列表（filtered None）
    source_cycles: list[str]               # 触发该 pattern 的 cycle_id 列表（最多 5 个）
    regime_tags: list[str]                 # 市场 regime 标签（top 3 频次）
    created_at: datetime
    last_updated_at: datetime
    # spec 021 b239a8d 加的 LLM-distill 字段
    description: str | None = None         # LLM 生成的描述
    body: str | None = None                # LLM 生成的详细 body
```

**Loading**：`_load_pattern_record(path: Path) -> PatternRecord` helper（spec 021 实装）解析 markdown frontmatter + body。

**Storage**：filesystem `agent_memory/{agent}/patterns/{slug}.md`（git tracked + 文件锁）

**Lifecycle**（spec 018 FSM）：observed → stable → active；或 → archived（Pareto frontier 之外）

**Validation**：
- `slug` 必须匹配 `^[a-z0-9][a-z0-9-]*[a-z0-9]$`（lowercase + 连字符）
- `pnl_track` 至少 1 项（patterns 由 ≥ 5 cases 触发）
- `regime_tags` ≤ 3 项

---

## HeartbeatEvent（新增）

**Source**：本 spec 定义于 `src/api/routes/events.py:HeartbeatEvent`

**Schema**（Pydantic v2）：
```python
class HeartbeatEvent(BaseModel):
    timestamp: datetime                    # event 发生时间（UTC ISO8601）
    trace_id: str                          # UUID — cycle 内 pair-level trace
    event_type: Literal[
        "verdict",                         # must
        "trade",                           # must
        "rejection",                       # must (risk_gate)
        "phase1_rejected",                 # must (Phase 1 hard reject)
        "evolution",                       # opt
        "oco_state_change",                # opt
    ]
    pair: str | None                       # e.g., "SOL/USDT:USDT"（evolution 类无 pair）
    payload: dict[str, Any]                # event-specific schema（见下方表）
```

**Payload schema 按 event_type 分类**：

| event_type | payload 关键字段 | source 节点 |
|---|---|---|
| `verdict` | `action / scale / cf / divergence / thesis / source` | `nodes/verdict.py` |
| `trade` | `algo_id / sz / sl / tp / entry / exit / contracts / side / pos_side` | `nodes/execution.py` |
| `rejection` | `check_name / reason` | `risk/checks/*.py` |
| `phase1_rejected` | `reason ∈ {low_rr, stop_too_tight, missing_sl_tp, direction_inverted}` | `nodes/verdict.py` |
| `evolution` | `event_subtype / artifact_name`（new pattern / new skill / pareto rerank） | `ops/daemon.py` |
| `oco_state_change` | `algo_id / state ∈ {placed, cancelled, triggered}` | `execution/exchange.py` |

**Cursor 编码**：
```python
def encode_cursor(timestamp: datetime, trace_id: str) -> str:
    return base64.urlsafe_b64encode(
        f"{timestamp.isoformat()}|{trace_id}".encode()
    ).decode()

def decode_cursor(token: str) -> tuple[datetime, str]:
    decoded = base64.urlsafe_b64decode(token).decode()
    ts_str, trace_id = decoded.split("|", 1)
    return datetime.fromisoformat(ts_str), trace_id
```

**SQL view**（PostgreSQL）：
```sql
CREATE VIEW events_heartbeat AS
SELECT
    timestamp,
    trace_id,
    event_type,
    pair,
    payload
FROM journal
WHERE event_type IN (
    'verdict_decision',
    'journal_trade_committed',
    'risk_gate_rejected',
    'phase1_rejected',
    'evolution_event',
    'oco_state_change'
)
ORDER BY timestamp DESC, trace_id DESC;
```

---

## SkillRecord (external)（新增）

**Source**：本 spec 定义于 `src/api/routes/skills.py:SkillRecord`

**Schema**（Pydantic v2）：
```python
class SkillRecord(BaseModel):
    name: str                              # path-friendly name
    description: str                       # 来自 YAML frontmatter
    frontmatter: dict[str, Any]            # 完整 YAML frontmatter（含 name + description）
    body: str                              # markdown body（不含 frontmatter delimiters）
    last_modified_at: datetime             # filesystem mtime
```

**Loading**：filesystem read `agent_skills/_external/<name>/SKILL.md` → split frontmatter / body → Pydantic 构造

**Response**：
- `GET /skill/<name>` 返回 markdown 原文（`Content-Type: text/markdown`）
- `GET /skill/<name>?format=json` 返回 SkillRecord JSON

**Storage**：filesystem `agent_skills/_external/<name>/SKILL.md`（git tracked）

**Validation**：
- frontmatter 必须含 `name` 和 `description` 字段
- body 长度 < 50 KB（agent context 窗口友好）

---

## ExternalSkillFetchEvent（observability 内部）

**Source**：本 spec 内部使用，不暴露 endpoint

**Schema**（dataclass，仅内存中）：
```python
@dataclass
class ExternalSkillFetchEvent:
    timestamp: datetime
    skill_name: str                        # e.g., "cryptotrader" or "verdict-feed"
    client_identifier: str                 # API_KEY hash 后 8 位
    response_status: int                   # 200 / 401 / 404
```

**Purpose**：`ExternalSkillFetchAggregator`（24h sliding window）聚合后驱动 `external_skill_fetch_count_24h` Prometheus gauge

**Lifetime**：仅 24h 内存中（不持久化）— 复用 spec 020a `CacheMetricsAggregator` 的 deque + Lock 模式

---

## 关系图

```
External agent (Codex/Cursor/Claude Code)
    │
    │ 1. Read /skill/cryptotrader (bootstrap)
    ▼
┌─────────────────────────────────────┐
│  /skill/<name> → SkillRecord        │ ◄── agent_skills/_external/<name>/SKILL.md
│  (markdown response)                │
└─────────────────────────────────────┘
    │
    │ 2. Follow child skill links（按 SKILL.md 中的路由表）
    ▼
┌─────────────────────────────────────┐
│  /api/memory/patterns → PatternRecord[] │ ◄── agent_memory/{agent}/patterns/*.md
│  /api/events/heartbeat → HeartbeatEvent[] │ ◄── journal table (via SQL view)
│  /api/verdicts/recent → VerdictRecord[] │ ◄── (既有 spec 014/015)
│  /api/snapshot/{pair} → SnapshotPayload │ ◄── (既有)
│  /api/journal/events → JournalEvent[] │ ◄── (既有)
└─────────────────────────────────────┘
    │
    │ 3. Observability (每次 fetch 记录)
    ▼
┌─────────────────────────────────────┐
│  ExternalSkillFetchEvent (24h buffer) │
│  → ExternalSkillFetchAggregator      │
│  → external_skill_fetch_count_24h    │ → Prometheus
│                                      │
│  HeartbeatPollAggregator             │ → events_heartbeat_poll_count_24h
│  HeartbeatPollLagAggregator          │ → events_heartbeat_poll_lag_seconds
└─────────────────────────────────────┘
```

---

## State Transitions

### PatternRecord maturity FSM（spec 018 已定义，本 spec 不改）
```
observed → stable → active → archived
   ↑________↓
   (Pareto rerank can demote)
```

### HeartbeatEvent 生成时机
- 同步：节点执行时 `journal.record_*()` 即写入 → 后续 heartbeat poll 可见
- 写入到读取延迟：< 100ms（PostgreSQL commit 同步）

### SkillRecord 生成时机
- spec 022 ship 时 5 个 `_external/*.md` checked-in
- 后续修改：直接 git edit + commit → filesystem mtime 自动更新
- arena serve 重启不必要（filesystem read on each request，无缓存）

---

## 持久化策略

| 实体 | 存储 | 写入路径 | 读取路径 |
|---|---|---|---|
| PatternRecord | filesystem `agent_memory/<agent>/patterns/*.md` | `distill_patterns()` (spec 021) | `/api/memory/patterns` (本 spec) |
| HeartbeatEvent | PostgreSQL `journal` 表（view 投影） | `journal.record_*()` 各节点 | `/api/events/heartbeat` (本 spec) |
| SkillRecord | filesystem `agent_skills/_external/<name>/SKILL.md` | git commit | `/skill/<name>` (本 spec) |
| ExternalSkillFetchEvent | 内存 deque（24h） | route handler 内 inline | Prometheus scrape `/metrics` |
