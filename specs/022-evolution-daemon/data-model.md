# Data Model: Spec 020b — Evolution Daemon

本 spec **无新数据 schema 变更 / 无 migrate 脚本**。仅扩展既有运行时 entity（OTel span / dataclass / Redis sorted set）。

## 运行时 Entity（无持久化 schema 变更）

### 1. EvolutionDaemonConfig

**位置**：`src/cryptotrader/config.py`（既有文件 modify，加 dataclass）+ `config/default.toml` 加 `[evolution_daemon]` section

**字段**：

| 字段名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enabled` | bool | True | toml 启用开关；env `EVOLUTION_DAEMON_ENABLED=false` 强制覆盖 |
| `cron` | str | `"0 0 * * *"` | APScheduler CronTrigger 表达式（UTC） |
| `actions` | list[str] | `["pareto", "regime", "skill_proposal"]` | 启用 reflect actions 的子集 + 顺序 |
| `llm_model` | str | `""` | 空时通过 spec 014 fallback 解析 |
| `propose_threshold` | int | 10 | per-agent active rules 触发 skill_proposal 的阈值 |

**Validation rules**：
- `cron` 必须是合法 crontab 字符串（5 字段）；解析失败时 entrypoint 抛 ValueError exit 1
- `actions` 子集 ⊆ `{"pareto", "regime", "skill_proposal"}`；其他值忽略
- `propose_threshold ≥ 1`

---

### 2. ActionResult / RunResult（dataclass）

**位置**：`src/cryptotrader/ops/daemon.py`

**ActionResult 字段**：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `name` | str | "pareto" / "regime" / "skill_proposal" |
| `status` | Literal["PASS", "SKIP", "FAIL"] | PASS=正常完成；SKIP=soft degrade（LLM 失败 / threshold 不足）；FAIL=未捕获异常（不期望发生） |
| `duration_ms` | int | 单 action 耗时（毫秒） |
| `details` | dict | 例：`{"archived_count": 12}` / `{"changed_count": 33}` / `{"drafts_created": ["agent_skills/foo/SKILL.md.draft"]}` / `{"reason": "OpenAIAPIError"}` |

**RunResult 字段**：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `actions_run` | list[ActionResult] | 按 config.actions 顺序排列 |
| `total_duration_ms` | int | run_once 总耗时 |
| `exit_code` | int | 0=正常（含 SKIP）；1=lock timeout 整次跳过；2=未捕获异常 |

---

### 3. EvolutionEvent（Redis sorted set entry）

**位置**：Redis key `evolution_daemon:events`（sorted set，score=unix timestamp）

**Member 格式**：JSON string

| 字段名 | 类型 | 说明 |
|---|---|---|
| `ts` | int | unix timestamp（与 sorted set score 一致） |
| `event_type` | Literal["run", "llm_failure", "skill_proposal_draft"] | 事件类型 |
| `details` | dict | 事件附加信息（如 action_name, duration_ms） |

**写入路径**：daemon `run_once()` 完成后 batch zadd 到 redis；TTL 由 sliding window 控制（每次写入时 zremrangebyscore 清理过期 entry）

**读取路径**：api `/metrics` endpoint 读 redis sorted set，按 timestamp 范围聚合：
- `evolution_daemon_run_count_24h`：count(events where event_type=run AND ts >= now - 86400)
- `evolution_daemon_llm_failure_rate_24h`：count(llm_failure 事件) / count(run 事件)（24h）
- `skill_proposal_draft_count_7d`：count(skill_proposal_draft 事件)（7 day）

**Validation rules**：
- 跨进程一致：daemon 写 + api 读，redis 是 single source of truth
- TTL 自管理：写入时清理超 7d 的 entry（最大 sliding window）

---

### 4. CacheTelemetryAttr / OTel span attribute（既有扩展）

**位置**：每个 `evolution.daemon.<action>` span（spec 010 既有）

**字段（spec 020b 新增）**：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `step.status` | str | "PASS" / "SKIP" / "FAIL" |
| `step.duration_ms` | int | 单 action 耗时 |
| `step.archived_count` | int | pareto action 专用 |
| `step.regime_changed_count` | int | regime action 专用 |
| `step.drafts_created` | int | skill_proposal action 专用（drafts list 长度） |

---

## 既有 entity 字段映射（不变）

下列 entity 在 spec 020b 中**字段不变**：
- `PatternRecord`（spec 018）：仅 `maturity` 字段值在 daemon 跑时被修改（active → archived），schema 不变
- `CaseRecord`（spec 018）：仅 `regime_tags` 字段值在 daemon 跑时被批量重新计算
- `Skill`（spec 019）：本 spec 不直接修改；仅触发 propose_new_skill 写新 .draft（与 spec 019 既有路径一致）
- `Prometheus REGISTRY`（spec 015 / 020a）：本 spec 注册 3 新 Gauge，不改既有 metric

## State Transitions

仅 1 个 transition：`PatternRecord.maturity = "active" → "archived"`，触发条件 = 非 Pareto frontier 成员（spec 018 既有 transition path 复用，本 spec 在 daemon 路径内调用）。

## 数据流向

```
APScheduler CronTrigger (daily UTC midnight)
   ↓ trigger
EvolutionDaemon.run_once()
   ↓ acquire fcntl.flock(cases) → fcntl.flock(patterns)
   ↓ for action in [pareto, regime, skill_proposal]:
     ↓ try:
     ↓   _action_<name>() → ActionResult
     ↓ except (LLMError, TimeoutError):
     ↓   ActionResult(status=SKIP)
   ↓ release locks
   ↓ ZADD evolution_daemon:events (run, [llm_failure], [skill_proposal_draft]×N)
   ↓ ZREMRANGEBYSCORE evolution_daemon:events 0 (now - 7d)

api /metrics endpoint
   ↓ ZRANGEBYSCORE evolution_daemon:events (now-86400) (now)
   ↓ Gauge.set(count / failure_rate / total)
   ↓ generate_latest()
prometheus output
```

## Concurrency Model

- daemon 单 instance（无 leader election；docker-compose `evolution-daemon` service replicas=1）
- 跨进程并发：daemon vs trading cycle 通过 fcntl.flock 互斥
- 进程内并发：daemon 单 asyncio loop 顺序跑 3 actions（无并行）
- redis sorted set 操作是原子的（单次 ZADD / ZREMRANGEBYSCORE 不需要事务）
