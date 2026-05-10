# Data Model: Spec 020c — Git Lineage

本 spec **无新数据 schema 变更 / 无 migrate 脚本**。仅扩展运行时 entity（dataclass / OTel span / git branch）。

## 运行时 Entity

### 1. CommitResult（dataclass）

**位置**：`src/cryptotrader/ops/lineage.py`

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | bool | True=commit 成功；False=commit 失败（soft fail） |
| `commit_sha` | str \| None | 成功时含 SHA；失败 / 无改动时为 None |
| `error` | str \| None | 失败时含 stderr；成功为 None |
| `duration_ms` | int | 总耗时 |

### 2. LineageMessageSummary（dict）

**位置**：传给 `GitLineageHook.commit_changes(summary: dict)`

#### Daemon summary（type="daemon"）
```python
{
  "type": "daemon",
  "actions": [
    {"name": "pareto", "status": "PASS", "details": {"archived_count": 12, "processed": 50, "transitions": [...]}},
    {"name": "regime", "status": "PASS", "details": {"changed_count": 33, "total": 142}},
    {"name": "skill_proposal", "status": "PASS", "details": {"drafts_created": ["agent_skills/foo/SKILL.md.draft"]}},
  ]
}
```

#### Transitions summary（type="transitions"）
```python
{
  "type": "transitions",
  "transitions": [
    {"rule_id": "foo", "agent_id": "tech", "old_state": "active", "new_state": "archived", "triggered_by": "fundamental_streak"},
    ...
  ]
}
```

### 3. Lineage Telemetry Attribute（OTel span attr）

**位置**：每个 `evolution.lineage.commit` span（spec 020c 新增）

| 字段 | 类型 | 说明 |
|---|---|---|
| `evolution.lineage.commit_sha` | str | 成功时 commit SHA |
| `evolution.lineage.summary_type` | str | "daemon" / "transitions" |
| `evolution.lineage.duration_ms` | int | 总耗时 |
| `evolution.lineage.error` | str | 失败时 error |
| `evolution.lineage.transitions_count` | int | transitions summary 时 |

### 4. Sliding Window Aggregator（in-process state）

**位置**：`src/cryptotrader/observability/daemon_metrics.py`

#### LineageCommitCountAggregator (24h window)
- `record()` push timestamp
- `count() -> int` — sliding window 内总 commit 数

#### LineageCommitFailureAggregator (24h window)
- `record(failed: bool)` push timestamp + failed
- `failure_rate() -> float` — failed_count / total_count

复用 spec 020a `CacheMetricsAggregator` deque + Lock 模式。

### 5. Evolution Branch（git branch）

**位置**：本地 git repo `evolution` branch（不强制 remote push）

**结构**：
- 由 `git checkout --orphan evolution` 首次创建（不继承 main 历史）
- 每次 daemon run / FSM transitions batch 添加 1 commit
- commit message 含 `Auto-Generated-By: spec-020c` trailer
- 可用 `git log evolution` 查看完整 audit trail

**Validation rules**：
- 每个 commit 必须有 `Auto-Generated-By: spec-020c` trailer（spec 020c commits）
- branch 仅含 agent_memory/ + agent_skills/ 路径改动（不混入其他文件）

## 既有 entity 字段映射（不变）

下列 entity 在 spec 020c 中**字段不变**：
- `RunResult` / `ActionResult`（spec 020b 既有）— 本 spec 仅读取，不修改
- `Transition` dataclass（spec 018 fsm.py 既有）— 不修改
- `PatternRecord` / `CaseRecord`（spec 018）— 不修改
- `Skill` dataclass（spec 019）— 不修改

## State Transitions（git branch 模型）

```
[initial state — main branch only]
       ↓ (daemon first run)
git checkout --orphan evolution
git rm -rf --cached .
git add agent_memory/ agent_skills/
git commit -m "evolution: daemon run summary..."
       ↓ (subsequent daemon runs)
git checkout evolution
git add agent_memory/ agent_skills/
git commit -m "..."
```

## Concurrency Model

- daemon 单 instance（spec 020b 约束）
- subprocess git 调用是阻塞的，但 daemon 已是独立 docker service 不影响 trading cycle
- FSM transitions 收集发生在 `_action_pareto()` 内部（同步），后续 lineage hook 触发也同步
- redis aggregator 操作原子（zadd / zrangebyscore 单命令）
