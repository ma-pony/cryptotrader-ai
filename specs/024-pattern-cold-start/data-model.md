# Data Model: Spec 021 — Pattern Cold-Start

本 spec **无新数据 schema 变更 / 无 migrate 脚本**。仅扩展既有 entity 写入路径 + 1 新 config 字段。

## 既有 Entity（不变）

- **PatternRecord**（spec 014 agents/skills/schema.py）：本 spec 创建新实例，schema 不变。字段全部由 `_create_pattern_from_cases()` helper 填充。
- **PnLTrack**（spec 014）：本 spec 创建实例（`PnLTrack(pnls=[...])`），schema 不变。
- **ReflectionRun**（spec 014）：本 spec 复用 `patterns_created` / `patterns_updated` / `patterns_archived` / `cases_processed` / `error` 字段。
- **ActionResult**（spec 020b ops/daemon.py）：本 spec 新增 daemon action 复用 schema，`details` dict 含 `new_count` / `updated_count` / `archived_count` / `cases_processed`。

## 新 Config 字段

### ExperienceConfig.min_cases_per_pattern

**位置**：`src/cryptotrader/config.py:ExperienceConfig`

**字段**：

| 字段名 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `min_cases_per_pattern` | int | 5 | (agent, applied_pattern) tuple 在 cases 中出现 ≥ N 次才创建 PatternRecord |

**TOML 配置**：`config/default.toml [experience]` 段加 `min_cases_per_pattern = 5`

## 新 Filesystem 实例

### Pattern Files（spec 014 既有目录结构，本 spec 创建新文件）

**路径**：`agent_memory/<agent>/patterns/<slug>.md`

**示例**（`agent_memory/tech/patterns/volume-spike-rsi-overbought.md`）：
```markdown
---
name: volume-spike-rsi-overbought
agent: tech
description: "Auto-distilled pattern: Volume Spike + RSI Overbought (from 8 cases)"
regime_tags: [high_vol, trending_up]
maturity: observed
pnl_track:
  pnls: [12.5, -3.2, 18.7, 0.0, -5.1, 22.4, 8.3, -1.8]
source_cycles: ["abc123def456...", "789xyz...", ...]
created: "2026-05-11T03:00:00Z"
version: 1
manually_edited: false
---

# Volume Spike + RSI Overbought

Auto-distilled from 8 cases.

Source cycles (first 5): ['abc123def456...', ...]
```

**Validation rules**：
- slug 仅含 `a-z0-9-`，长度 ≤ 60
- 同名 collision 自动加 `-2` / `-3` / ... 后缀
- maturity 创建时固定 `"observed"`（spec 014 FSM 初始态，后续由 evaluate_node 触发 promote）

## 运行时 Entity（无变化）

OTel span / structlog 字段：本 spec 仅加 1 新 span name `learning.distill.cold_start`，attrs：
- `patterns_created` (int)
- `patterns_updated` (int)
- `cases_processed` (int)

## 数据流向

```
agent_memory/cases/*.md                          (spec 014 既有，每 cycle 累积)
   ↓ _read_cases()
list[case dict]                                  (spec 014 既有解析)
   ↓ _parse_applied_from_body() per case
agent_pattern_counts: dict[agent, dict[pattern, list[pnl/regime_tags]]]
   ↓ NEW (spec 021): filter by min_cases_per_pattern + create PatternRecord
agent_memory/<agent>/patterns/<slug>.md          (NEW path enabled — was always empty)
   ↓ 既有 (spec 018 evaluate_node)
FSM transitions → archived rules                 (spec 018 既有路径，patterns 非空后才能工作)
   ↓
API /api/memory/{rules,transitions,archived}     (spec 014 既有读取)
   ↓
dashboard /memory page                            (实际数据非空，cold-start gap 解决)
```

## 数据迁移

**无 schema 变更**。已部署的 agent_memory/ 目录可直接跑本 spec：
1. `arena experience distill --memory-dir agent_memory --cycles-window 200` 一次性 backfill
2. daemon `pattern_extraction` daily 增量蒸馏新 cases

回退路径：删除 `agent_memory/<agent>/patterns/*.md` 文件（不影响 cases，cases 是 source of truth）。
