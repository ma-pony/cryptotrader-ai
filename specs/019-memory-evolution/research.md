# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联前置研究**：[spec 016](../016-research-skill-evolution-prior-art/) / [spec 017a](../017-agent-prompt-externalization/) / [spec 017b](../018-agent-prompt-builder-integration/)
**关联 brainstorm**：[brainstorm/04-spec-018-memory-evolution.md](../../brainstorm/04-spec-018-memory-evolution.md)
**Date**: 2026-05-09

## 概述

本 spec 的 6 项关键设计决策已在 brainstorm 阶段（2026-05-09）完成。本文档不重复决策推导，仅记录最终决定 + 4 项 spot-check 修订 + 实施细节研究。

## Technical Context 中无 NEEDS CLARIFICATION 项

Brainstorm 6 项决策 + 4 项 spot-check 已消除全部 ambiguity。

## 6 项关键决策（来自 brainstorm）

| # | 决策 | 来源 |
|---|---|---|
| Q1 切分 | B 三 spec（018 Memory / 019 Skill / 020 Ops） | trilogy 范围切分 |
| Q2 范围 | C 完整（含 IVE 失败分类 LLM 调用） | spec 016 D-EV-04 全采纳 |
| Q3 case 扩展 | C 完整因果链（Trade Execution + Causal Chain + IVE Classification） | spec 016 D-MW-02 推迟项 |
| Q4 迁移 | A 一次性脚本（含 --dry-run + 幂等） | "无 fallback" 偏好 |
| Q5 IVE 频率 | A 每 case 必跑（~3600 LLM/月） | 完整失败信号 |
| Q6 容错 | C Empty placeholder + warning log | cycle 不 break |

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | 现有 case schema 抽样 | ✓ 符合假设（cycle_id / timestamp / pair / verdict_action / final_pnl / risk_gate_passed + body Agent Analyses + Verdict Reasoning） |
| 2 | ExperienceRule dataclass 命名 | ❌ 实际是 `PatternRecord`（spec 014 在 `agents/skills/schema.py:74`）；spec 修订 FR-Z6 改用 PatternRecord；success_count/failure_count 已封装在 PnLTrack |
| 3 | graph 节点名 | ❌ `risk_check`（函数）vs `risk_gate`（节点）；journal 拆 `journal_trade` / `journal_rejection`；spec 修订 FR-Z23 用真实节点名 |
| 4 | Sidebar 路径 | ❌ `web/src/components/layout/sidebar.tsx`（不是 `Sidebar.tsx`）；spec 修订 FR-Z27 改路径 |

## 实施细节决策

### Decision 1：Maturity 沿用 4 状态 + 加 archived（不重新定义 5 状态）

**Decision**：spec 014 已有 `Maturity = Literal["observed","probationary","active","deprecated"]`。本 spec 仅加 `archived` 终态，**不**用 spec 016 D-EV-03 字面提到的"draft/tested/stable/clean/mature"5 状态。

**Rationale**：
- spec 014 的 4 状态已生产运行，重新定义意味着大规模 schema 迁移
- spec 016 D-EV-03 的"5-signal"实质是"5 个状态转换信号"，不是"5 个状态"——把这些信号映射到 4+1 状态即可
- 映射：`observed`（D-EV-03 "draft"）/ `probationary`（"tested"）/ `active`（"stable"+"clean"+"mature" 合并）/ `deprecated`（人工标记）/ `archived`（自动归档）

**Alternatives considered**：
- 完全重新定义 5 状态（拒绝：违反"沿用 spec 014 schema"决策；迁移成本大）
- 仅加 `archived` 不动其他（采纳）

### Decision 2：Pareto frontier maturity_weight 映射

**Decision**：FR-Z14 的 `confidence_proxy = importance × maturity_weight`：
- `active` → 1.0
- `probationary` → 0.6
- `observed` → 0.3
- `deprecated` → 0.0
- `archived` → 0.0（不应被加载，但兜底为 0）

**Rationale**：
- active 是"已验证可信"的正常排序；probationary 减半（不完全信任）；observed 30%（早期试错期）
- deprecated/archived 为 0 让 Pareto 排序自然过滤

### Decision 3：IVE 5 诊断问题 prompt 模板

**Decision**：FR-Z15 的 5 诊断问题用单个 LLM 调用 + JSON 输出（含 5 个 yes/no/uncertain 答案 + reasoning + final classification）。不拆 5 次 LLM 调用。

**Rationale**：
- 单次调用 1.5x token vs 拆 5 次 5x token，节省成本
- LLM 在单次 prompt 中能综合 5 问题做判断，比独立判断更合理

**Prompt 模板结构**：
```
SYSTEM: 你是 crypto trading 失败诊断专家。基于以下交易 case，回答 5 诊断问题，输出 JSON。

USER:
[Case context — pair / regime_tags / pnl / agent analyses / verdict reasoning]
[Trade Execution — entry / sl / tp / exit / hit_sl / hit_tp]
[Same-regime context — last 3 cases with same regime_tags]

5 诊断问题：
1. 是否同 regime 下其他规则也亏损？(yes/no/uncertain)
2. 进出场价格是否在合理区间？(yes/no/uncertain)
3. 是否撞了停损？(yes/no/uncertain)
4. 是否符合该规则的 invalidation 条件？(yes/no/uncertain)
5. 规模是否过大？(yes/no/uncertain)

输出 JSON：
{"diagnostic_answers": [...], "reasoning": "...", "failure_type": "implementation|fundamental|noise", "confidence": 0.0-1.0}
```

### Decision 4：迁移脚本字段填充策略

**Decision**：FR-Z3 迁移脚本对旧 patterns 的新字段填默认值：
- `importance`：0.5（中性，待 reflect 调整）
- `access_count`：0（重新累计）
- `last_accessed_at`：file mtime（合理近似）
- `last_modified_at`：file mtime（合理近似）
- `fundamental_failure_streak`：0（重新累计）

**Rationale**：file mtime 是文件最后修改时间，对"近期被 reflect 修改"的语义最准确；其他字段重新累计避免脏数据。

### Decision 5：API 路由策略

**Decision**：本 spec 的 4 个 endpoints 全部用 prefix `/api/memory`：
- `GET /api/memory/rules`
- `GET /api/memory/cases`
- `GET /api/memory/transitions`
- `GET /api/memory/archived`

**Rationale**：
- spec 014/15 既有 API 路径模式：`/api/{domain}/{resource}`（如 `/api/decisions` / `/api/metrics`）
- memory 是 domain，下面 4 个 resource

### Decision 6：前端 React Query 缓存策略

**Decision**：Web `/memory` 页面 4 sections 各自用 React Query hook 独立查询：
- Rules Grid：`useMemoryRules({ agent, status })`，stale_time = 30s
- Cases Timeline：`useMemoryCases({ from, to })`，stale_time = 60s
- Archived Rules：`useArchivedRules()`，stale_time = 5min
- Recent Transitions：`useRecentTransitions({ since })`，stale_time = 30s

**Rationale**：rules/cases/transitions 频繁变化，archived 不变化（直到下次归档），各自不同 stale_time 减少 API 调用。

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice（spec 014/17a/17b 既有协议复用）
- [x] 所有 integration 已找到 pattern（API router / React Query / FSM 状态机）

Phase 0 输出完成。
