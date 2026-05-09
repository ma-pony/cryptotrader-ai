# Brainstorm: Spec 018 — Memory Evolution

**Date:** 2026-05-09
**Status:** active
**Spec:** （待 `/speckit-specify` 创建，目录 `specs/019-...`，文档内引用为 "spec 018"）

## Problem Framing

trilogy 第 3 段（016 研究 / 017a 基建 / 017b 集成 / 018-020 进化算法）。spec 016 完成了 8 项目研究，spec 017a/b 完成了 PromptBuilder 基建 + 4 agent 真切换。本 spec 是 **trilogy 切分后的 Memory 子域**：把 spec 014 的 `learning/memory.py` + `learning/curation.py` 升级为可进化的 EvolvingMemoryProvider，替换 spec 017a 的 DefaultMemoryProvider（其路径 bug 一并修复）。

3 块整合：
1. **路径修复 + 数据迁移**：spec 017a 的 DefaultMemoryProvider 错读 `agent_memory/<agent>/cases.jsonl`，实际是 `agent_memory/cases/<cycle_id>.md`（全局）+ `agent_memory/<agent>/patterns/*.md`（子目录）。一次性迁移脚本把现有 ~80 case + 任何 float-maturity patterns → 新 schema
2. **EvolvingMemoryProvider 完整算法栈**：5-signal maturity FSM（D-EV-03）+ Pareto frontier 双目标排序（D-EV-02）+ IVE 失败分类（D-EV-04）+ 三连续 fundamental → 永久归档
3. **Cycle case 扩展**：cases/<id>.md 加 `## Trade Execution` + `## Causal Chain` + `## IVE Classification` 段

落地后：4 agent 真正收到进化版 memory 注入；前端 `/memory` 页面可视化 fsm_transition / ive_classification / archived_rules 事件。

## 6 项关键设计决策

### Q1 — trilogy 切分策略

**选项**：A 单 spec / B 三 spec（018/019/020）/ C 两 spec
**决策**：**B 三 spec** —— 018 Memory / 019 Skill / 020 Ops。Memory 路径修复独立 P0；Memory 算法与 Skill retrieval 互不依赖；Ops 层（cache/daemon/lineage）独立运维概念

### Q2 — Memory 子域内部范围

**选项**：A MVP / B 中度 / C 完整（含 IVE）
**决策**：**C 完整范围（含 IVE 失败分类 LLM 调用）**。理由：D-EV-04 是 spec 016 主线决策，IVE 失败分类是"三连续 fundamental → 永久归档"的前提；不含 IVE = 范围 B 退化

### Q3 — 因果链 case 扩展（D-MW-02 推迟决策）

**选项**：A 不扩展 / B 中度（仅 Trade Execution）/ C 完整 SkillClaw 因果链
**决策**：**C 完整因果链**。理由：spec 016 标记 D-MW-02 为"高价值杠杆"；IVE 失败分类的 5 诊断问题需要进出场价 + invalidation 条件等执行细节

### Q4 — 数据迁移策略

**选项**：A 一次性脚本 / B 双 schema 并存 / C Reset baseline
**决策**：**A 一次性迁移**。理由：~80 个 cycle 历史是进化算法初期"金子"；B 违反"无 fallback"偏好；C 丢历史

### Q5 — IVE 失败分类触发频率

**选项**：A 每 case 必跑 / B 仅亏损 / C 仅亏损+止损 / D 周期批量 / E 阈值
**决策**：**A 每 case 必跑**。月成本：3600 LLM 调用 × ~500 token = 1.8M token / 月（$0.27/月按 GPT-4o-mini，可忽略）；最完整失败信号

### Q6 — Provider 故障容错策略

**选项**：A Strict / B Soft fallback / C Empty placeholder + log / D Per-rule isolation
**决策**：**C Empty placeholder + warning log**。理由：cycle 关键路径不被 memory 故障 break；与 spec 014 / 015 "agent 失败兜底"哲学一致；不引入旧 fallback 路径

## US-Z5 修订

**P2 → P1，前端独立可视**：US-Z5 升级为 "新建 `/memory` 页面（4 sections：Rules Grid / Cases Timeline / Archived Rules / Recent Transitions）+ 4 个 API endpoints"。理由：用户明确 "需要直接在前端页面中观测到"。

## 6 段 Spec Outline

### Section 1 — Purpose

完成 trilogy Memory 子域：路径修复 + EvolvingMemoryProvider 完整算法栈（FSM/Pareto/IVE）+ cases schema 扩展。落地后 4 agent 真正收到进化 memory；spec 017a DefaultMemoryProvider 退役。

### Section 2 — User Stories

- **US-Z1（Architect）— 4 agent 收到进化版 memory（P1，MVP）**
- **US-Z2（Architect）— 5-signal Maturity FSM 自动转换（P1）**
- **US-Z3（Maintainer）— IVE 失败分类自动归档（P1）**
- **US-Z4（Operator）— 数据迁移与 Provider 切换零中断（P1）**
- **US-Z5（Operator/Reviewer）— 进化事件前端独立可视化（P1）** —— 新建 `/memory` 页面

### Section 3 — Functional Requirements

33 条 FR-Z1 至 FR-Z33，分 9 子模块：
- Schema & Migration（FR-Z1~Z6）
- EvolvingMemoryProvider（FR-Z7~Z10）
- 5-signal FSM（FR-Z11~Z13）
- Pareto Frontier（FR-Z14）
- IVE 失败分类（FR-Z15~Z19）
- nodes 改造（FR-Z20~Z23）
- 前端 `/memory` 页面（FR-Z24~Z30）
- Telemetry（FR-Z31）
- Migration Tooling（FR-Z32~Z33）

### Section 4 — Success Criteria

25 条 SC-Z1 至 SC-Z25，覆盖：
- Schema/Migration 验收（SC-Z1~Z4）
- 单元测试 PASS：≥10 EvolvingMemoryProvider / ≥12 FSM / ≥6 Pareto / ≥8 IVE / ≥8 migration
- E2E + 回归（SC-Z20~Z21）
- 前端测试 PASS（SC-Z16~Z19）
- Quality gate（SC-Z22~Z25：review-spec / review-plan / review-code / stamp）

### Section 5 — Dependencies & Out of Scope

**Upstream**：spec 017a / 017b / 014 / 015 / 010 / 016
**Downstream**：spec 019（Skill 子域）/ spec 020（Ops 子域）
**移至 spec 019**：SKILL.md schema 升级 / EvolvingSkillProvider / load_skill_tool 决策
**移至 spec 020**：Anthropic prompt cache / Offline reflect daemon / Git lineage

### Section 6 — Implementation Outline

4 commit 单 PR，~60 task，~8 天工作量：
- C1：迁移工具（~400 行）
- C2：算法层 FSM/Pareto/IVE（~900 行）
- C3：Provider + nodes 集成 atomic（~1100 行）
- C4：API + 前端 `/memory` + E2E（~1100 行）

## Approaches Considered（核心 6 决策综合）

每个 Q 都列了 3-5 个 alternative。整体架构哲学：
- "explicit > magic"：不引入 batch reflect daemon（推迟 spec 020）；不双 schema 并存
- "isolation > coupling"：FSM/Pareto/IVE 独立模块；Provider 容错不引入 fallback 路径
- "no fallback"：直接删 DefaultMemoryProvider 路径错代码；迁移脚本一次性
- "scope-driven evolution"：本 spec 不动 skill 进化（留 spec 019），不动 Ops（留 spec 020）

## Decision

按 6 项决策落地：B / C / C / A / A / C。整合范围：~25 文件，~3500 行 diff，4 commit 单 PR。

## Open Threads（已 spot-check 解决，2026-05-09）

### ✅ Thread 1：现有 case schema 抽样

抽查 `agent_memory/cases/0aef65c79ca4e2cd.md`：frontmatter（cycle_id / timestamp / pair / verdict_action / final_pnl / risk_gate_passed）+ body (`# Cycle Record` + `## Agent Analyses` ×4 + `## Verdict Reasoning`)。**符合 FR-Z2 假设** — 加 3 段（Trade Execution / Causal Chain / IVE Classification）安全。

### 🔴 Thread 2：Schema dataclass 命名修订（重要发现）

spec 014 实际定义在 `src/cryptotrader/agents/skills/schema.py`，**不是 `ExperienceRule`**：

| spec 018 brainstorm 假设 | spec 014 实际 | spec 修订 |
|---|---|---|
| `ExperienceRule` | `PatternRecord` | FR-Z6 改名为 PatternRecord |
| 字段 `success_count: int` / `failure_count: int` | 封装在 `pnl_track: PnLTrack` | FR-Z6/Z11 改用 `pnl_track.update(pnl)` 接口 |
| `maturity: float` | `maturity: Maturity = "observed"`（字面量类型） | FSM 设计需兼容现有 "observed" 状态值（待 spec 018 实施时核对完整 Maturity Literal 值） |
| 已有：`version: int` / `manually_edited: bool` / `source_cycles: list[str]` | 同左 | 不重复声明，沿用 |

**spec 修订**：FR-Z6 + FR-Z11 + FR-Z15 + 数据 model 段全部使用 `PatternRecord` 替代假设的 `ExperienceRule`。

### 🔴 Thread 3：graph 节点名修订（重要发现）

我 spec FR-Z23 说"在 `risk_check` 之后、`journal` 之前"，但实际：
- `build_trading_graph()` 是 HITL gate 版（路由到 `_build_full_graph()`）
- 节点名是 `risk_gate`（**函数**是 `risk_check`，**节点**是 `risk_gate`）
- backtest_graph 不含 journal 节点（"backtest engine handles stop-loss, execution, and journaling internally"）
- journal 节点名应为 `journal_trade` / `journal_rejection`（按 verdict 结果分支）

**spec 修订**：FR-Z23 改为"在 `risk_gate` 节点之后、`journal_trade` / `journal_rejection` 之前插入 `evaluate` 节点"。具体插入策略由 implementation 阶段确定（可能需要在 risk_router 后加一个 evaluate 节点共享给两条 journal 分支）。

### 🔴 Thread 4：Frontend 路径修订

| spec 018 假设 | 实际 |
|---|---|
| `web/src/components/Sidebar.tsx` | `web/src/components/layout/sidebar.tsx` |

路由顺序参考：spec 014/15 既有 `/decisions`（line 56）/ `/metrics`（line 66）；插入 `/memory` 应在 `/risk` 之后 `/metrics` 之前。

**spec 修订**：FR-Z29 路径改为 `web/src/components/layout/sidebar.tsx`。

## Decision Updates（基于 spot-check 结果，2026-05-09）

| 字段 | 旧定义 | 新定义 |
|---|---|---|
| FR-Z6 (dataclass) | "ExperienceRule dataclass 同步加新字段" | "PatternRecord dataclass（src/cryptotrader/agents/skills/schema.py 既有）同步加新字段：importance / access_count / last_accessed_at / last_modified_at / fundamental_failure_streak。`success_count / failure_count` 仍用 PnLTrack（不重复定义）" |
| FR-Z11 (FSM input) | 取 rule.success_count | 取 rule.pnl_track.successes / .losses |
| FR-Z23 (graph) | "risk_check 之后、journal 之前" | "risk_gate 节点之后、journal_trade/journal_rejection 之前" |
| FR-Z29 (frontend path) | `web/src/components/Sidebar.tsx` | `web/src/components/layout/sidebar.tsx` |
| Ship hint | — | 实施 subagent 必须先读 `src/cryptotrader/agents/skills/schema.py` 获取 PatternRecord/CaseRecord/Maturity 现状，不要凭脑补设计 |
