# Feature Specification: Memory Evolution（spec 018）

**Feature Branch**: `019-memory-evolution`
**Created**: 2026-05-09
**Status**: Draft
**Input**: User description: "memory-evolution — 修复 spec 017a DefaultMemoryProvider 路径 bug + 实现 EvolvingMemoryProvider（5-signal FSM + Pareto frontier + IVE 失败分类）+ cases schema 扩展 + 前端 /memory 页面"

> **目录命名说明**：本 spec 是 trilogy（016 / 017a / 017b / 018-020）切分后的 Memory 子域，逻辑名 "spec 018"。spec-kit 按递增序号分配到 019。文档内引用一律称 "spec 018" 以保持上下文连续性。

## Purpose

完成 trilogy 第 3 段的 Memory 子域：基于 spec 016 的 8 项目研究决策（D-MW-01..03 / D-EV-02..04 / D-EVAL-01），把 spec 014 的 `learning/memory.py` + `learning/curation.py` 升级为可进化的 EvolvingMemoryProvider，替换 spec 017a 的 DefaultMemoryProvider（其路径 bug 一并修复）。

3 块整合：

1. **路径修复 + 数据迁移**：spec 014 真实目录是 `agent_memory/cases/<cycle_id>.md`（全局 per-cycle markdown）+ `agent_memory/<agent>/patterns/*.md`（子目录），而 spec 017a DefaultMemoryProvider 错读为 `agent_memory/<agent>/cases.jsonl`。一次性迁移脚本把现有 ~80 case + 任何已有 patterns → 新 schema（含 Trade Execution + Causal Chain + IVE Classification + 进化字段）

2. **EvolvingMemoryProvider 完整算法栈**：替换 DefaultMemoryProvider，含
   - 5-signal maturity 状态转换（D-EV-03，沿用 spec 014 既有 4 状态枚举 `Maturity = Literal["observed", "probationary", "active", "deprecated"]`，本 spec 在该基础上增加状态转换条件 + `archived` 终态）
   - Pareto frontier 双目标排序（win_rate × confidence_proxy，D-EV-02）
   - IVE 失败分类（每个 case 跑 1 次 LLM，输出 `failure_type: implementation|fundamental|noise`，D-EV-04）
   - "三连续 fundamental → 永久归档规则"自动化
   - importance + access_count + last_accessed_at + last_modified_at + fundamental_failure_streak 元数据（D-MW-01）

3. **Cycle case 扩展**：`agent_memory/cases/<cycle_id>.md` 加 `## Trade Execution`（entry/sl/tp/exit/fill_status/hit_sl/hit_tp/exit_reason）+ `## Causal Chain`（4 agent 的 tool_calls 摘要 + verbal_reinforcement 输入 + debate 中间结果）+ `## IVE Classification`（failure_type + reasoning + confidence + 5 诊断答案）。`nodes/execution.py` + `nodes/journal.py` 写入路径同步更新

落地后：
- `agent_memory/cases/*.md` 含完整决策因果链
- `agent_memory/<agent>/patterns/*.md` 内的 maturity 仍是 `Maturity` 字面量但状态转换由 5-signal FSM 驱动
- spec 017a DefaultMemoryProvider 退役（路径错的代码删除）
- 4 agent 真正收到进化版 memory 注入
- 新建 `/memory` 前端页面可视化进化事件

本 spec **直接删旧不留 fallback**：DefaultMemoryProvider 路径错代码删除；IVE/FSM/Pareto 任一失败时返回空 markdown + warning log（cycle 关键路径不被 memory 故障 break）。回滚走 git revert。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 4 agent 收到进化版 memory（Priority: P1）🎯 MVP

作为架构师，spec 017a/b 落地的 PromptBuilder + Provider 协议真正用上"会进化的 memory"——4 agent 在每个 cycle 的 prompt 中收到的 `recent_memory` section 是经 5-signal FSM + Pareto 排序 + 元数据加权后的 top-k 案例和规则，不再是占位。

**Why P1**：spec 017a 的 DefaultMemoryProvider 是僵尸代码，spec 017b 用 experience 参数旁路过去。本 spec 是 trilogy 真正的 memory 落地。

**Independent Test**：跑 1 mocked cycle → 4 agent 的 SystemMessage / UserMessage 含 `agent_memory/<agent>/patterns/` 中至少 1 条 active 状态规则的 markdown 内容（按 importance × access_count 加权选取）。

**Acceptance Scenarios**：
1. **Given** `agent_memory/<agent>/patterns/*.md` 含 5 条规则（含不同 maturity 状态），**When** PromptBuilder 通过 EvolvingMemoryProvider 调 `get_recent_memory()`，**Then** 返回的 markdown 至少含 active/probationary 状态规则文本（不含 deprecated/archived）
2. **Given** `agent_memory/cases/*.md` 含 80+ 历史 case，**When** Provider 加载，**Then** 按 Pareto frontier 排序 top-N
3. **Given** Provider 内部任一步骤失败，**When** 调用方调 `get_recent_memory()`，**Then** 返回空字符串 + warning log 写入；cycle 不 break

---

### User Story 2 - 5-signal Maturity 自动状态转换（Priority: P1）

作为架构师，PatternRecord.maturity 状态由 5 客观信号驱动（PnL trade success count / time-since-modification / structural quality / code quality / fundamental_failure_streak），不凭主观判断。

**Why P1**：D-EV-03；现有 spec 014 的 maturity 是 `Maturity = Literal["observed","probationary","active","deprecated"]`，但状态转换条件未明确 — 本 spec 加 FSM 自动转换。

**Independent Test**：构造 fixture rule (maturity="observed", pnl_track.successes=2)，跑 evaluate_transitions → 不变（≥3 才升档）；再跑（successes=3）→ "probationary"；再跑（5 cycle 无 reflect 修改）→ "active"。

**Acceptance Scenarios**：
1. **Given** rule 状态 `observed` + `pnl_track.successes >= 3`，**When** evaluate_transitions，**Then** 升至 `probationary` + 写入 `last_modified_at`
2. **Given** rule 状态 `probationary` + `(now - last_modified_at) >= 5 cycle 或 3 day` 且无 reflect 修改，**When** evaluate，**Then** 升至 `active`
3. **Given** rule 状态 `active` + `fundamental_failure_streak >= 3`，**When** evaluate，**Then** 状态变 `archived`（移到 `.archived/`）
4. **Given** rule 在 `active` 状态被 reflect 修改，**When** evaluate，**Then** 状态降级回 `probationary`（D-EV-03 撤销条件）
5. **Given** rule 状态 `deprecated`（spec 014 既有终态），**When** evaluate，**Then** 不变（终态不再处理）

---

### User Story 3 - IVE 失败分类自动归档（Priority: P1）

作为后续维护者，每个 case（含盈利与亏损）经 IVE 5 诊断问题 LLM 分析，输出 `failure_type ∈ {implementation, fundamental, noise}`。**三连续 fundamental → 永久归档该规则**。

**Why P1**：D-EV-04；spec 014 当前所有亏损一视同仁进 reflection，可能让噪声亏损误导规则。IVE 让进化可控。

**Independent Test**：构造 fixture case (pnl=-200, hit_sl=true)，跑 IVE → mock LLM 返回 fundamental；累计第 3 次 fundamental 后 rule 变 archived；archived rule 不再注入 prompt。

**Acceptance Scenarios**：
1. **Given** case 含完整 Trade Execution 段，**When** IVE 跑，**Then** LLM prompt 含 5 诊断问题 + case context
2. **Given** rule X 收到第 3 个 fundamental 分类，**When** evaluate_transitions，**Then** rule X 状态变 `archived` + 文件移到 `.archived/`
3. **Given** rule X archived，**When** PromptBuilder 调 `get_recent_memory()`，**Then** 不含 rule X
4. **Given** 同 cycle 5 case 全跑 IVE，**When** Provider 完成，**Then** trace 含 5 个 LLM 调用 attribute

---

### User Story 4 - 数据迁移与 Provider 切换零中断（Priority: P1）

作为运维，spec 018 落地后第一次 cycle 不应失败 — 现有 ~80 case + 任何 patterns 自动迁移到新 schema，PromptBuilder 自动用 EvolvingMemoryProvider 替代 DefaultMemoryProvider。

**Why P1**：生产环境直接接入，不能因迁移破坏。

**Independent Test**：fixture 目录跑 `scripts/migrate_017_to_018.py` → fixture 旧 case + 旧 patterns 变成新 schema；跑 1 mocked cycle 全 PASS。

**Acceptance Scenarios**：
1. **Given** `agent_memory/cases/<cycle_id>.md` 不含新 3 段，**When** 迁移脚本跑，**Then** 加上默认空字段
2. **Given** patterns/*.md frontmatter 缺 `importance / access_count` 等新字段，**When** 迁移，**Then** 加默认值（`importance=0.5 / access_count=0 / fundamental_failure_streak=0`）
3. **Given** 迁移完成，**When** nodes/agents.py 启动期实例化 PromptBuilder，**Then** 注入 EvolvingMemoryProvider
4. **Given** 迁移失败，**When** 启动期初始化 Provider，**Then** 进程 fail-fast 抛异常

---

### User Story 5 - 进化事件前端独立可视化（Priority: P1）

作为运维 / reviewer，新建 `/memory` 页面，可视化 4 agent 当前 rules 状态分布（按 maturity 分组）、cases IVE 分类历史、archived rules 列表、近期 fsm_transition 事件流。

**Why P1**：用户明确"需要直接在前端页面中观测到"。

**Independent Test**：跑 1 mocked cycle 后 web UI 访问 `/memory` → 看到 rules grid 按 maturity 分组 + IVE classification 时间线 + archived rules 列表 + recent transitions。

**Acceptance Scenarios**：
1. **Given** `/memory` 页面加载，**When** 调 `/api/memory/rules?agent=tech`，**Then** 返回 200 + JSON list of patterns（含 maturity / pnl_track / importance）
2. **Given** 某 cycle 含 5 IVE classification，**When** 访问 `/memory` Cases Timeline section，**Then** 显示 5 条 classification 事件
3. **Given** rule X 刚被 archived，**When** 访问 Archived Rules section，**Then** 显示 rule X + archived 时间 + fundamental_streak 历史
4. **Given** sidebar 含 7 路由（spec 014/15 既有），**When** spec 018 落地，**Then** sidebar 含 8 路由（`/memory` 在 `/risk` 之后 `/metrics` 之前）

---

### Edge Cases

- `agent_memory/cases/` 为空 → recent_memory section 占位（"暂无历史记忆"），不报错
- `agent_memory/<agent>/patterns/` 为空（仅 .gitkeep）→ Provider 返回 empty rules list
- IVE LLM 调用 timeout / rate limit → 返回 `failure_type=noise` + warning log
- IVE LLM 输出非合法 JSON → 重试 1 次 + log，仍失败返回 noise
- FSM 评估时 `last_modified_at` 字段缺失（旧 schema 残留）→ 使用 file mtime 作 fallback
- 迁移脚本遇到损坏 frontmatter → 跳过 + warning log，不阻塞其他文件
- Provider 加载时遇到非法 schema 文件 → 跳过 + warning log
- 进程内未在 OpenTelemetry tracing 上下文 → telemetry 字段降级到 structured log（沿用 017a/b 实现）

## Requirements *(mandatory)*

### Functional Requirements

#### Schema & Migration

- **FR-Z1**：`agent_memory/<agent>/patterns/<rule_name>.md` frontmatter MUST 含 spec 014 既有字段（`name / agent / description / regime_tags / pnl_track / maturity / source_cycles / created / version / manually_edited`）+ spec 018 新增字段：`importance: float (0.0-1.0)` / `access_count: int` / `last_accessed_at: ISO8601` / `last_modified_at: ISO8601` / `fundamental_failure_streak: int`
- **FR-Z2**：`agent_memory/cases/<cycle_id>.md` body MUST 含 spec 014 既有段（`# Cycle Record` + `## Agent Analyses` × 4 + `## Verdict Reasoning`）+ 本 spec 新增段：`## Trade Execution`（entry_price / stop_loss / take_profit / actual_exit_price / fill_status / hit_sl / hit_tp / exit_reason）+ `## Causal Chain`（4 agent 的 tool_calls list 摘要 ≤500 字符 + verbal_reinforcement 输入 + debate 中间结果）+ `## IVE Classification`（failure_type / reasoning / confidence / 5 诊断答案）
- **FR-Z3**：`scripts/migrate_017_to_018.py` MUST 提供一次性迁移脚本：(a) 旧 case 加 3 个新段（默认空字段）；(b) 旧 patterns 加 5 个新字段（默认 `importance=0.5 / access_count=0 / fundamental_failure_streak=0` / `last_accessed_at = file mtime` / `last_modified_at = file mtime`）
- **FR-Z4**：迁移脚本 MUST 是幂等的（重复跑不损坏数据）
- **FR-Z5**：迁移脚本 MUST 支持 `--dry-run` 模式输出预览，不实际修改文件
- **FR-Z6**：`src/cryptotrader/agents/skills/schema.py:PatternRecord` dataclass MUST 加新字段（importance / access_count / last_accessed_at / last_modified_at / fundamental_failure_streak）；现有字段（pnl_track / maturity / version / manually_edited）保留不变。**注意**：本 spec 不重新定义 `success_count / failure_count` —— 这些封装在 `pnl_track: PnLTrack` 内部
- **FR-Z6b**：`src/cryptotrader/agents/skills/schema.py:CaseRecord` dataclass MUST 加新字段：`trade_execution: dict | None`（entry/sl/tp/exit/fill_status/hit_sl/hit_tp/exit_reason）+ `causal_chain: dict | None`（per-agent tool_calls + verbal_reinforcement_input + debate_intermediate）+ `ive_classification: dict | None`（failure_type / reasoning / confidence / diagnostic_answers）

#### EvolvingMemoryProvider

- **FR-Z7**：`src/cryptotrader/learning/evolution/provider.py:EvolvingMemoryProvider` MUST 实现 spec 017a 的 `MemoryProvider` Protocol（`get_recent_memory(agent_id, snapshot, k=5) -> str`）
- **FR-Z8**：`get_recent_memory()` 内部 MUST：
  1. 从 `agent_memory/<agent>/patterns/*.md` 读所有非 archived/deprecated 状态规则
  2. 按 Pareto frontier（win_rate × confidence_proxy）排序
  3. 二次排序：`importance × log(1 + access_count) × time_decay(last_accessed_at)`
  4. 取 top-k；写入 `access_count += 1` / `last_accessed_at = now()` 回文件
  5. 从 `agent_memory/cases/*.md` 读最近 N case（按 timestamp 倒序）
  6. 渲染 markdown：`### Patterns`（top-k rules）+ `### Cases`（top-N cases 的简洁摘要）
- **FR-Z9**：`get_recent_memory()` MUST 在内部任一步骤异常时（FSM / Pareto / IVE / IO）catch 异常 → log warning（含 stack trace + agent_id + 错误类型）→ 返回空字符串。**不抛异常**
- **FR-Z10**：`src/cryptotrader/nodes/agents.py:_get_or_build_pb` MUST 把 spec 017a/b 的 `DefaultMemoryProvider` 替换为 `EvolvingMemoryProvider`；`DefaultMemoryProvider` 类的路径错代码（spec 017a `src/cryptotrader/agents/prompt_builder.py` 内）MUST 删除

#### 5-signal Maturity FSM

- **FR-Z11**：`src/cryptotrader/learning/evolution/fsm.py:evaluate_transitions(rule: PatternRecord) -> PatternRecord | None` MUST 实现状态转换。**沿用 spec 014 既有 `Maturity = Literal["observed","probationary","active","deprecated"]`，本 spec 增加 `archived` 终态**。映射 spec 016 D-EV-03 的"5-signal"到 4+1 状态：
  - `observed → probationary`：`pnl_track.successes >= 3`（D-EV-03 信号 1）
  - `probationary → active`：`(now - last_modified_at) >= 5 cycle 或 3 day` 且 frontmatter 字段全填 + body ≤ 300 行（D-EV-03 信号 2 + 3 合并）
  - `active → archived`：`fundamental_failure_streak >= 3`（D-EV-03 信号 5）
  - `active → probationary`：rule 在 active 状态被 reflect 修改（D-EV-03 撤销条件）
  - `deprecated`：终态（spec 014 既有，本 spec 不动）
  - `archived`：终态（本 spec 新增，不再 evaluate）
- **FR-Z12**：`evaluate_transitions()` MUST 返回新 PatternRecord（含更新后状态）或 `None`（无变化）；调用方负责持久化
- **FR-Z13**：FSM evaluation MUST 在 `EvolvingMemoryProvider.evaluate_all_rules() -> list[Transition]` 内统一触发；触发时机由 spec 020 决定（trigger 接口契约）

#### Pareto Frontier 排序

- **FR-Z14**：`src/cryptotrader/learning/evolution/pareto.py:rank_rules(rules: list[PatternRecord]) -> list[PatternRecord]` MUST 实现双目标 Pareto frontier：
  - 目标 1：`win_rate = pnl_track.successes / (pnl_track.successes + pnl_track.losses)`（0 trade → win_rate=0.5 默认）
  - 目标 2：`confidence_proxy = importance × maturity_weight`（active=1.0 / probationary=0.6 / observed=0.3 / deprecated=0.0 / archived=0.0）
  - 输出：先按 Pareto 非支配前沿分层，再在层内按 `win_rate * confidence_proxy` 倒序

#### IVE 失败分类

- **FR-Z15**：`src/cryptotrader/learning/evolution/ive.py:classify_case(case: CaseRecord) -> FailureClassification` MUST 实现 D-EV-04 5 诊断问题 LLM 调用：
  1. "是否同 regime 下其他规则也亏损？" — 查 cases/ 中同 regime_tags 的 case
  2. "进出场价格是否在合理区间？" — 查 entry / exit vs 当时市场价
  3. "是否撞了停损？" — 查 hit_sl 字段
  4. "是否符合该规则的 invalidation 条件？" — LLM 判断
  5. "规模是否过大？" — 查 position_size vs 配置 max_position_size
- **FR-Z16**：`classify_case()` MUST 输出 `FailureClassification(case_id, failure_type: Literal["implementation","fundamental","noise"], reasoning: str, confidence: float, diagnostic_answers: list[str])`，写回 case 文件 `## IVE Classification` 段
- **FR-Z17**：每个 case（含盈利与亏损）MUST 跑 IVE；预算 1.8M token / 月（120 case/day × 30 day × ~500 token）
- **FR-Z18**：IVE LLM 调用失败时 MUST 返回 `failure_type=noise` + warning log
- **FR-Z19**：rule 累计 `fundamental_failure_streak >= 3` MUST 自动归档（移文件到 `agent_memory/<agent>/patterns/.archived/<rule_name>.md` + 更新 frontmatter `maturity: archived`）

#### nodes 改造

- **FR-Z20**：`src/cryptotrader/nodes/journal.py:write_case` MUST 写入新 schema（含 Trade Execution / Causal Chain / IVE Classification 默认空字段）
- **FR-Z21**：`src/cryptotrader/nodes/execution.py` MUST 在 trade 执行后把 Trade Execution 字段（entry / sl / tp / fill_status / hit_sl / hit_tp / exit_reason）回写 case
- **FR-Z22**：新增 `src/cryptotrader/nodes/evolution.py:evaluate_node(state) -> dict` 节点：cycle 末段调 `provider.evaluate_all_rules()` + `provider.classify_pending_cases()`；写 telemetry attribute（FR-Z31）
- **FR-Z23**：`src/cryptotrader/graph.py:_build_full_graph` MUST 在 `risk_gate` 节点之后、`journal_trade` / `journal_rejection` 之前插入 `evaluate` 节点（具体路由策略：在 risk_router 后加 evaluate 共享给两条 journal 分支）

#### 前端 `/memory` 页面

- **FR-Z24**：`src/api/routes/memory.py`（NEW）含 4 个 endpoints：
  - `GET /api/memory/rules?agent={id}&status={maturity}` 返回 patterns（frontmatter + body 摘要 + 当前 maturity）
  - `GET /api/memory/cases?from={iso}&to={iso}&agent={id}` 返回近期 case 含 IVE classification
  - `GET /api/memory/transitions?since={iso}` 返回近期 fsm_transition 事件
  - `GET /api/memory/archived` 返回所有 archived rules
- **FR-Z25**：`src/api/main.py` MUST 注册 memory router（`include_router(memory.router, prefix="/api/memory")`）
- **FR-Z26**：`web/src/pages/memory/MemoryPage.tsx`（NEW）含 4 sections：(a) Rules Grid（按 4 agent × 5 状态分组）；(b) Cases Timeline（最近 24h IVE classification 时间线）；(c) Archived Rules（含 archived 时间 + fundamental streak 历史）；(d) Recent Transitions（fsm_transition 事件流，10 条）
- **FR-Z27**：`web/src/components/layout/sidebar.tsx` MUST 加 `/memory` 路由项（在 `/risk` 之后、`/metrics` 之前）
- **FR-Z28**：`web/src/App.tsx` 路由表 MUST 含 `/memory` 路由（lazy-load 同其他页面）
- **FR-Z29**：`web/src/i18n/{zh-CN,en-US}.ts` MUST 加 `/memory` 页面文案（nav.memory + 4 sections title）

#### Telemetry & Observability

- **FR-Z30**：每次 `evaluate_node` 跑 MUST 写以下 OpenTelemetry span attributes：
  - `memory.evolution.fsm_transitions` (list[dict]，每项 `{rule_id, agent_id, old_state, new_state}`)
  - `memory.evolution.ive_classifications` (list[dict]，每项 `{case_id, failure_type, agent_id}`)
  - `memory.evolution.archived_rules` (list[str]，rule_id list)
  - `memory.evolution.duration_ms` (float)
  - `memory.evolution.ive_llm_calls` (int)
  - `memory.evolution.ive_llm_tokens` (int)

#### Migration Tooling

- **FR-Z31**：`scripts/migrate_017_to_018.py` MUST 在 spec 018 落地前手动运行；MUST 输出迁移日志 + 失败行 audit
- **FR-Z32**：迁移脚本 MUST 单测覆盖（`tests/test_migrate_017_to_018.py`），含 fixture 旧 case + 旧 patterns
- **FR-Z33**：迁移前 MUST 备份提示（脚本启动时 print "建议先 cp -r agent_memory agent_memory.backup_<timestamp>"）

### Key Entities

- **PatternRecord**：spec 014 既有 dataclass（`src/cryptotrader/agents/skills/schema.py:74`），本 spec 加 5 字段：importance / access_count / last_accessed_at / last_modified_at / fundamental_failure_streak
- **CaseRecord**：spec 014 既有 dataclass，本 spec 加 3 字段：trade_execution / causal_chain / ive_classification
- **PnLTrack**：spec 014 既有 dataclass（封装 successes / losses / total_pnl），本 spec 沿用不动
- **Maturity**：spec 014 既有 `Literal["observed","probationary","active","deprecated"]`，本 spec 加 `"archived"` 终态
- **FailureClassification**：本 spec 新增 dataclass（case_id / failure_type / reasoning / confidence / diagnostic_answers）
- **EvolvingMemoryProvider**：本 spec 新增 class，实现 spec 017a `MemoryProvider` Protocol
- **PromptBuilderSingleton**：spec 017b 既有 module-level dict in `nodes/agents.py`，本 spec 用 EvolvingMemoryProvider 实例替换 DefaultMemoryProvider

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-Z1**：`scripts/migrate_017_to_018.py` 跑完后 `agent_memory/cases/*.md` 全部含 3 个新段（即使内容为空占位）
- **SC-Z2**：迁移后 `agent_memory/<agent>/patterns/*.md` 全部含 5 个新字段
- **SC-Z3**：`tests/test_migrate_017_to_018.py` ≥ 8 用例 PASS
- **SC-Z4**：`tests/test_evolving_memory_provider.py` ≥ 10 用例 PASS
- **SC-Z5**：`grep -rn "class DefaultMemoryProvider" src/cryptotrader/` 返回空（spec 017a 路径错的实现退役）
- **SC-Z6**：`tests/test_fsm.py` ≥ 12 用例 PASS
- **SC-Z7**：`tests/test_pareto.py` ≥ 6 用例 PASS
- **SC-Z8**：`tests/test_ive.py` ≥ 8 用例 PASS
- **SC-Z9**：1 mocked cycle 跑后 `agent_memory/cases/<cycle_id>.md` 的 `## IVE Classification` 段非空（含 failure_type + reasoning + 5 诊断答案）
- **SC-Z10**：`tests/test_evolution_node.py` ≥ 4 用例 PASS
- **SC-Z11**：`graph.py` `_build_full_graph` 显示 evaluate 节点在 risk_gate 之后、journal_trade/rejection 之前
- **SC-Z12**：`tests/test_api_memory.py` ≥ 6 用例 PASS（4 endpoints 各覆盖正常 + 1 错误）
- **SC-Z13**：`tests/web/test_memory_page.tsx`（vitest）≥ 4 用例 PASS
- **SC-Z14**：`web/src/components/layout/sidebar.tsx` 渲染含 `/memory` 链接，在 `/risk` 之后 `/metrics` 之前
- **SC-Z15**：`tests/test_e2e_memory_evolution.py` 跑 1 mocked cycle 全链路 PASS：(a) 4 agent prompt 含 fixture active rule body；(b) cycle 末 evaluate_node 写至少 1 fsm_transition + 5 ive_classification；(c) telemetry 6 字段全填；(d) `/api/memory/rules` 返回更新后状态
- **SC-Z16**：现有 spec 014/15/17a/17b 测试不回归（`pytest tests/ -x --ignore=tests/test_e2e_memory_evolution.py 2>&1 | tail -5` 通过基线 ≥ 2173）
- **SC-Z17**：通过 `/spex:review-spec` 无 P0 / P1 issues
- **SC-Z18**：通过 `/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-Z19**：通过 `/spex:review-code` 合规评分 ≥ 95% + Deep Review Report 含 5 个 review 视角
- **SC-Z20**：通过 `/spex:verification-before-completion` stamp gate（全套测试 ≥ 2200 pass / 0 fail）

## Assumptions

- spec 017a/b 公开 API（PromptBuilder.build / Provider Protocol）签名稳定，本 spec 不破坏
- 现有 `agent_memory/cases/` 80+ 个 case 数据格式一致；迁移脚本可处理少数偏差（warning log + 跳过）
- 现有 `agent_memory/<agent>/patterns/` 目录大部分为空（仅 .gitkeep）；本 spec 落地后第一次 reflect 触发会写入 patterns
- spec 014 的 `verbal_reinforcement` 节点输出 experience: str 进 state，本 spec 不改其输出格式
- spec 014 既有 `Maturity = Literal["observed","probationary","active","deprecated"]`，本 spec 加 `"archived"` 终态向后兼容
- IVE LLM 调用使用项目默认 `models.analysis` 模型；月预算 1.8M token 可吸收（GPT-4o-mini ~$0.27/月）
- 生产环境 OpenTelemetry trace 后端（spec 010）保留期 ≥ 7 天

## Dependencies

**Upstream**：
- **spec 017a**（已合并 main，commit `cfd3acc` + merge `f1e37a9`）
- **spec 017b**（已合并 main，commit `5b65a4a` + 后续 P2 fixes `18e231e`）
- **spec 014** —— `agent_memory/<agent>/patterns/*.md` + `agent_memory/cases/<cycle_id>.md` 既有目录 + `PatternRecord` / `CaseRecord` / `Maturity` / `PnLTrack` dataclass + `learning/memory.py` IO
- **spec 015** —— `sanitize_input` 防注入函数
- **spec 010** —— OpenTelemetry tracing 基础设施
- **spec 016** —— 8 项目研究决策

**Downstream**：
- **spec 019**（待立项，trilogy Skill 子域）—— 依赖本 spec EvolvingMemoryProvider 接入路径，spec 019 在同一 module-level singleton 中注入 EvolvingSkillProvider
- **spec 020**（待立项，trilogy Ops 子域）—— 依赖本 spec `evaluate_node` + `provider.evaluate_all_rules()` 接口；加 cron daemon 调度 + git lineage + Anthropic cache
- 现有生产 cycle —— 第一次发版后 4 agent recent_memory 自动从空占位变为含 active rule body

## Out of Scope

**移至 spec 019（Skill 子域）**：
- SKILL.md schema 升级（D-DS-01）
- EvolvingSkillProvider（IDF + regime filter，D-RT-01）
- `load_skill_tool` 删除决策

**移至 spec 020（Ops 子域）**：
- Anthropic prompt cache 配置
- Offline reflect daemon 调度（D-ENG-01）
- Git lineage 自动化（D-ENG-02）

**本 spec 显式不动**：
- spec 014 既有 verbal_reinforcement 流转
- spec 014 既有 `learning/regime.py:tag_regime`
- 4 agent 类（spec 017b 稳定）
- `learning/skill_proposal.py` / `learning/curation.py`（除非本 spec 范围内必需）
- 进化算法的 GPU 加速（spec 016 NO GPU）
- `Maturity` 类型重新定义（沿用 4 状态 + 加 archived）

## Reversibility

本 spec 落地后**部分可逆**：
- **可逆**：4 agent runtime 切换、`/memory` 前端、API endpoints — git revert
- **半可逆**：cases/<id>.md schema 升级 — git revert 后旧 schema 重新生效但已写入新段不丢
- **不可逆**：迁移脚本一次性写入新字段 — 需反向迁移脚本才能"返回 spec 014 schema"

降低风险措施：
- 迁移脚本含 `--dry-run` 模式
- 迁移脚本幂等
- C3 atomic commit 单 commit 包含全部 EvolvingMemoryProvider 切换；revert 该 commit 即返 spec 017b 状态

## Implementation Outline

### Commit 序列（4 commit 单 PR）

#### C1 — 数据迁移工具 + schema 字段（无 behavior 变化）

- `scripts/migrate_017_to_018.py`（NEW，~300 行）
- `tests/test_migrate_017_to_018.py`（NEW，≥ 8 用例）
- `tests/fixtures/memory_old_format/`（NEW）
- `src/cryptotrader/agents/skills/schema.py` MODIFY — `PatternRecord` + `CaseRecord` + `Maturity` 加新字段（新字段都有 default，旧 instance 兼容）

CI 状态：所有现有测试 PASS；新增测试 PASS。
预估 diff：~600 行

#### C2 — 算法层（FSM + Pareto + IVE）

- `src/cryptotrader/learning/evolution/__init__.py`（NEW）
- `src/cryptotrader/learning/evolution/fsm.py`（NEW，~200 行）
- `src/cryptotrader/learning/evolution/pareto.py`（NEW，~100 行）
- `src/cryptotrader/learning/evolution/ive.py`（NEW，~250 行）
- `tests/test_fsm.py`（NEW，≥ 12 用例）
- `tests/test_pareto.py`（NEW，≥ 6 用例）
- `tests/test_ive.py`（NEW，≥ 8 用例）

CI 状态：算法层独立单测 PASS。
预估 diff：~900 行

#### C3 — Provider + nodes 集成（atomic 切换）

- `src/cryptotrader/learning/evolution/provider.py`（NEW，~300 行）— `EvolvingMemoryProvider`
- `src/cryptotrader/nodes/evolution.py`（NEW，~150 行）— `evaluate_node`
- `src/cryptotrader/graph.py` MODIFY — `_build_full_graph` 中插入 evaluate 节点
- `src/cryptotrader/nodes/journal.py` MODIFY — `write_case` 写新 schema
- `src/cryptotrader/nodes/execution.py` MODIFY — 回写 trade_execution
- `src/cryptotrader/nodes/agents.py` MODIFY — `_get_or_build_pb` 切换到 EvolvingMemoryProvider
- `src/cryptotrader/agents/prompt_builder.py` MODIFY — 删 DefaultMemoryProvider class
- `tests/test_evolving_memory_provider.py`（NEW，≥ 10 用例）
- `tests/test_evolution_node.py`（NEW，≥ 4 用例）

⚠️ Atomic：所有 Provider 切换 + node 接入必须同 commit，中间状态会让 4 agent 拿不到 memory。

预估 diff：~1100 行

#### C4 — 前端 `/memory` + API + E2E

- `src/api/routes/memory.py`（NEW，~150 行）
- `src/api/main.py` MODIFY — register router
- `tests/test_api_memory.py`（NEW，≥ 6 用例）
- `web/src/pages/memory/MemoryPage.tsx`（NEW，~300 行）
- `web/src/pages/memory/components/{RulesGrid,CasesTimeline,ArchivedRules,RecentTransitions}.tsx`（NEW，~80 行/个）
- `web/src/pages/memory/queries.ts`（NEW）
- `web/src/components/layout/sidebar.tsx` MODIFY
- `web/src/App.tsx` MODIFY
- `web/src/i18n/{zh-CN,en-US}.ts` MODIFY
- `tests/web/test_memory_page.tsx`（NEW，vitest，≥ 4 用例）
- `tests/test_e2e_memory_evolution.py`（NEW）

预估 diff：~1100 行

### 任务总数

约 60 task。具体细分由 `/speckit-tasks` 生成。

### 估时

| 阶段 | 工作量 |
|---|---|
| C1（迁移工具 + schema 字段） | 0.5 天 |
| C2（FSM + Pareto + IVE） | 2 天 |
| C3（Provider + nodes 集成） | 2 天 |
| C4（API + 前端 `/memory` + E2E） | 2.5 天 |
| Code review 修复 + dry-run 验证 | 1 天 |
| **合计** | **8 天** |

### Migration Strategy（生产环境）

1. 部署前在 staging 跑 `python scripts/migrate_017_to_018.py --dry-run` 验证
2. 备份 `agent_memory/` 整目录到 `agent_memory.backup_pre_018_<timestamp>/`
3. 实跑 `python scripts/migrate_017_to_018.py`（无 --dry-run）
4. 部署 spec 018 代码（`git pull` + `arena scheduler restart`）
5. 监控第 1 个 cycle：`/api/memory/rules` 返回正确数据；`evaluate_node` 写 telemetry 6 字段
6. 监控第 24h：archived rules 计数；IVE LLM 调用统计
7. 若发生回退：`git revert <C3 commit>` + 把备份的 agent_memory/ 复原
