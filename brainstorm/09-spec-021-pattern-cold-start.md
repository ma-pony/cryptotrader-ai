# Spec 021 — Pattern Cold-Start（trilogy 进化系统数据链补完）

**关联 spec**：[014](../specs/014-trilogy-research/) / [018](../specs/019-memory-evolution/) / [020b](../specs/022-evolution-daemon/)
**Date**: 2026-05-11
**Status**: brainstorm 完成，待 ship

## 问题陈述

trilogy 落地后实测发现 trader-grade 进化系统数据链断在 `distill` 这环：

- `agent_memory/cases/` — **246 文件**（每 cycle 决策快照 ✅）
- `agent_memory/<agent>/patterns/` — **0 文件**（应从 cases 提炼）
- `agent_memory/transitions/` — 目录不存在

**根因**：
1. **`distill_patterns()` 函数只更新已有 patterns 的 maturity，没有创建新 patterns 的代码**（patterns_dir 空 → 空循环 → 0 输出）
2. **`run_reflection()` 节点函数定义在 `nodes/reflection.py:18` 但从未在 graph / scheduler / CLI 注册触发**

→ 整 trilogy 数据链 `cases → distill → patterns → evaluate (FSM) → transitions` 断裂；spec 018 Memory Evolution + spec 019 Skill Evolution + spec 020b Evolution Daemon 全部建立在 patterns 非空假设上，cold-start gap 导致 dashboard 整段空。

## 5 项关键决策

### Q1：Pattern 创建触发器位置

**Decision**：A — `distill_patterns()` 内置 cold-start。

**Rationale**：
- 与 spec 014 既有函数命名 `distill_patterns`（蒸馏 = 提炼新 + 更新已有）语义吻合
- 最少 surgical 改动（单文件 ~50-80 LOC）
- 调用方接口不变（向后兼容）

### Q2：Run_reflection 调度频率

**Decision**：C + D — daemon 每天 1 次 + CLI manual。

**Rationale**：
- daemon 自动每天跑，与 spec 020b Evolution Daemon 模式一致（加第 4 个 action `pattern_extraction`）
- CLI `arena experience distill` 紧急触发（首次 backfill + dev debug）
- 不 per-cycle 跑（重复蒸馏成本高 + 数据增量小）

### Q3：Pattern 创建阈值

**Decision**：默认 `min_cases_per_pattern = 5` + 配置化。

**Rationale**：
- 5 平衡 patterns 集增长 vs 噪声（3 太低，10 太严格）
- 配置化通过 `[experience] min_cases_per_pattern = 5` 暴露
- 与 spec 014 `min_commits_required = 10` 不同：那是触发 reflection 的最小数据量；这是单个 pattern 被创建的最小观测次数

### Q4：新 Pattern 初始 maturity

**Decision**：默认 `maturity = "observed"`（spec 014 FSM 初始态）。

**Rationale**：
- 与 spec 014 `PatternRecord(maturity: Maturity = "observed")` 默认一致
- 后续由 spec 018 5-signal FSM 自动 promote（observed → probationary → active）

### Q5：Spec 切分 / Scope

**Decision**：A — 完整 spec ship（spec.md / plan.md / tasks.md / review）。

**Rationale**：
- 跨 4 模块：`learning/memory.py` distill + `nodes/reflection.py` graph wire + `ops/daemon.py` 加 action + `cli/main.py` 加 command + tests
- ~150-200 LOC + 单测 + e2e + dashboard 验证
- 符合 spec 标准切分（与 spec 020b/c 体量类似）

## 5 项 spot-check 结果（2026-05-11）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `_parse_applied_from_body()` + `VALID_AGENT_IDS` 既有 | ✅ memory.py:476 可直接复用 |
| 2 | `PatternRecord` schema 完整 | ✅ name/agent/desc/body/regime_tags/pnl_track/maturity/source_cycles 字段全 |
| 3 | `ExperienceConfig` `min_cases_per_pattern` 字段 | ⚠️ 不存在，FR 需新加；既有 `min_commits_required=10` / `every_n_cycles=20` 字段语义不同不能复用 |
| 4 | daemon `actions` list 配置驱动 | ✅ ops/daemon.py:110 `for action_name in self.config.actions` 加第 4 个 action 容易 |
| 5 | CLI `arena experience distill` 命令 | ⚠️ 未实现，需新建（cli/main.py 加 typer command） |

## 6 节速览

### 1. Purpose

补完 trilogy 进化系统 cold-start 数据链：`distill_patterns()` 加创建新 patterns 逻辑 + `run_reflection` 接入 evolution-daemon（daily）+ CLI manual trigger。

### 2. User Stories

- **US-P1（P1）Trader**：dashboard `/memory` 页能看到 patterns / transitions / archived 实际数据（非空）
- **US-P2（P1）Architect**：trilogy 进化系统数据链不再断裂（246 cases → ~15-30 patterns → FSM transitions → dashboard）
- **US-P3（P2）SRE**：evolution-daemon daily run 跑第 4 个 action `pattern_extraction`，与 pareto/regime/skill_proposal 并列
- **US-P4（P2）Dev**：`arena experience distill --once` CLI 紧急触发首次 backfill + dev debug

### 3. Functional Requirements（~14 条）

- **FR-P1**：`src/cryptotrader/learning/memory.py:distill_patterns()` MUST 加 cold-start 逻辑：扫 cases 提炼 `(agent, applied_pattern)` 频次 + PnL → 当频次 ≥ `min_cases_per_pattern` 创建 `PatternRecord(maturity="observed")` 写到 `agent_memory/<agent>/patterns/<pattern_name>.md`
- **FR-P2**：新 pattern 字段：`name`（applied_pattern slug） / `agent`（per-agent） / `description`（自动生成）/ `body`（含 first 5 cases 的 cycle_id list）/ `pnl_track`（cases 中的 PnL 列表）/ `regime_tags`（cases 中 regime tags 投票 top）
- **FR-P3**：`config/default.toml` `[experience]` 加 `min_cases_per_pattern = 5` 字段
- **FR-P4**：`src/cryptotrader/config.py:ExperienceConfig` 加 `min_cases_per_pattern: int = 5` 字段
- **FR-P5**：`src/cryptotrader/ops/daemon.py` 加第 4 个 action `pattern_extraction`：调 `distill_patterns()` + 记录 ActionResult.details（new_count / updated_count / archived_count）
- **FR-P6**：`config/default.toml` `[evolution_daemon].actions` 默认列表加 `"pattern_extraction"`
- **FR-P7**：`src/cli/main.py` 加 `arena experience distill` typer command：默认跑 `distill_patterns(memory_dir, cycles_window=lookback_commits)` 并输出 ReflectionRun summary
- **FR-P8**：`distill_patterns()` 改动 MUST 向后兼容：已有 patterns 的 maturity 更新行为不变
- **FR-P9**：cold-start 路径 MUST 写 OTel span `learning.distill.cold_start`，attr `patterns_created` / `cases_processed`
- **FR-P10**：cold-start failed pattern 写入 MUST log warning，不影响其他 patterns 创建
- **FR-P11**：单测 `tests/test_distill_patterns_cold_start.py`：5 用例覆盖（empty cases / single agent / freq below threshold / freq above / pnl_track populated）
- **FR-P12**：e2e `tests/test_e2e_pattern_cold_start.py`：fixture 200+ cases → 跑 daemon `pattern_extraction` action → 验证 ≥ 3 patterns 创建
- **FR-P13**：dashboard `/memory` 页面端到端 smoke：API `/api/memory/rules` 返回非空 list
- **FR-P14**：本 spec 直接删旧不留 fallback（用户偏好延续）

### 4. Success Criteria（~10 条）

- SC-P1：`uv run arena experience distill --memory-dir agent_memory --cycles-window 200` 一次成功创建 ≥ 1 patterns
- SC-P2：跑后 `find agent_memory/{tech,chain,news,macro}/patterns -name "*.md" | wc -l` ≥ 3
- SC-P3：dashboard `/api/memory/rules` 返回 `total > 0` 且 items 含 maturity="observed" 的 PatternRecord
- SC-P4：evolution-daemon `--once` 跑后 4 actions 全 PASS（含新 `pattern_extraction`）
- SC-P5：tests/test_distill_patterns_cold_start.py 5 PASS
- SC-P6：tests/test_e2e_pattern_cold_start.py PASS
- SC-P7：spec 014/15/17a/17b/18/19/20a/20b/20c 既有测试不回归（baseline 2458 → ≥ 2458 pass / 0 fail）
- SC-P8：`/spex:review-spec` 无 P0/P1
- SC-P9：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md
- SC-P10：本 spec 单 PR ≤ 4 commit（C1 distill + config / C2 daemon + CLI / C3 tests / C4 E2E + final gate）

### 5. Dependencies

- **Upstream**：spec 014（PatternRecord schema）/ spec 018（FSM）/ spec 020b（Evolution Daemon）
- **Downstream**：无（trilogy 数据链 cold-start gap 收尾）

**External tooling**：无新 runtime 依赖

### 6. Out of Scope

- ❌ Pattern auto-deletion / pruning（spec 018 既有 maturity FSM 处理）
- ❌ Pattern manual editing UI（spec 014 既有 manually_edited 字段，trader 用 IDE 改 .md）
- ❌ Per-agent 不同 min_cases_per_pattern 阈值（统一一个值，可后续 spec 扩展）
- ❌ LLM-driven pattern naming（用 applied_pattern slug 直接命名，省 LLM 成本）
- ❌ Cross-agent pattern dedup（每 agent 独立维护 patterns）
- ❌ Per-pattern initial PnL 分布检验（min PnL data points 由 spec 018 FSM 处理）

## 落地约束

- 不破坏 spec 014/15/17a/17b/018/019/020a/020b/020c 公开 API
- 不引入新 runtime 依赖
- 直接删旧不留 fallback（用户偏好延续）
- Markdown 简体中文
- 范围预估 ~3-5 天

## 衔接 spec 020b daemon

spec 021 落地后 daemon `actions` 默认 `["pareto", "regime", "skill_proposal", "pattern_extraction"]` 4 个：
- pattern_extraction → 触发新 patterns 创建
- pareto → 对新 patterns（+ 既有）做全局 frontier 重排
- regime → 重新 tag cases
- skill_proposal → 满足 threshold 时 auto-propose 新 skill

数据链补完后整 trilogy 进化系统第一次进入"自循环"状态。
