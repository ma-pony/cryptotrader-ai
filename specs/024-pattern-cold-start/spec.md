# Feature Specification: Spec 021 — Pattern Cold-Start（trilogy 进化数据链补完）

**Feature Branch**: `024-pattern-cold-start`
**Created**: 2026-05-11
**Status**: Draft
**Input**: User description: "spec-021-pattern-cold-start — trilogy 落地实测发现 cases→distill→patterns→evaluate→transitions 数据链断在 distill 这环。distill_patterns 只更新已有 patterns 不创建新的；run_reflection 节点定义但从未挂到 graph/daemon/CLI。生产实测 cases/ 246 文件，patterns/ 0 文件。dashboard /memory 全空。修复方向：A 内置 cold-start in distill / C+D daemon daily + CLI manual / 5 配置化阈值 / A 完整 spec ship。"

## Background

trilogy（spec 014/15/17a/17b/18/19/20a/20b/20c）已合并 main。落地后 2026-05-11 dashboard `/memory` 页面实测全空：

- `agent_memory/cases/` — 246 文件（每 cycle 决策快照，spec 014 正常持续累积）✅
- `agent_memory/<agent>/patterns/` — **0 文件**（应从 cases 提炼）❌
- `agent_memory/transitions/` — 目录不存在 ❌

根因 2 处：

1. **`distill_patterns()` 只更新已有 patterns 的 maturity，没有创建新 patterns 的代码**（src/cryptotrader/learning/memory.py:447）：函数遍历 `patterns_dir.glob("*.md")`，若目录空则空循环 → 0 输出
2. **`run_reflection()` 节点定义于 nodes/reflection.py:18 但从未被任何 graph / scheduler / CLI 注册触发**

→ spec 018 Memory Evolution + spec 019 Skill Evolution + spec 020b Evolution Daemon 全部建立在 "patterns 非空"假设上；cold-start gap 导致整段数据链断裂、dashboard 全空、daemon 跑过 1 次后 `evaluate_node` 拿空 patterns 跑 FSM 自然 0 transitions。

本 spec 补完 cold-start 路径：`distill_patterns()` 加创建新 patterns 逻辑 + `run_reflection` 接入 evolution-daemon 作第 4 个 action + CLI `arena experience distill` 紧急触发入口。直接删旧不留 fallback。Markdown 简体中文。

## Clarifications

### Session 2026-05-11

- Q: FR-P2 pattern `name` 字段从 applied_pattern 文本生成 slug 的规则？ → A: lowercase + 替换非 alnum 为 `-` + 截断 ≤ 60 字符 + 去除前后 `-`；collision 时加 `-N` 数字后缀（N=2,3,...）
- Q: FR-P2 `pnl_track` 字段 — cases 中 `final_pnl=None` 的如何处理？ → A: 跳过（不计入 pnl_track.pnls）；若所有 cases 都 None 则 `pnl_track = PnLTrack(pnls=[])` 空但仍创建 pattern
- Q: FR-P2 `regime_tags` 字段 — cases 跨多个 regime_tags，如何投票 top 3？ → A: 跨该 pattern 引用的所有 cases，统计 tag 出现频次，频次降序取前 3；频次并列时按字母序兜底

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Pattern Cold-Start 数据链补完（Priority: P1）

作为架构师 / trader，trilogy 进化系统的 `cases → distill → patterns → evaluate → transitions → dashboard` 数据链当前在 distill 环节断裂，我希望 distill_patterns 能从已有 cases 提炼出首批 patterns（不仅是更新已有），让整个 trilogy 数据链跑通。

**Why this priority**：trilogy 进化系统的核心价值（自主学习 + 状态演化 + 决策审计）依赖 patterns 非空；当前生产实测 patterns 永远空 → dashboard `/memory` 全空 → 用户看不到进化效果 → spec 018/019/020 全部失效。

**Independent Test**：跑 `uv run arena experience distill --memory-dir agent_memory --cycles-window 200`，验证：
- `agent_memory/{tech,chain,news,macro}/patterns/` 各目录创建 ≥ 1 个 `*.md` PatternRecord 文件
- `curl /api/memory/rules` 返回 `total > 0` 且 items 含 maturity="observed" pattern

**Acceptance Scenarios**：

1. **Given** agent_memory/cases/ 含 ≥ 5 个 case 引用同一 (agent, applied_pattern) tuple，**When** distill_patterns 跑完，**Then** 该 (agent, pattern) 在 `agent_memory/<agent>/patterns/<pattern_slug>.md` 创建 PatternRecord，maturity=observed
2. **Given** (agent, applied_pattern) 频次 < `min_cases_per_pattern`（默认 5），**When** distill_patterns 跑，**Then** 不创建该 pattern（噪声防护）
3. **Given** 已有 patterns_dir 含旧 pattern 文件，**When** distill_patterns 跑，**Then** 旧 pattern 的 maturity 仍按既有 FSM 路径更新（向后兼容，不破坏 spec 014 行为）

---

### User Story 2 - Daemon Daily Pattern Extraction（Priority: P1）

作为 SRE，evolution-daemon 当前 3 个 actions（pareto / regime / skill_proposal）依赖 patterns 非空，但 cold-start 后没人持续蒸馏新 patterns。我希望 daemon 加第 4 个 action `pattern_extraction`，daily 跑一次自动蒸馏。

**Why this priority**：与 spec 020b daemon 设计一致（"独立 docker 容器跑批量进化"）；4 个 action 并列，daemon 跑完后 trilogy 全数据链自循环。

**Independent Test**：跑 `uv run arena evolution-daemon --once`，验证：
- 4 个 actions 全部 PASS（pareto / regime / skill_proposal / **pattern_extraction**）
- OTel trace 含 `evolution.daemon.pattern_extraction` 子 span + `step.status=PASS` attr
- Daemon log 输出 `[pattern_extraction] PASS — new=N updated=M archived=K`

**Acceptance Scenarios**：

1. **Given** daemon `config.actions = ["pareto", "regime", "skill_proposal", "pattern_extraction"]`，**When** daemon `--once` 跑，**Then** 4 actions 全部 fire + log 输出
2. **Given** distill_patterns 失败抛异常，**When** pattern_extraction action 触发，**Then** action 标 SKIP（soft degrade）+ daemon exit 0 + 其他 actions 仍跑（与 spec 020b FR-D10 一致）
3. **Given** patterns 已有 N 个，**When** daemon 跑完，**Then** 新创建 patterns 写入 `agent_memory/<agent>/patterns/`，dashboard 实时反映

---

### User Story 3 - CLI Manual Trigger（Priority: P2）

作为 dev，初次部署 / debug / 紧急 backfill 需要绕过 daemon 立即触发 distill。希望提供 `arena experience distill` CLI 命令。

**Why this priority**：daemon 是 daily 节奏，不适合 dev 调试快速迭代；CLI 紧急触发 + 输出 ReflectionRun summary 便于排查。

**Independent Test**：跑 `uv run arena experience distill --memory-dir agent_memory --cycles-window 200`，验证：
- exit 0
- stdout 含 `cases_processed: N / patterns_created: M / patterns_updated: P / patterns_archived: K`
- 跑完后 `agent_memory/<agent>/patterns/` 出现新文件

**Acceptance Scenarios**：

1. **Given** dev 机干净 `agent_memory/` 含 cases，**When** 跑 `arena experience distill --once`，**Then** patterns 创建 + ReflectionRun summary 打印
2. **Given** `--memory-dir custom/path`，**When** 跑命令，**Then** 仅扫该路径 cases；不影响默认路径
3. **Given** `--cycles-window 50`，**When** 跑，**Then** 仅最近 50 cases 用于蒸馏（spec 014 既有参数语义）

---

### Edge Cases

- cases 目录空（0 文件）→ distill_patterns 返回 ReflectionRun(cases_processed=0)，不抛异常
- 单 case PnL=None → 仅计入频次，不算 PnL track 平均
- 同名 pattern 已存在（手动 seed 或之前自动创建）→ 跳过创建，走既有 maturity FSM 更新
- daemon `pattern_extraction` action SKIP 时 → action_result.details 含 SKIP 原因
- CLI 命令在生产容器内跑 → 不与 daemon 冲突（两者都走 fcntl.flock 文件锁，spec 020c FR-L12 既有）
- 跨 agent applied_pattern 名相同 → 各 agent 独立创建（spec 014 设计：每 agent 自己 patterns/ 目录）
- LLM 不可用 → cold-start 不依赖 LLM（仅扫 cases body 文本 parse + frequency 统计），跑不受影响

## Requirements *(mandatory)*

### Functional Requirements

#### Distill 函数补完

- **FR-P1**：`src/cryptotrader/learning/memory.py:distill_patterns()` MUST 加 cold-start 逻辑：扫 cases → 按 `(agent, applied_pattern)` tuple 统计频次 + PnL → 满足 `count >= min_cases_per_pattern` 创建 `PatternRecord(maturity="observed")` 写到 `agent_memory/<agent>/patterns/<pattern_slug>.md`
- **FR-P2**：新 pattern PatternRecord MUST 含字段：
  - `name`：applied_pattern 文本 → slug = `lowercase + replace non-alnum with "-" + truncate ≤ 60 chars + strip leading/trailing "-"`；collision 时 append `-2`、`-3` 等数字后缀
  - `agent` ∈ {tech, chain, news, macro}
  - `description` = `"Auto-distilled pattern: {applied_pattern} (from {N} cases)"`
  - `body` = 含 source_cycles 前 5 个 cycle_id list
  - `pnl_track` = `PnLTrack(pnls=[case.final_pnl for case in cases if case.final_pnl is not None])`；若全部 None 则 `pnl_track = PnLTrack(pnls=[])` 空但仍创建 pattern
  - `regime_tags` = 跨该 pattern 所有引用 cases 的 regime tags 频次降序前 3；频次并列时字母序兜底
  - `maturity = "observed"` (spec 014 FSM 初始态)
- **FR-P3**：cold-start 路径 MUST 不破坏已有 patterns 更新（已有 patterns_dir 中文件继续走 maturity FSM 路径）
- **FR-P4**：`distill_patterns()` 创建新 pattern 失败 MUST log warning + 继续处理其他 patterns；不影响整体函数 exit
- **FR-P5**：cold-start 路径 MUST 写 OTel span `learning.distill.cold_start`，attrs：`patterns_created` / `patterns_updated` / `cases_processed`

#### Config

- **FR-P6**：`src/cryptotrader/config.py:ExperienceConfig` MUST 加字段 `min_cases_per_pattern: int = 5`
- **FR-P7**：`config/default.toml` `[experience]` 段 MUST 加 `min_cases_per_pattern = 5` 默认值

#### Daemon 集成（第 4 个 action）

- **FR-P8**：`src/cryptotrader/ops/daemon.py` MUST 加 `_action_pattern_extraction()` 方法：调 `distill_patterns(cycles_window=cfg.experience.lookback_commits)` + 写 `ActionResult.details={"new_count": ..., "updated_count": ..., "archived_count": ...}`
- **FR-P9**：`config/default.toml` `[evolution_daemon].actions` 默认列表 MUST 加 `"pattern_extraction"`（4 个 actions）
- **FR-P10**：daemon action dispatch（daemon.py:221 处的 `if action_name == "pareto" ... elif`）MUST 加 `elif action_name == "pattern_extraction"` 分支
- **FR-P11**：pattern_extraction action soft degrade：异常时 ActionResult(status=SKIP)；与 spec 020b FR-D10 一致

#### CLI

- **FR-P12**：`src/cli/main.py` MUST 加 `arena experience distill` typer command：
  - `--memory-dir <path>`（默认 `agent_memory/`）
  - `--cycles-window <N>`（默认 `config.experience.lookback_commits = 30`）
  - 调 `distill_patterns()` + 输出 ReflectionRun summary
- **FR-P13**：CLI command MUST 在异常时打印 error + exit 1（不静默）

### Key Entities

- **PatternRecord**（spec 014 既有 dataclass）：本 spec 仅扩展创建路径，不改 schema
- **ReflectionRun**（spec 014 既有）：本 spec 复用 `patterns_created` / `patterns_updated` / `patterns_archived` / `cases_processed` / `error` 字段
- **ApplyPatternSlug**：从 applied_pattern 文本生成 filesystem-safe slug（如 "Volume Spike + RSI Overbought" → "volume-spike-rsi-overbought"）；本 spec 内部 helper

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-P1**：`uv run arena experience distill --memory-dir agent_memory --cycles-window 200` 一次成功 + exit 0 + 创建 ≥ 1 patterns
- **SC-P2**：跑后 `find agent_memory/{tech,chain,news,macro}/patterns -name "*.md" 2>/dev/null | wc -l` ≥ 3
- **SC-P3**：`curl /api/memory/rules` 返回 `total > 0` + items 含 maturity="observed" PatternRecord
- **SC-P4**：`uv run arena evolution-daemon --once` 跑后 4 actions 全 PASS：pareto / regime / skill_proposal / **pattern_extraction**
- **SC-P5**：`tests/test_distill_patterns_cold_start.py` 5 用例 PASS（empty / single agent / freq below / freq above / pnl_track）
- **SC-P6**：`tests/test_e2e_pattern_cold_start.py` 端到端 PASS（fixture 200+ cases → daemon pattern_extraction → ≥ 3 patterns）
- **SC-P7**：spec 014 / 15 / 17a / 17b / 18 / 19 / 20a / 20b / 20c 既有测试不回归（baseline 2458 → ≥ 2458 pass / 0 fail）
- **SC-P8**：`/spex:review-spec` 无 P0 / P1
- **SC-P9**：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-P10**：本 spec 单 PR ≤ 4 commit（C1 distill + config / C2 daemon + CLI / C3 tests + e2e / C4 final gate）

## Assumptions

- spec 014 `_read_cases()` / `_parse_applied_from_body()` / `_load_pattern()` / `_save_pattern()` helper 接口稳定可复用
- spec 014 `PatternRecord` schema 完整（不需要扩展字段）
- spec 014 既有 distill_patterns 的 maturity FSM 更新逻辑保留（向后兼容）
- spec 020b daemon `actions` 配置 list-driven，加 action 无 breaking change
- pattern slug 生成可直接用 applied_pattern 文本 lowercase + 替换非 alnum 为 `-`，不需要 LLM
- 每 agent 独立维护 patterns/（spec 014 既有目录结构）
- min_cases_per_pattern = 5 是合理初始值（小于 cycles_window=30）
- cold-start 不需要 LLM 调用（纯文本 parse + 频次统计 + 文件写入）；与 spec 014 FR-008 既有 4-layer 防过拟合中"min sample count"约束一致

## Dependencies

**Upstream**：
- spec 014（PatternRecord schema + distill_patterns 框架 + _parse_applied_from_body）
- spec 018（FSM maturity transitions — 新创建的 observed patterns 由 spec 018 后续 promote）
- spec 020b（Evolution Daemon — pattern_extraction 加第 4 个 action）

**Downstream**：无（trilogy cold-start gap 收尾）

**External tooling**：无新 runtime 依赖

## Out of Scope

- ❌ Pattern auto-deletion / pruning（spec 018 既有 maturity FSM "deprecated/archived" 路径处理）
- ❌ Pattern manually-edited 字段更新 UI（trader 用 IDE 改 .md 文件）
- ❌ Per-agent 不同 min_cases_per_pattern 阈值（统一一个值，可后续 spec 扩展）
- ❌ LLM-driven pattern naming（直接 applied_pattern slug 命名，省 LLM 成本）
- ❌ Cross-agent pattern dedup（每 agent 独立 patterns/ 目录）
- ❌ Per-pattern PnL 分布检验（min PnL data points 由 spec 018 FSM 自动处理）
- ❌ Pattern_extraction action 内嵌 IVE LLM 调用（保持轻量 + 纯算法）
- ❌ 历史 cases 重导入 / 数据迁移脚本（本 spec 用既有 cases，不改 cases schema）

## Reversibility

本 spec 落地后可通过 git revert 单 PR 回退（无 schema 变更，无数据迁移）。回退后：
- `distill_patterns()` cold-start 部分代码删除 → 恢复 spec 014 既有仅更新行为
- `[experience].min_cases_per_pattern` 字段删除（不影响 ExperienceConfig 既有字段）
- `_action_pattern_extraction()` 方法删除（不影响 spec 020b 既有 3 actions）
- `[evolution_daemon].actions` 默认值删除 `"pattern_extraction"`（unknown action name 在 daemon 内被 ignore）
- `arena experience distill` CLI 命令删除（不影响其他 CLI 命令）
- 历史已创建 patterns 文件保留（不删除，可手动清理 `rm agent_memory/<agent>/patterns/*.md`）

## Implementation Outline

### 单 PR 切 4 commit（与 spec 019 / 020a / 020b / 020c 同 pattern）

**C1 — Distill cold-start + config**：
- `src/cryptotrader/learning/memory.py:distill_patterns()` 加 cold-start 逻辑 + helper `_create_pattern_from_applied()` 私有函数
- `src/cryptotrader/config.py:ExperienceConfig` 加 `min_cases_per_pattern: int = 5`
- `config/default.toml` `[experience]` 加 `min_cases_per_pattern = 5`

**C2 — Daemon 集成 + CLI**：
- `src/cryptotrader/ops/daemon.py` 加 `_action_pattern_extraction()` + dispatch 分支
- `config/default.toml` `[evolution_daemon].actions` 加 `"pattern_extraction"`
- `src/cli/main.py` 加 `arena experience distill` command

**C3 — 单测 + e2e**：
- `tests/test_distill_patterns_cold_start.py` 5 用例
- `tests/test_e2e_pattern_cold_start.py` 端到端

**C4 — Final Gate**：
- 跑 SC-P1 ~ SC-P7 全套验证
- pyproject.toml RUF / per-file-ignores（如需）
- 验证 dashboard `/memory` API 返回非空
