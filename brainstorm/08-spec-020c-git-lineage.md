# Spec 020c — Git Lineage（trilogy 终段）

**关联 spec**：[研究 016](../specs/016-research-skill-evolution-prior-art/) / [017a](../specs/017-agent-prompt-externalization/) / [017b](../specs/018-agent-prompt-builder-integration/) / [018](../specs/019-memory-evolution/) / [019](../specs/020-skill-evolution/) / [020a](../specs/021-trilogy-ops/) / [020b](../specs/022-evolution-daemon/)
**Date**: 2026-05-09
**Status**: brainstorm 完成，待 ship

## 目标

trilogy 收尾终段。落地 spec 016 D-ENG-02 git lineage 决策：

- daemon `run_once()` 跑完 + maturity FSM transitions（active↔archived / observed→probationary→active）触发 auto-commit 到独立 `evolution` branch
- commit author = current git user；commit message 含 `Auto-Generated-By: spec-020c` trailer
- commit 失败 soft fail（改动保留 + OTel error）
- 同时收尾 3 项跨 spec P2 高 ROI advisory：daemon asyncio.sleep / SIGTERM handler / SkillsGrid a11y aria-label

**不**含其他 6 项 P2 advisory（独立 fix PR 处理）。

## 5 项关键决策

### Q1：Git Lineage 触发范围

**Decision**：B — daemon 触发 + maturity FSM transitions。

**Rationale**：
- daemon 单 commit 汇总最清晰（与 spec 020b run_once batch 模式 1:1 对应）
- maturity transitions（spec 018 既有 `Transition` dataclass）是可识别的 evolution 事件，比 per-cycle evaluate_node 颗粒度更合理
- 噪声可控：~1 commit/day（daemon）+ 偶发 transitions ≈ 1-3 commit/day
- 不做 per-cycle evaluate_node commit 避免 24 commits/day 噪声

### Q2：Commit 粒度 + branch 模型

**Decision**：B — 独立 `evolution` branch（不 merge 回 main）。

**Rationale**：
- main branch log 不被 auto commits 污染（trader-grade 审计标准做法）
- revert 整段 evolution 容易（直接 reset evolution branch）
- 与 spec 014 既有 git workflow 兼容（main 仅 human commits）
- 用户 manual periodic merge 到 main（如有审计需求时）

### Q3：Commit 作者 + 签名

**Decision**：C — Author=current git user + message trailer `Auto-Generated-By: spec-020c`。

**Rationale**：
- 不引入新 git identity（与 spec 014 commit 风格兼容）
- trailer 机制清晰可 grep（`git log --grep="Auto-Generated-By"`）
- author=Pony.Ma 与既有 main commits 一致；trailer 标识 auto 来源

### Q4：020a/020b P2 advisory 收尾范围

**Decision**：B — 3 项高 ROI 子集：
- spec 020b P2-4：`_try_acquire_locks` `time.sleep` → `asyncio.sleep`（防 event loop 阻塞）
- spec 020b P2-5：`run_forever()` SIGTERM signal handler（防数据丢失）
- spec 020a P2-2：SkillsGrid badge aria-label（前端 a11y 合规）

**Rationale**：
- 这 3 项有生产 ROI（性能 / 数据完整 / 合规）
- 其余 6 项 P2（TimeoutError 双检 / `raise exc` 简化 / 类型注解 / e2e 断言路径 / `llm_call_failed` deprecate / staging step 4 集成测试）属代码风格微调，与 lineage 主题不耦合，独立 PR 处理更清爽

### Q5：失败处理

**Decision**：B — Soft fail + 改动保留。

**Rationale**：
- 与 spec 020a Q5 / spec 020b Q5 soft degrade 决策一致
- 改动是数据进化结果不应丢失（commit 失败不等于改动需 revert）
- commit 失败下次 daemon 触发时再尝试（lineage hook 自动检测 working tree dirty）

## 5 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | FSM transitions 写入路径 | ✓ src/cryptotrader/learning/evolution/fsm.py 4 transitions（probationary/active/archived）通过 `maturity=` 赋值 + 返回 `Transition` |
| 2 | daemon `_try_acquire_locks` time.sleep | ✓ daemon.py:458 确认 sync `time.sleep(0.1)` — FR-L10 锚点正确 |
| 3 | run_forever signal handler | ✓ 当前 daemon.py:146 `asyncio.Event().wait()` 无 SIGTERM — FR-L11 锚点正确 |
| 4 | SkillsGrid badge a11y | ⚠ 0 个 aria-label — FR-L12 加 ≥ 3 hits（regime / triggers / inference_failed badges） |
| 5 | git config user | ✓ Pony.Ma / mtf201013@gmail.com 已配 |

## 6 节速览

### 1. Purpose

Trilogy 收尾终段。落地 D-ENG-02 git lineage：daemon + transitions auto-commit 到 `evolution` branch，commit message 含 trailer。同时收尾 3 项 P2 advisory（asyncio.sleep / SIGTERM / a11y）。

### 2. User Stories

- **US-L1（P1）SRE**：daemon 改动 auto-commit 到 evolution branch（commit message 列 actions_run summary）
- **US-L2（P1）Auditor**：FSM transitions 触发 commit；可追溯 rule 状态变化历史
- **US-L3（P1）Maintainer**：daemon `_try_acquire_locks` 改 asyncio.sleep + run_forever SIGTERM handler
- **US-L4（P2）UI**：SkillsGrid badge aria-label a11y 合规

### 3. Functional Requirements（~14 条）

- **FR-L1~3**：`src/cryptotrader/ops/lineage.py` 新模块（`GitLineageHook` 类 + `commit_changes()` 方法）
- **FR-L4**：daemon `run_once()` 末尾调 `commit_changes(actions_run=results)`
- **FR-L5**：commit message 模板含 `Auto-Generated-By: spec-020c` trailer + actions_run 详情
- **FR-L6**：`src/cryptotrader/learning/evolution/fsm.py` 4 transitions 写入路径加 lineage hook（archived / promoted / probationary）
- **FR-L7**：commit 到 `evolution` branch；branch 不存在时 daemon orphan branch 创建
- **FR-L8**：commit author=current git user（不改写）；trailer 唯一标识来源
- **FR-L9**：commit 失败 → 改动保留 + OTel error span + dashboard panel；不阻塞 daemon
- **FR-L10**：`src/cryptotrader/ops/daemon.py:_try_acquire_locks` `time.sleep(0.1)` → `await asyncio.sleep(0.1)`
- **FR-L11**：`run_forever()` 加 SIGTERM/SIGINT signal handler（cancel APScheduler + close redis + flush OTel）
- **FR-L12**：`web/src/pages/memory/components/SkillsGrid.tsx` 3 类 badge 加 aria-label（regime / triggers / inference_failed）
- **FR-L13~14**：`src/api/routes/metrics.py` 加 2 Prometheus Gauge：`evolution_commit_count_24h` / `evolution_commit_failure_rate_24h`

### 4. Success Criteria（~10 条）

- SC-L1：daemon `--once` 跑完 `git log evolution -1` 有新 commit + message 含 `Auto-Generated-By: spec-020c`
- SC-L2：`git log evolution --grep="Auto-Generated-By"` 输出 ≥ 1 行
- SC-L3：`grep "asyncio.sleep" src/cryptotrader/ops/daemon.py` ≥ 1 hit；`grep "time.sleep" daemon.py` 0 hits
- SC-L4：daemon 进程收 SIGTERM 后 graceful shutdown（pytest 验证 OTel span 写完 + redis closed）
- SC-L5：`grep "aria-label" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 3 hits
- SC-L6：commit 失败时 daemon exit 0 + working tree 改动保留（pytest 验证）
- SC-L7：spec 014/15/17a/17b/18/19/20a/20b 既有测试不回归（baseline 2439 → ≥ 2439 pass / 0 fail）
- SC-L8：`/spex:review-spec` 无 P0/P1
- SC-L9：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md
- SC-L10：单 PR ≤ 4 commit

### 5. Dependencies

- **Upstream**：spec 010 / 015 / 018（fsm transitions）/ 019（SkillsGrid）/ 020a / 020b（daemon）
- **Downstream**：无（trilogy 终段）

### 6. Out of Scope

- ❌ 全 9 项 P2 advisory 收尾（仅做 3 项高 ROI 子集；其余独立 fix PR）
- ❌ Per-cycle evaluate_node auto-commit（Q1 B 决定）
- ❌ Skill `.draft` save 触发 commit（保留 human gate）
- ❌ `evolution` branch 自动 merge 回 main（manual periodic）
- ❌ GPG sign / multi-author signing（author=current user）
- ❌ Lineage 告警 alertmanager（仅 dashboard）
- ❌ Git LFS / submodule（数据文件 < 1MB）

## 落地约束

- 不破坏 spec 014/15/17a/17b/018/019/020a/020b 公开 API
- 不引入新 runtime 依赖（gitpython 不引入；用 subprocess `git commit`）
- 复用 spec 020a/020b observability 模式
- 直接删旧不留 fallback
- Markdown 简体中文
- 范围预估 ~3-5 天

## 衔接（trilogy 终结）

spec 020c 落地后 trilogy 完整收尾：
- D-ENG-01 reflect daemon ✅（020b）
- D-ENG-02 git lineage ✅（020c）
- 3 高 ROI P2 advisory ✅（020c）

剩余 6 项 P2 advisory 后续作 fix-only PR 单独处理（与 trilogy 主题解耦）。
