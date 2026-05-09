# Spec 020b — Evolution Daemon（reflect daemon 算法运维）

**关联 spec**：[研究 016](../specs/016-research-skill-evolution-prior-art/) / [017a](../specs/017-agent-prompt-externalization/) / [017b](../specs/018-agent-prompt-builder-integration/) / [018](../specs/019-memory-evolution/) / [019](../specs/020-skill-evolution/) / [020a](../specs/021-trilogy-ops/)
**Date**: 2026-05-09
**Status**: brainstorm 完成，待 ship

## 目标

trilogy 收尾 ops 子域第 2 段（trilogy 第 7 段）。落地 spec 016 D-ENG-01 reflect daemon 决策：

- 独立 docker-compose service 跑 daily 进化算法
- 触发 Pareto 全局重排 + Regime cluster 重新计算 + Skill proposal auto-trigger
- Soft degrade（LLM 不可用时跑算法部分跳过 LLM 部分）
- 3 个 dashboard panel 可视化（不告警）

**不**含 git lineage（→ spec 020c）。**不**含 020a P2 advisory 收尾（→ spec 020c）。

## 7 项关键决策

### Q1：Spec 拆分策略

**Decision**：B — 拆 020b（daemon）+ 020c（lineage + 020a P2）。

**Rationale**：
- daemon 是高 ROI 算法工作；lineage 是审计工具，ROI 取决于 daemon 改动量
- 先 020b daemon 稳定 1 周观察数据增长，再 020c lineage 决定（数据驱动）
- 拆分降低单 spec PR 复杂度；与 trilogy 单议题风格一致
- daemon 涉及独立 service + LLM 算法；lineage 涉及 git auto-commit 数据风险，二者风险维度不同

### Q2：Daemon 进程模型

**Decision**：B — 独立 docker-compose service。

**Rationale**：
- trading cycle 是生产关键路径（实盘风险），daemon 含 LLM 不稳定调用 + 批量 IO，故障隔离值得
- spec 014 / 015 已建立 docker-compose 模式（7 个 service），加 service 是惯例
- 独立资源限制 + 独立重启 + dev 关闭 cycle 仍可观察 daemon
- 跨进程 lock 用 fcntl.flock + 5s timeout（spec 内 FR-D12 落地）

### Q3：触发频率

**Decision**：C — 每天 cron `0 0 * * *`（UTC midnight）。

**Rationale**：
- 批量算法（pareto / regime filter）天然适合 daily cadence；24h × 1 cycle/h × ~3 case/cycle ≈ 70 case 适合 pareto
- 与 Anthropic 1h beta cache 反向匹配（spec 020a Q2 升级路径预留）
- LLM 成本 ~$0.005-0.02/day（单次批量 call）
- 进化 lag 1 day 可接受（与 spec 014 reflection 周期对齐）

### Q4：Daemon 触发的进化动作

**Decision**：C — Pareto + Regime + Skill proposal auto-trigger（不 auto-save）。

**Rationale**：
- A + B（Pareto / Regime）算法运维必做，与 evaluate_node per-cycle 路径不重复（per-cycle 是单 case 增量；daemon 是全局批量重排）
- C 是 daemon 核心 ROI（人工 manual 触发频率低，daemon 才能驱动 skill 自然增长，spec 019 brainstorm Q4 提到的"长期 skill 集 ≥ 20"目标）
- `.draft` 不 auto-save 保留 human gate（spec 019 propose_new_skill 已有路径），降低风险
- 不做 D（IVE re-classify cross-validation），成本翻倍 ROI 低

### Q5：Daemon failure mode

**Decision**：B — Soft degrade（无 LLM 时跑算法部分）+ 仅 dashboard 可视。

**Rationale**：
- 与 spec 020a Q5 决策一致（不告警，避免 alert fatigue）
- soft degrade 比 skip 进化连续性好（pareto / regime filter 不依赖 LLM 仍可跑）
- LLM 失败时 skill_proposal step 标 SKIP，不影响 daemon exit code
- 跨进程文件 lock 冲突 → 短 timeout 跳过本次 trigger，下次 daily 重试

### Q6：Monitoring 扩展

**Decision**：B — 3 个 dashboard panel（run_count / llm_failure_rate / draft_count）。

**Rationale**：
- 锚定 Q5/Q4 关键决策的可观测性
- 不引入告警（与 spec 020a Q5 一致）
- 复用 spec 020a observability aggregator 模式（sliding window deque + Prometheus Gauge）

### Q7：Daemon Config 入口

**Decision**：A — TOML config + env override。

**Rationale**：
- 与 spec 014 `[scheduler]` / spec 015 / spec 020a 风格一致
- 本地 dev 改 toml 比改 env 灵活
- docker-compose 仍可通过 env override 关键配置（如 `EVOLUTION_DAEMON_ENABLED=false`）

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | `pareto.py:rank_rules` 可独立调用 | ✓ src/cryptotrader/learning/evolution/pareto.py:54 接口存在，daemon 直接 reuse |
| 2 | `propose_new_skill` 入口 | ✓ src/cryptotrader/learning/skill_proposal.py:202 |
| 3 | scheduler.py APScheduler 模式 | ✓ AsyncIOScheduler + CronTrigger 现成可借鉴 |
| 4 | docker-compose.yml 多 service 模式 | ✓ 7 services（含 scheduler），加 evolution-daemon 惯例 |
| 5 | CLI 入口路径 | ⚠️ 修订：实际 src/cli/main.py（不是 src/cryptotrader/cli/）；entrypoint `arena = cli.main:app`。FR-D2 改写在 src/cli/main.py |
| 6 | ops/ 子包是否存在 | ⚠️ 修订：不存在，FR-D1 需要新建 src/cryptotrader/ops/__init__.py + daemon.py |
| 7 | regime filter 函数位置 | ⚠️ 修订：在 src/cryptotrader/learning/memory.py:358（非 evolution 子包），daemon 调用路径需适配 |

## 6 节速览

### 1. Purpose

Trilogy 收尾 ops 子域第 2 段。独立 docker-compose service 跑 daily reflect daemon，触发 Pareto + Regime + Skill proposal auto-trigger。**不**含 git lineage（→ 020c）。

### 2. User Stories

- **US-D1（P1）Architect**：daemon daily 跑 Pareto 全局重排清理低质量 rules
- **US-D2（P1）Architect**：daemon daily 重新计算 regime cluster 适应漂移
- **US-D3（P1）Researcher**：daemon active rules ≥ 10 时 auto-propose 新 skill 到 .draft（不 auto-save）
- **US-D4（P2）SRE**：daemon LLM 不可用时 soft-degrade 跑算法部分；3 dashboard panel 可视

### 3. Functional Requirements（~16 条）

- **FR-D1**：`src/cryptotrader/ops/__init__.py` + `daemon.py` 新模块（`EvolutionDaemon` 类）
- **FR-D2**：`src/cli/main.py` 加 `arena evolution-daemon` 命令（含 `--once` flag for dry-run）
- **FR-D3**：`EvolutionDaemon.run()` async 方法跑 3 reflect actions
- **FR-D4**：APScheduler `CronTrigger("0 0 * * *", timezone="UTC")` 触发
- **FR-D5**：`config/default.toml` 加 `[evolution_daemon]` section（enabled / cron / actions / llm_model / propose_threshold）
- **FR-D6**：Pareto rerank 调用 `cryptotrader.learning.evolution.pareto:rank_rules`
- **FR-D7**：Regime filter 调用 `cryptotrader.learning.memory:_filter_records_by_regime`
- **FR-D8**：Skill proposal auto-trigger 阈值 `active_rules >= 10` 触发，调 `cryptotrader.learning.skill_proposal:propose_new_skill`
- **FR-D9**：Skill proposal 仅写 `.draft`（不 auto-save，复用 spec 019 既有路径）
- **FR-D10**：LLM 不可用时 daemon `soft_degrade=True`：pareto + regime 仍跑；skill_proposal step 标 SKIP
- **FR-D11**：daemon 写 OTel error span 当 LLM call 失败；不告警
- **FR-D12**：跨进程文件 lock（`fcntl.flock` + 5s timeout）防与 trading cycle 写冲突
- **FR-D13**：`src/cryptotrader/observability/daemon_metrics.py` 新增 3 sliding window aggregator
- **FR-D14**：`src/api/routes/metrics.py` 注册 3 Prometheus Gauge：`evolution_daemon_run_count_24h` / `evolution_daemon_llm_failure_rate_24h` / `skill_proposal_draft_count_7d`
- **FR-D15**：`docker-compose.yml` 加 `evolution-daemon` service（独立容器，复用 ctdata volume）
- **FR-D16**：env `EVOLUTION_DAEMON_ENABLED=false` 全局关闭（覆盖 toml）

### 4. Success Criteria（~10 条）

- SC-D1：`arena evolution-daemon --once` 单次 dry-run（mocked LLM）成功 exit 0
- SC-D2：daemon 单次跑后 OTel trace 含 3 reflect actions span（pareto / regime / skill_proposal）
- SC-D3：grep `EvolutionDaemon` src/cryptotrader/ops/ ≥ 1 hit
- SC-D4：spec 014/15/17a/17b/18/19/20a 既有测试不回归（baseline 2391 → ≥ 2391）
- SC-D5：3 Prometheus Gauge 在 `/metrics` 输出
- SC-D6：docker-compose `evolution-daemon` service 可独立 `docker compose up evolution-daemon`
- SC-D7：LLM 失败时 daemon 跑完算法部分 exit 0 + skill_proposal step 标 SKIP
- SC-D8：`/spex:review-spec` 无 P0/P1
- SC-D9：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md
- SC-D10：单 PR ≤ 4 commit

### 5. Dependencies

- **Upstream**：spec 010（OTel）/ 015（metrics endpoint）/ 018（pareto / regime / FSM）/ 019（propose_new_skill）/ 020a（observability aggregator + Prometheus 既有 gauge）
- **Downstream**：spec 020c（git lineage 自动化 + 020a P2 收尾）

### 6. Out of Scope

- ❌ Git lineage 自动 commit（→ 020c）
- ❌ 020a P2 advisory 收尾（→ 020c）
- ❌ Skill proposal auto-save（保留 human gate）
- ❌ Daemon LLM 失败告警（仅 dashboard）
- ❌ Cross-validation IVE re-classify（Q4 D 拒绝）
- ❌ 新算法（pareto / regime / skill_proposal 复用既有）
- ❌ 1h beta cache 升级（独立 spec）
- ❌ Daemon multi-instance / leader election（单 instance 简化）

## 落地约束

- 不破坏 spec 014/15/17a/17b/018/019/020a 公开 API
- 不引入新 runtime 依赖（APScheduler / fcntl / OTel SDK 已存在）
- 复用 spec 020a observability aggregator 模式
- 直接删旧不留 fallback
- Markdown 简体中文
- 范围预估 1 周

## 衔接 020c

020b 落地稳定 1 周后启动 020c：

```python
# 020c 加 git lineage hook
from cryptotrader.ops.lineage import GitLineageHook

# daemon 跑完后触发
hook = GitLineageHook(branch="evolution", batch_mode=True)
hook.commit_changes(actions_run=["pareto", "regime", "skill_proposal"])

# 020c 同时收尾 020a P2 advisory
# - staging step 4 集成测试缺口
# - SkillsGrid badge a11y aria-label
```

均不破坏本 spec 接口契约。
