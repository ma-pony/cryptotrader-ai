# Feature Specification: Spec 020b — Evolution Daemon（reflect daemon 算法运维）

**Feature Branch**: `022-evolution-daemon`
**Created**: 2026-05-09
**Status**: Draft
**Input**: User description: "spec-020b-evolution-daemon — Trilogy 收尾 ops 子域第 2 段。落地 spec 016 D-ENG-01 reflect daemon 决策：独立 docker-compose service 跑 daily reflect daemon，触发 Pareto 全局重排 + Regime cluster 重新计算 + Skill proposal auto-trigger（不 auto-save），soft degrade，3 dashboard panel 可视化（不告警）。不含 git lineage（→ 020c）。"

## Clarifications

### Session 2026-05-09

- Q: FR-D6 "Pareto frontier 之外" 的 archived 判定？ → A: 被任一 Pareto frontier rule 支配的 rules（即非 frontier 成员）转 maturity=archived；frontier 成员保留 active
- Q: FR-D8 propose_threshold 是 per-agent 独立检查还是全局求和？ → A: per-agent 独立（4 agents 各独立检查 active_rules_per_agent ≥ 10），单次 daemon 可触发 0-4 个 propose_new_skill 调用
- Q: FR-D12 文件 lock 获取顺序？ → A: 字母顺序（先 cases/.lock 再 patterns/.lock）防 deadlock；与 spec 018 既有写路径一致

## Background

trilogy 收尾 ops 子域第 2 段（trilogy 第 7 段）。承接已合并 main 的 spec 014 / 015 / 17a / 17b / 18 / 19 / 20a。

trilogy 落地后剩余 ops gap：
1. **Per-cycle 进化的全局视角缺失**：spec 018 evaluate_node 是 per-cycle 单 case 增量评估；缺少周期性全局批量重排（Pareto frontier 全局过滤、regime cluster 全局聚类）
2. **Skill 集长期停留 5 个**：spec 019 propose_new_skill 仅 CLI manual 触发，人工触发频率低，导致 skill 集无法自然增长到目标的 ≥ 20

本 spec 落地独立 docker-compose service 跑 daily reflect daemon，解决以上 gap。**不**含 git lineage（→ spec 020c）。**不**含 020a P2 advisory 收尾（→ spec 020c）。直接删旧不留 fallback。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Pareto 全局重排（Priority: P1）

作为架构师，我希望 daemon 每天自动跑一次 Pareto 全局重排，清理低质量 rules（低 win_rate × confidence_proxy 的 active rules 自动转 archived），保持 active rule 集质量随时间递增不退化。

**Why this priority**：spec 018 evaluate_node 是 per-cycle 增量评估；缺少全局视角的批量重排，时间长了 active 集会被低质量 rules 污染。

**Independent Test**：构造 fixture 50 active rules（含 10 个低 win_rate）→ 跑 `arena evolution-daemon --once` → 断言 ≥ 10 个 rules 转 archived 状态，剩余 active rules 全部位于 Pareto frontier。

**Acceptance Scenarios**：

1. **Given** 50 个 active rules 含 10 个低 win_rate（< 0.4）+ 40 个高 win_rate（≥ 0.6），**When** daemon `--once` 跑完，**Then** ≥ 10 个低 win_rate rules 转 maturity=archived 状态
2. **Given** active rules 全部在 Pareto frontier（无支配关系），**When** daemon 跑完，**Then** 0 个 rules 转 archived
3. **Given** 0 active rules（空仓），**When** daemon 跑完，**Then** Pareto step PASS 0ms + 不抛异常

---

### User Story 2 - Regime Cluster 重新计算（Priority: P1）

作为架构师，市场 regime 漂移（牛熊切换 / fed rate cycle）后，我希望 daemon 每天重新计算 regime cluster，让 cases 自动重新分类到当前 regime。

**Why this priority**：spec 018 `_filter_records_by_regime` 仅按当前 snapshot regime tag 检索；缺少全局重新聚类，旧 cases 的 regime tag 不会随市场漂移更新。

**Independent Test**：构造 fixture 100 cases（带 stale regime_tags）→ 跑 daemon → 断言 cases regime_tags 被重新计算 + ≥ 30% cases 的 tags 改变。

**Acceptance Scenarios**：

1. **Given** 100 cases 含 stale regime_tags（基于 30 天前 market state 计算），**When** daemon 跑完，**Then** ≥ 30 个 cases 的 regime_tags 改变（反映当前 market state）
2. **Given** cases 全部已经按当前 regime 标记，**When** daemon 跑完，**Then** 0 个 cases tags 改变（idempotent）
3. **Given** 0 cases（首次部署），**When** daemon 跑完，**Then** Regime step PASS 0ms + 不抛异常

---

### User Story 3 - Skill Proposal Auto-Trigger（Priority: P1）

作为 prompt engineer / researcher，我希望 daemon 在 active rules ≥ 10 时自动调 spec 019 propose_new_skill 写 `.draft` 文件，让 skill 集随 trading data 自然增长不需 manual CLI 触发。`.draft` 不 auto-save 保留 human review gate。

**Why this priority**：spec 019 propose_new_skill 当前 CLI manual 触发，人工触发频率低导致 skill 集长期停留 5 个；daemon 是驱动 skill 集 ≥ 20 的核心机制。

**Independent Test**：构造 fixture 12 active rules（agent:tech scope）→ 跑 daemon → 断言 `agent_skills/<proposed_name>/SKILL.md.draft` 被创建，frontmatter 含 LLM 推断 metadata。

**Acceptance Scenarios**：

1. **Given** agent:tech scope 含 12 active rules，**When** daemon 跑完，**Then** `agent_skills/<proposed_name>/SKILL.md.draft` 被创建
2. **Given** agent:tech scope 含 8 active rules（< threshold 10），**When** daemon 跑完，**Then** 不创建 .draft + skill_proposal step 标 PASS（threshold-not-met 不算失败）
3. **Given** propose_new_skill 调用成功，**When** 检查 `.draft` frontmatter，**Then** 含 regime_tags / triggers_keywords / importance / confidence / inference_failed=false 字段

---

### User Story 4 - Soft Degrade + Monitoring（Priority: P2）

作为 SRE，我希望 daemon 在 LLM 不可用时 soft-degrade（跑 pareto + regime 算法部分，跳过 skill proposal），保证 daemon 即使 LLM provider 故障期间仍有部分进化输出。dashboard 含 3 panel 可视化 daemon 健康（run_count / llm_failure_rate / draft_count），不告警避免 alert fatigue。

**Why this priority**：spec 020a Q5 决策一致（不告警，仅 dashboard 可视）；soft degrade 比 hard fail 进化连续性好。

**Independent Test**：mock LLM 抛 OpenAIAPIError → 跑 daemon → 断言 (a) daemon exit 0；(b) skill_proposal step 标 SKIP；(c) pareto + regime steps 仍 PASS；(d) `evolution_daemon_llm_failure_rate_24h` Prometheus gauge ≥ 1.0。

**Acceptance Scenarios**：

1. **Given** LLM provider 抛 OpenAIAPIError，**When** daemon 跑，**Then** daemon exit 0 + skill_proposal step status=SKIP + pareto/regime status=PASS
2. **Given** LLM 超时（30s），**When** daemon 跑，**Then** skill_proposal step SKIP；不阻塞 daemon 整体退出
3. **Given** daemon 跑完后访问 `/metrics`，**When** 检查输出，**Then** 含 3 个新 Prometheus Gauge（`evolution_daemon_run_count_24h` / `llm_failure_rate_24h` / `skill_proposal_draft_count_7d`）

---

### Edge Cases

- daemon 跑时 trading cycle 同时写 patterns 文件 → fcntl.flock + 5s timeout，超时 daemon 跳过本次 trigger 下次 daily 重试
- LLM 超时（30s）→ skill proposal step 跳过，pareto + regime 仍跑
- daemon 单次跑超时（默认 30 min）→ APScheduler 自动取消，daemon 写 OTel error span
- active rules < 10 时 skill proposal trigger 不触发（不强制写空 .draft）
- `EVOLUTION_DAEMON_ENABLED=false` 时 daemon 启动后立即 exit 0
- docker-compose `evolution-daemon` service 单独重启不影响 `scheduler` / `api` service
- daemon 启动时 `agent_memory/cases/.lock` 不存在 → daemon 创建空 lock 文件
- pareto frontier 计算用空 list 调用 → 返回空 list 不抛异常

## Requirements *(mandatory)*

### Functional Requirements

#### Daemon 模块

- **FR-D1**：`src/cryptotrader/ops/__init__.py` + `src/cryptotrader/ops/daemon.py` 新模块 MUST 存在；含 `EvolutionDaemon` 类（`__init__(config: EvolutionDaemonConfig)` + `async def run_once() -> RunResult` + `async def run_forever() -> None`）
- **FR-D2**：`src/cli/main.py` MUST 加 `arena evolution-daemon` 命令（args：`--once`（dry-run 单次）/ `--config <path>`），调 `EvolutionDaemon.run_once()` 或 `.run_forever()`

#### 触发器 + 配置

- **FR-D3**：`run_forever()` MUST 用 APScheduler `AsyncIOScheduler` + `CronTrigger("0 0 * * *", timezone="UTC")` 触发 `run_once()`
- **FR-D4**：`config/default.toml` MUST 加 `[evolution_daemon]` section：`enabled` (bool, default true) / `cron` (str, default `"0 0 * * *"`) / `actions` (list, default `["pareto", "regime", "skill_proposal"]`) / `llm_model` (str, default `""` resolves via fallback) / `propose_threshold` (int, default 10)
- **FR-D5**：env `EVOLUTION_DAEMON_ENABLED=false` MUST 覆盖 toml 配置；entrypoint 检测后立即 exit 0

#### 3 reflect actions

- **FR-D6**：Pareto rerank action MUST 调用 `cryptotrader.learning.evolution.pareto:rank_rules(active_rules)`；返回排名结果后**被任一 Pareto frontier rule 支配的 rules**（即非 frontier 成员）MUST 转 `maturity=archived`；frontier 成员保留 `active` 状态
- **FR-D7**：Regime filter action MUST 调用 `cryptotrader.learning.memory:_filter_records_by_regime`（在本 spec 加 thin public wrapper 公开），重新计算所有 cases 的 regime_tags 写回
- **FR-D8**：Skill proposal action MUST 对 4 个 agent（tech / chain / news / macro）**独立**检查触发条件 `len(active_rules_per_agent) >= propose_threshold` (default 10)；每满足条件的 agent 调 1 次 `cryptotrader.learning.skill_proposal:propose_new_skill(scope=f"agent:{agent_id}")`；单次 daemon 可触发 0-4 个 propose_new_skill 调用
- **FR-D9**：Skill proposal MUST 仅写 `agent_skills/<name>/SKILL.md.draft`（不 auto-save）；复用 spec 019 既有 propose_new_skill 路径

#### Soft Degrade + 文件 lock

- **FR-D10**：LLM call 抛异常时（OpenAIAPIError / TimeoutError / NetworkError），daemon MUST：(a) 跳过当前 LLM-dependent step（标记 SKIP）；(b) 写 OTel error span；(c) 继续跑下一个 step；(d) 不告警
- **FR-D11**：daemon `run_once()` MUST 写 OTel span `evolution.daemon.run`，子 span `evolution.daemon.<action>`（pareto / regime / skill_proposal），含 attr `step.status` ∈ {`PASS` / `SKIP` / `FAIL`}
- **FR-D12**：daemon MUST 用 `fcntl.flock` + 5s timeout 锁 `agent_memory/cases/.lock` + `agent_memory/patterns/.lock`，**按字母顺序获取**（先 cases 再 patterns）防 deadlock；timeout 时 daemon 整次跳过（exit 0 + log warning），释放已获取的 lock

#### Observability + Monitoring

- **FR-D13**：`src/cryptotrader/observability/daemon_metrics.py` 新模块 MUST 存在，含 3 个 sliding window aggregator：`DaemonRunCountAggregator` (24h) / `DaemonLLMFailureAggregator` (24h) / `SkillProposalDraftAggregator` (7d)；复用 spec 020a `CacheMetricsAggregator` 模式（deque + Lock）
- **FR-D14**：`src/api/routes/metrics.py` MUST 注册 3 个新 Prometheus Gauge：`evolution_daemon_run_count_24h` / `evolution_daemon_llm_failure_rate_24h` / `skill_proposal_draft_count_7d`；`/metrics` endpoint 触发前 lazy update from aggregator

#### Docker-compose service

- **FR-D15**：`docker-compose.yml` MUST 加 `evolution-daemon` service，复用 `ctdata` volume + `redis` / `postgres` 依赖；命令 `arena evolution-daemon`；restart=unless-stopped
- **FR-D16**：`evolution-daemon` service MUST 与 `scheduler` service 进程级别隔离（独立容器，独立资源限制）

### Key Entities

- **EvolutionDaemonConfig**：dataclass，含 enabled / cron / actions / llm_model / propose_threshold 字段；从 `[evolution_daemon]` TOML section 解析
- **RunResult**：dataclass，含 `actions_run: list[ActionResult]` / `total_duration_ms: int` / `exit_code: int`；用于 `run_once()` 返回值
- **ActionResult**：dataclass，含 `name: str` / `status: Literal["PASS","SKIP","FAIL"]` / `duration_ms: int` / `details: dict`
- **DaemonMetricsAggregator**（3 个）：sliding window deque + Lock，`record(ts, value)` + `count()` / `failure_rate()` / `total()` 接口

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-D1**：`uv run arena evolution-daemon --once` 单次 dry-run（mocked LLM）成功 exit 0，运行时间 ≤ 30s
- **SC-D2**：daemon 单次跑后 OTel trace 含 `evolution.daemon.run` 父 span + 3 子 span（pareto / regime / skill_proposal），各含 `step.status` attribute
- **SC-D3**：`grep -n "class EvolutionDaemon" src/cryptotrader/ops/daemon.py` ≥ 1 hit
- **SC-D4**：spec 014 / 015 / 17a / 17b / 18 / 19 / 20a 既有测试不回归（baseline 2391 → ≥ 2391 pass / 0 fail）
- **SC-D5**：`curl /metrics | grep -E "evolution_daemon|skill_proposal_draft"` 返回 3 个 Prometheus Gauge
- **SC-D6**：`docker compose config evolution-daemon` 可解析；`docker compose up evolution-daemon` 单独启动成功（仅依赖 redis/postgres）
- **SC-D7**：mocked LLM 抛 OpenAI 异常时跑 daemon → daemon exit 0 + skill_proposal step 标 SKIP + pareto + regime 仍 PASS
- **SC-D8**：`/spex:review-spec` 无 P0 / P1 issues
- **SC-D9**：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-D10**：本 spec 单 PR ≤ 4 commit（C1 文档 + C2 daemon 算法 + C3 docker + monitoring + C4 E2E gate）

## Assumptions

- spec 020a observability aggregator 模式可直接复用（CacheMetricsAggregator 含 deque + Lock + sliding window）
- spec 018 `_filter_records_by_regime` 函数可作为 internal API 调用（私有函数将在本 spec 加 thin public wrapper）
- spec 019 `propose_new_skill` 接口稳定（spec 019 brainstorm Q6 决定保留 .draft 写入路径）
- docker-compose 部署环境已支持多 service（spec 014 已建立）
- pytest-asyncio 在 dev deps（spec 018 已引入）
- daemon 单 instance 部署（无 leader election / multi-replica 需求）
- `agent_memory/cases/.lock` + `agent_memory/patterns/.lock` 文件首次跑时由 daemon 创建（不预存）
- APScheduler `AsyncIOScheduler` + `CronTrigger` 可在独立 service entrypoint 启动（与 spec 014 既有 scheduler service 模式一致）
- spec 014 既有 `arena = cli.main:app` entrypoint 路径稳定（CLI 在 `src/cli/main.py`，非 `src/cryptotrader/cli/main.py`）

## Dependencies

**Upstream**：
- spec 010（OpenTelemetry tracing 基建）
- spec 015（metrics endpoint）
- spec 018（Memory Evolution — pareto / regime / FSM / case schema）
- spec 019（Skill Evolution — propose_new_skill）
- spec 020a（Trilogy Ops — observability aggregator pattern + Prometheus 既有 gauge）

**Downstream**：
- spec 020c（git lineage 自动化 + 020a P2 advisory 收尾）

**External tooling**：无新 runtime 依赖（APScheduler / fcntl / OTel SDK 已存在）

## Out of Scope

- ❌ Git lineage 自动 commit cases / patterns / skills（→ spec 020c）
- ❌ 020a P2 advisory 收尾（staging step 4 集成测试 / SkillsGrid badge a11y）→ spec 020c
- ❌ Skill proposal auto-save（保留 human review gate；用户 manual save .draft → SKILL.md）
- ❌ Daemon LLM 失败 alertmanager 告警（与 spec 020a Q5 一致仅 dashboard）
- ❌ Cross-validation IVE re-classify（brainstorm Q4 D 拒绝，成本翻倍 ROI 低）
- ❌ 新算法（pareto / regime / skill_proposal 复用既有，不重写）
- ❌ Anthropic prompt cache 1h beta cache 升级（独立 spec）
- ❌ Daemon multi-instance / leader election（单 instance 简化）
- ❌ 配置热重载（重启 service 应用配置）
- ❌ Daemon 失败重试 exponential backoff（每天 daily 重试足够）

## Reversibility

本 spec 落地后可通过 git revert 单 PR + docker compose restart 回退（无 schema 变更，无数据迁移）。回退后：
- `src/cryptotrader/ops/` 子包删除（不影响生产）
- `src/cli/main.py` 移除 `arena evolution-daemon` 命令（不影响生产）
- `config/default.toml` `[evolution_daemon]` section 移除（不影响生产）
- `docker-compose.yml` `evolution-daemon` service 移除（手动 docker compose stop evolution-daemon）
- `/metrics` 移除 3 新 gauge（不影响 spec 020a 既有 gauge）
- 历史 archived rules（daemon 跑过的）保留状态不变（spec 018 archived 路径既有，可继续操作）

## Implementation Outline

### 单 PR 切 4 commit（与 spec 019 / 020a 同 pattern）

**C1 — 基础设施 + config**：
- `src/cryptotrader/ops/__init__.py`（新）
- `src/cryptotrader/ops/daemon.py`（新 EvolutionDaemon 骨架 + run_once / run_forever）
- `src/cli/main.py`（修改 — 加 evolution-daemon 命令）
- `config/default.toml`（修改 — 加 `[evolution_daemon]` section）
- `src/cryptotrader/config.py`（修改 — 加 EvolutionDaemonConfig dataclass）

**C2 — 3 reflect actions + soft degrade + lock**：
- `src/cryptotrader/ops/daemon.py`（实现 pareto / regime / skill_proposal 三 step + soft_degrade try/except + fcntl.flock）
- `src/cryptotrader/learning/memory.py`（加 thin wrapper 公开 `_filter_records_by_regime`）
- 单测（pytest-asyncio）

**C3 — Observability + docker-compose**：
- `src/cryptotrader/observability/daemon_metrics.py`（新 3 aggregator）
- `src/api/routes/metrics.py`（修改 — 注册 3 Gauge）
- `docker-compose.yml`（修改 — 加 evolution-daemon service）
- 单测

**C4 — E2E + 最终 Gate**：
- `tests/test_e2e_evolution_daemon.py`（mocked LLM cycle 验证 3 actions + soft degrade）
- `pyproject.toml`（per-file-ignores 如需）
- grep / pytest / ruff 全部 gate
