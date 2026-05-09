# Tasks: Spec 020b — Evolution Daemon

**Branch**: `022-evolution-daemon` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Phase 1: Setup（无新依赖）

本 spec 复用既有 Python / APScheduler / OTel / prometheus-client / fcntl 基建，无新 dependency 安装。

## Phase 2: Foundational

- [ ] T001 [P] 在 `src/cryptotrader/config.py` 加 `EvolutionDaemonConfig` dataclass（enabled / cron / actions / llm_model / propose_threshold 5 字段，按 data-model.md Entity 1）
- [ ] T002 [P] 在 `config/default.toml` 加 `[evolution_daemon]` section（enabled=true / cron="0 0 * * *" / actions=["pareto","regime","skill_proposal"] / llm_model="" / propose_threshold=10）
- [ ] T003 [P] 在 `src/cryptotrader/config.py` 的 `load_config()` 中解析 `[evolution_daemon]` section 到 `EvolutionDaemonConfig`

---

## Phase 3: User Story 1 — Pareto 全局重排（P1）

**Goal**：daemon 跑 Pareto rerank，被支配 rules 转 archived。

**Independent Test**：50 active rules（含 10 低 win_rate）→ daemon 跑完后 ≥ 10 rules 转 archived。

- [ ] T004 [P] [US1] 创建 `src/cryptotrader/ops/__init__.py`（空文件，标记包）
- [ ] T005 [US1] 创建 `src/cryptotrader/ops/daemon.py`：`EvolutionDaemon` 类骨架（`__init__` + `run_once` + `run_forever` + `_acquire_locks` 上下文管理器，按 research.md Decision 1）
- [ ] T006 [US1] 在 `src/cryptotrader/ops/daemon.py` 实现 `_action_pareto()`：调 `cryptotrader.learning.evolution.pareto:rank_rules` + 被支配 rules 转 maturity=archived（按 research.md Decision 2 + spec FR-D6 clarify Q1）
- [ ] T007 [P] [US1] 创建 `tests/test_evolution_daemon.py`：`test_pareto_action_archives_dominated_rules` 用例（构造 50 fixture rules，断言 ≥ 10 archived）
- [ ] T008 [US1] 在 `tests/test_evolution_daemon.py` 加 `test_pareto_action_empty_rules` 用例（0 active rules → PASS 0ms 不抛异常）
- [ ] T009 [US1] 在 `tests/test_evolution_daemon.py` 加 `test_pareto_action_all_frontier` 用例（全部 frontier 成员 → 0 archived，idempotent）

---

## Phase 4: User Story 2 — Regime Cluster 重新计算（P1）

**Goal**：daemon 跑 regime filter，stale regime_tags 重新计算。

**Independent Test**：100 cases stale → ≥ 30% regime_tags 改变。

- [ ] T010 [P] [US2] 在 `src/cryptotrader/learning/memory.py` 加 `refilter_records_by_regime() -> int` thin public wrapper（按 research.md Decision 3 + spec FR-D7）
- [ ] T011 [US2] 在 `src/cryptotrader/ops/daemon.py` 实现 `_action_regime()`：调 `refilter_records_by_regime()`，返回 changed_count 到 ActionResult.details
- [ ] T012 [P] [US2] 在 `tests/test_evolution_daemon.py` 加 `test_regime_action_recalculates_stale_tags` 用例（构造 100 cases stale → 断言 ≥ 30 changed）
- [ ] T013 [US2] 在 `tests/test_evolution_daemon.py` 加 `test_regime_action_idempotent` 用例（已是 current regime → 0 changed）

---

## Phase 5: User Story 3 — Skill Proposal Auto-Trigger（P1）

**Goal**：active rules ≥ 10 时 daemon per-agent 触发 propose_new_skill 写 .draft。

**Independent Test**：12 active rules（agent:tech）→ `.draft` 被创建含 LLM metadata。

- [ ] T014 [US3] 在 `src/cryptotrader/ops/daemon.py` 实现 `_action_skill_proposal()`：4 agents 独立循环检查 `len(active_rules_per_agent) >= propose_threshold`，满足时调 `propose_new_skill(scope=...)` （按 spec FR-D8 clarify Q2）
- [ ] T015 [P] [US3] 在 `tests/test_evolution_daemon.py` 加 `test_skill_proposal_threshold_met` 用例（12 rules agent:tech → .draft 创建 + frontmatter 含 metadata）
- [ ] T016 [P] [US3] 在 `tests/test_evolution_daemon.py` 加 `test_skill_proposal_threshold_not_met` 用例（8 rules → 不创建 .draft + step PASS）
- [ ] T017 [US3] 在 `tests/test_evolution_daemon.py` 加 `test_skill_proposal_per_agent_independent` 用例（tech=12 / chain=8 / news=15 / macro=5 → 创建 2 .draft 文件）

---

## Phase 6: User Story 4 — Soft Degrade + Monitoring（P2）

**Goal**：LLM 失败 soft degrade + 3 Prometheus Gauge 可视。

**Independent Test**：mock OpenAIAPIError → daemon exit 0 + skill_proposal SKIP + pareto/regime PASS + Gauge 输出。

- [ ] T018 [US4] 在 `src/cryptotrader/ops/daemon.py` 加 soft_degrade 逻辑：`_run_action()` 包装 try/except (OpenAIAPIError, TimeoutError, NetworkError) → ActionResult(status=SKIP)（按 research.md Decision 2 + spec FR-D10）
- [ ] T019 [US4] 在 `src/cryptotrader/ops/daemon.py` 加 OTel span 写入：`evolution.daemon.run` 父 span + `evolution.daemon.<action>` 子 span，attr `step.status` / `step.duration_ms` / `step.<details>`（按 spec FR-D11）
- [ ] T020 [US4] 在 `src/cryptotrader/ops/daemon.py` 实现 `_acquire_locks()` fcntl.flock 5s timeout（字母顺序 cases→patterns，按 spec FR-D12 clarify Q3）；timeout 时 daemon 整次跳过 exit 0 + log warning
- [ ] T021 [P] [US4] 创建 `src/cryptotrader/observability/daemon_metrics.py`：3 个 redis-backed sliding window helper（按 research.md Decision 4 简化版 — redis sorted set）
- [ ] T022 [US4] 在 `src/api/routes/metrics.py` 注册 3 Prometheus Gauge：`evolution_daemon_run_count_24h` / `evolution_daemon_llm_failure_rate_24h` / `skill_proposal_draft_count_7d`；endpoint lazy update from redis
- [ ] T023 [P] [US4] 创建 `tests/test_daemon_metrics.py`：3 aggregator 单测（record / count / failure_rate / total，含 sliding window evict 边界）
- [ ] T024 [P] [US4] 在 `tests/test_evolution_daemon.py` 加 `test_soft_degrade_llm_failure` 用例（mock OpenAIAPIError → exit 0 + 1 SKIP + 2 PASS）
- [ ] T025 [US4] 在 `tests/test_evolution_daemon.py` 加 `test_lock_timeout_skips_run` 用例（已被锁 → daemon exit 0 + log warning + 不跑 actions）
- [ ] T026 [P] [US4] 创建 `tests/test_metrics_endpoint_evolution.py`：mock redis events → /metrics 输出含 3 Gauge

---

## Phase 7: CLI + Docker-compose

- [ ] T027 [US4] 在 `src/cli/main.py` 加 `arena evolution-daemon` 命令（按 research.md Decision 6）：`--once` flag / env check / 调用 EvolutionDaemon
- [ ] T028 [US4] 在 `docker-compose.yml` 加 `evolution-daemon` service（按 research.md Decision 5）：command / env / volumes / depends_on / resources
- [ ] T029 [P] 创建 `tests/test_evolution_daemon_cli.py`：CLI 入口测试（`--once` / `EVOLUTION_DAEMON_ENABLED=false` 立即 exit / typer 输出格式）

---

## Phase 8: Polish & Cross-Cutting

- [ ] T030 [P] 创建 `tests/test_e2e_evolution_daemon.py`：mocked 单 cycle 跑完后断言 OTel trace 含 `evolution.daemon.run` + 3 子 span，redis 含事件，3 Gauge 更新
- [ ] T031 跑 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -3` 验证 ≥ 2391 passed / 0 failed（SC-D4）
- [ ] T032 跑 `uv run ruff check src/ tests/` 修复任何新增 lint warning（如需 per-file-ignores 加到 pyproject.toml）
- [ ] T033 跑 SC-D1：`uv run arena evolution-daemon --once` exit 0
- [ ] T034 跑 SC-D3：`grep -n "class EvolutionDaemon" src/cryptotrader/ops/daemon.py` ≥ 1 hit
- [ ] T035 跑 SC-D5：`curl /metrics | grep -c "evolution_daemon\|skill_proposal_draft"` ≥ 3
- [ ] T036 跑 SC-D6：`docker compose config evolution-daemon` 解析成功
- [ ] T037 跑 `git log --oneline 022-evolution-daemon..main | wc -l` ≤ 4 commit 校验（SC-D10）

---

## Dependencies

```
Phase 2 Foundational (T001-T003: config)
   ↓
US1 (Pareto)            T004-T009 (依赖 T001-T003 config)
   ↓
US2 (Regime)            T010-T013 (依赖 T005 daemon 骨架)
   ↓
US3 (Skill proposal)    T014-T017 (依赖 T005 daemon 骨架)
   ↓
US4 (Soft + monitoring) T018-T026 (依赖 T005 + T021 aggregator)
   ↓
Phase 7 (CLI + docker)  T027-T029 (依赖 T005-T020 daemon 完整)
   ↓
Phase 8 Polish          T030-T037 (依赖全部前序 task)
```

T005 (`EvolutionDaemon` 骨架) 是所有 US 的前置依赖。
T010 (regime wrapper) 必须在 T011 (`_action_regime`) 前跑。
T021 (aggregator) 必须在 T022 (Gauge 注册) 前跑。
T018 (soft degrade) 必须在 T024 (soft degrade test) 前跑。

---

## Parallel Execution

### 内部并行（同 US 不同文件）

US1: T004 / T007 可并行（不同文件）
US3: T015 / T016 可并行（独立 test 用例）
US4: T021 / T023 / T024 / T026 可并行（不同文件）

### 跨 US 并行

US2 + US3 + 部分 US4（aggregator 部分）之间可并行（不同文件）。

---

## Implementation Strategy

### MVP scope

**最小可发布**：T001-T013 (Phase 2 + US1 + US2) = config + Pareto + Regime 算法部分。**不**含 US3 skill proposal（依赖 LLM）+ US4 monitoring（依赖 aggregator）。

但建议同 PR 全部交付（4 个 US 是 daemon 单一概念的不同切面）。

### 4-commit 切分

| commit | 涵盖 task | 说明 |
|---|---|---|
| C1 | T001-T005 + T027 + T028 | 基础设施 + config + daemon 骨架 + CLI + docker（纯新增）|
| C2 | T006-T017 | 3 reflect actions + 单测（含 US1/US2/US3）|
| C3 | T018-T026 + T029 | Soft degrade + lock + monitoring + CLI 测试（US4）|
| C4 | T030-T037 | E2E + 最终 gate |

### 增量交付

落地顺序建议：
1. C1 优先（纯新增无回归风险）
2. C2 次（3 actions 算法主体）
3. C3 再次（observability + soft degrade）
4. C4 最后（gate + 全套回归）

---

## Validation

任务总数：37
- Foundational: 3 task
- US1: 6 task
- US2: 4 task
- US3: 4 task
- US4: 9 task
- CLI + Docker: 3 task
- Polish: 8 task

每个 user story 含 independent test 标准；每个 task 含具体文件路径；checklist format 全部合规。
