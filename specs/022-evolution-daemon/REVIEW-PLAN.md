# Plan Review: Spec 020b — Evolution Daemon

**Spec dir**: specs/022-evolution-daemon/
**Date**: 2026-05-09
**Reviewer**: Claude (spex:review-plan)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：plan + tasks 结构合规，37 task 覆盖 16 FR + 10 SC，无 P0/P1 issues。MVP 切分清晰（4 commit），4 user story 间通过 daemon 骨架 T005 串联。可进入 implement。

## 0. Scope Check

4 个 user story 覆盖**同一 daemon 子域**的不同切面（Pareto / Regime / Skill proposal / Soft degrade + monitoring）。共享 EvolutionDaemon 类与 docker-compose service。**不**跨独立 subsystem。

US 间通过 T005 daemon 骨架串联（4 个 US 都依赖 daemon 类）；不构成"应拆 spec"信号。

## 1. Task Quality Enforcement

### 检查项

- ✓ Actionable：37 task 全部含明确动作动词（创建 / 实现 / 加 / 跑 / 验证）
- ✓ Testable：所有 task 含具体 file path + 可执行 acceptance check
- ✓ Atomic：单 task 单产出
- ✓ Ordered：T005 daemon 骨架前置；T010 regime wrapper 在 T011 前；T021 aggregator 在 T022 前
- ✓ 文件路径具体
- ✓ Phase 顺序合理（Setup→Foundational→US1→US2→US3→US4→CLI/Docker→Polish）

### 文件结构映射

plan.md `Source Code` 段列出 8 个文件改动 + 4 个新文件，每个文件单一责任：
- `ops/daemon.py` — EvolutionDaemon 类
- `ops/__init__.py` — 包标记
- `observability/daemon_metrics.py` — redis-backed sliding window helper
- `learning/memory.py` — refilter_records_by_regime thin wrapper
- `config.py` — EvolutionDaemonConfig dataclass
- `metrics.py` — 3 Prometheus Gauge
- `cli/main.py` — arena evolution-daemon 命令
- `docker-compose.yml` — evolution-daemon service

无 vague utils / helpers 文件。

## 2. Coverage Matrix

### Functional Requirements

| FR | Story | Tasks | Status |
|---|---|---|---|
| FR-D1（ops/daemon.py + EvolutionDaemon） | US1+ | T004 / T005 | ✓ |
| FR-D2（CLI evolution-daemon 命令） | CLI | T027 | ✓ |
| FR-D3（CronTrigger 0 0 * * *） | US1+ | T005 | ✓ |
| FR-D4（[evolution_daemon] TOML） | Foundational | T002 | ✓ |
| FR-D5（env override） | CLI | T027 | ✓ |
| FR-D6（Pareto archive 非 frontier） | US1 | T006 | ✓ |
| FR-D7（regime filter wrapper） | US2 | T010 / T011 | ✓ |
| FR-D8（per-agent threshold） | US3 | T014 | ✓ |
| FR-D9（仅写 .draft） | US3 | T014 | ✓ |
| FR-D10（soft degrade try/except） | US4 | T018 | ✓ |
| FR-D11（OTel span hierarchy） | US4 | T019 | ✓ |
| FR-D12（fcntl.flock 字母顺序） | US4 | T020 | ✓ |
| FR-D13（3 sliding window aggregator） | US4 | T021 | ✓ |
| FR-D14（3 Prometheus Gauge） | US4 | T022 | ✓ |
| FR-D15（docker-compose service） | Docker | T028 | ✓ |
| FR-D16（service 隔离） | Docker | T028 | ✓ |

### Success Criteria

| SC | Tasks | Verification | Status |
|---|---|---|---|
| SC-D1（arena --once exit 0） | T033 | shell exec | ✓ |
| SC-D2（OTel 3 子 span） | T030 | pytest e2e | ✓ |
| SC-D3（grep EvolutionDaemon） | T034 | shell grep | ✓ |
| SC-D4（≥ 2391 test pass） | T031 | pytest | ✓ |
| SC-D5（3 Prometheus gauge） | T035 | curl + grep | ✓ |
| SC-D6（docker compose config） | T036 | shell | ✓ |
| SC-D7（soft degrade scenario） | T024 | pytest | ✓ |
| SC-D8（review-spec 无 P0/P1） | （已 PASS） | REVIEW-SPEC.md | ✓ |
| SC-D9（review-plan + REVIEW-PLAN.md） | （本文档） | this | ✓ |
| SC-D10（≤ 4 commit） | T037 | git log | ✓ |

### Edge cases coverage

| Edge case | Task | Status |
|---|---|---|
| 空 active rules → pareto PASS 0ms | T008 | ✓ |
| LLM 超时 → skill_proposal SKIP | T024 | ✓ |
| daemon 单次跑超时 30 min | （APScheduler 既有，无新 task） | ✓ |
| active rules < 10 不触发 propose | T016 | ✓ |
| EVOLUTION_DAEMON_ENABLED=false 立即 exit | T029 | ✓ |
| docker service 单独重启 | T028（service definition）+ docker 既有 | ✓ |
| .lock 文件首次跑创建 | T020（mkdir -p parent） | ✓ |
| frontier 全是非支配 → 0 archived | T009 | ✓ |

全部 16 FR + 10 SC + 8 edge case 100% 覆盖。

## 3. Red Flag Scanning

- ✓ 无 vague task
- ✓ 无 monster task（最大 task 单文件单职责）
- ✓ 无 missing file paths
- ✓ Phase 顺序合理（Polish 必须最后）
- ✓ 无跨 spec 依赖泄漏
- ✓ NFR 已落到 task（性能 ≤ 30s / 内存 redis sorted set / 并发 fcntl.flock）

## 4. NFR Validation

- ✓ 性能：daemon `run_once` ≤ 30s（mocked），约束在 plan.md 显式
- ✓ 内存：redis sorted set 上限自管理（7d sliding window evict）
- ✓ 并发：fcntl.flock 5s timeout（跨进程）+ Lock（aggregator 进程内）
- ✓ 可观测性：OTel 父子 span + 3 Prometheus Gauge + structlog（既有）
- ✓ 可回滚：spec.md Reversibility 段显式覆盖

## 5. Recommendations

### Critical (Must Fix Before Implementation)
无

### Important (Should Fix)
无

### Optional (Nice to Have)
- [ ] T021 redis-backed aggregator 实现细节可在 implement 时进一步简化（如直接用 prometheus_client `Counter` 配合 redis 持久化），plan 阶段保留抽象 OK
- [ ] T028 docker-compose service `restart: unless-stopped` policy 可在 plan 中显式（当前在 research.md Decision 5 已写）

## 6. MVP & Incremental Strategy

- ✓ MVP scope：US1 + US2 算法部分（不含 LLM）= Pareto + Regime 即可发布；US3 + US4 是增量
- ✓ 4 commit 切分（C1 文档+骨架 / C2 算法 / C3 monitoring / C4 gate）— 与 spec 019 / 020a 一致
- ✓ 4 US 通过 daemon 骨架串联，但不阻塞并行（T005 完成后 US1-US4 可并行）

## Conclusion

plan + tasks 结构完整、覆盖率 100%、无 red flag、MVP 切分清晰。可进入 `/speckit-implement`。

**Ready for implementation**: Yes
