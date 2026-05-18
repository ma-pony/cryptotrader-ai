# Plan Review: Spec 022 — Agent-Native Skill Protocol Layer

**Spec dir**: specs/025-agent-native-skill-protocol/
**Date**: 2026-05-18
**Reviewer**: Claude (spex:review-plan)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：plan + tasks 结构合规，43 task 覆盖 25 FR + 10 SC，无 P0/P1 issues。MVP 切分清晰（US-A3 单独 = 闭 spec 021 T021 = MVP 最小可交付）。4 US 中 3 个 P1 独立可发布。Ready for implement。

## 0. Scope Check

4 US 围绕同一「外部 agent 协议层」主题的 4 个切面：
- US-A1（SKILL.md 协议）— 整体协议入口
- US-A2（Heartbeat events）— 事件流
- US-A3（Patterns API）— 内容暴露（闭 T021）
- US-A4（OpenAPI 静态化 + Observability）— 工具链 + 可观测

共享底层：Pydantic v2 schema / FastAPI router pattern / spec 020a aggregator 模式 / journal 表 SQL view。不跨独立 subsystem。

## 1. Task Quality

- ✓ Actionable：43 task 全部明确动作动词（创建 / 修改 / 跑 / git mv 等）
- ✓ Testable：所有 task 含具体 file path
- ✓ Atomic：单 task 单产出，最大粒度是 5 个 SKILL.md（仍分别原子可 [P] 并行）
- ✓ Ordered：依赖图清晰（T007/T008 journal helper → T009 verdict hook → T010 daemon hook → T011 SQL view → T012 events endpoint）
- ✓ Phase 顺序合理（Foundational → MVP US3 → US2 → US1 → US4 → Polish）

### 文件结构映射（plan.md Source Code 段）

完整覆盖，每文件单一责任：
- `src/api/routes/events.py` — heartbeat endpoint（新）
- `src/api/routes/memory.py` — + patterns endpoint（修改）
- `src/api/routes/skills.py` — /skill/<name>（新）
- `src/api/routes/metrics.py` — + 3 gauge（修改）
- `src/cryptotrader/learning/evolving_skill_provider.py` — 路径常量改
- `src/cryptotrader/nodes/journal.py` — + 2 record helpers
- `src/cryptotrader/observability/heartbeat_metrics.py` — 3 aggregator（新）
- `src/cryptotrader/ops/daemon.py` — + 3 evolution event 钩子
- `scripts/{export_openapi,demo_external_client}.py`（新 ×2）
- `agent_skills/_internal/` 重组 + `_external/` 5 个新 SKILL.md
- `docs/api/*.yaml` ×6 生成

无 vague utils / monster task。

## 2. Coverage Matrix

### Functional Requirements

| FR | Story | Tasks | Status |
|---|---|---|---|
| **FR-022-1** (skill 路径重组 `_internal/`) | US1 | T016 | ✓ |
| **FR-022-2** (`_external/` 5 个 SKILL.md) | US1 | T001（placeholder）+ T020-T024（内容）| ✓ |
| **FR-022-3** (YAML frontmatter) | US1 | T020-T024 + T028 验证 | ✓ |
| **FR-022-4** (bootstrap SKILL.md 含 install/auth/路由) | US1 | T020 | ✓ |
| **FR-022-5** (child SKILL.md self-contained) | US1 | T021-T024 | ✓ |
| **FR-022-6** (EvolvingSkillProvider 路径改) | US1 | T017 + T018 + T019 验证 | ✓ |
| **FR-022-7** (export_openapi.py) | US4 | T029 | ✓ |
| **FR-022-8** (5 yaml + combined) | US4 | T030 | ✓ |
| **FR-022-9** (pre-commit hook) | US4 | T031 | ✓ |
| **FR-022-10** (events heartbeat endpoint) | US2 | T012 | ✓ |
| **FR-022-11** (复用 journal + SQL view) | US2 | T011 + T012 | ✓ |
| **FR-022-12** (response schema) | US2 | T012 | ✓ |
| **FR-022-13** (cursor pagination) | US2 | T012（cursor helper inline）| ✓ |
| **FR-022-14** (OTel span client_identifier) | US2 | T014 | ✓ |
| **FR-022-15** (journal record helpers ×2) | US2 | T007 + T008 | ✓ |
| **FR-022-16** (daemon evolution event 钩子 ×3) | US2 | T010 | ✓ |
| **FR-022-17** (`/api/memory/patterns` handler) | US3 | T005 | ✓ |
| **FR-022-18** (复用 _load_pattern_record) | US3 | T005 | ✓ |
| **FR-022-19** (response schema `{items, total}`) | US3 | T004 + T005 | ✓ |
| **FR-022-20** (query params `?agent=&maturity=&limit=`) | US3 | T005 + T006 验证 | ✓ |
| **FR-022-21** (heartbeat_metrics 3 aggregator) | Foundational | T002 | ✓ |
| **FR-022-22** (3 Prometheus gauge 注册 + lazy update) | Foundational | T003 | ✓ |
| **FR-022-23** (demo_external_client.py) | Polish | T034 | ✓ |
| **FR-022-24** (demo client 友好错误处理) | Polish | T034 | ✓ |
| **FR-022-X1** (`/skill/<name>` endpoint) | US1 | T025 + T026 + T027 + T028 | ✓ |

25/25 FR 100% 覆盖。

### Success Criteria

| SC | Tasks | Verification | Status |
|---|---|---|---|
| **SC-Q1** (demo client < 30s 端到端) | T038 | `time python demo_external_client.py` | ✓ |
| **SC-Q2** (5 _external SKILL.md valid frontmatter) | T020-T024 + T028 | markdown lint + yaml.safe_load | ✓ |
| **SC-Q3** (`/api/memory/patterns total >= 3` **闭 T021**) | T039 | curl + jq | ✓ |
| **SC-Q4** (heartbeat returns ≤ 50 + valid cursor) | T040 | curl + jq | ✓ |
| **SC-Q5** (6 OpenAPI 3.0+ yaml valid) | T041 | openapi-spec-validator | ✓ |
| **SC-Q6** (pre-commit auto-regen + stage) | T031 | hook trigger test | ✓ |
| **SC-Q7** (baseline ≥ 2476 pass / 0 fail) | T036 | pytest --no-cov | ✓ |
| **SC-Q8** (3 gauge 可见 + ≥ 0) | T042 | curl /metrics grep | ✓ |
| **SC-Q9** (review-spec + review-plan + REVIEW-PLAN.md) | （已 PASS）| REVIEW-SPEC.md + 本文档 | ✓ |
| **SC-Q10** (单 PR ≤ 4 commits) | T043 | git log count | ✓ |

10/10 SC 100% 覆盖。

### Edge Cases coverage

| Edge case | Task | Status |
|---|---|---|
| API_KEY 错误 → 401 + WWW-Authenticate | T028 (e) | ✓ |
| heartbeat since 未来时间 → 空 list 不报错 | T015 (e) | ✓ |
| heartbeat 同传 since + cursor → cursor 优先 | T015 (c) | ✓ |
| patterns endpoint 空目录 → items=[] | T006 (e) | ✓ |
| 路径迁移后 EvolvingSkillProvider 仍正常 load | T019 | ✓ |
| arena serve 重启后 _external SKILL.md 可 fetch | 集成于 T038 e2e | ✓ |
| OpenAPI deprecated route → `deprecated: true` | T032 (c) | ✓ |
| demo client API_KEY 未设 → 友好错误 | T034（FR-022-24 内含） | ✓ |

8/8 edge case 100% 覆盖。

## 3. Red Flag Scanning

- ✓ 无 vague task
- ✓ 无 monster task（最大 T020-T024 是 5 个独立 [P] SKILL.md，每个 ~30 行 markdown 可单独 review）
- ✓ 无 missing file paths
- ✓ Phase 顺序合理（Foundational 先于 US，US3 MVP 先于其他 US）
- ✓ 无跨 spec 依赖泄漏（spec 014-021 全部 upstream 显式列出）
- ✓ 无 P0 risk 在 critical path（path migration atomic + restart timing 已识别为 plan-level risk 而非 spec gap）

## 4. NFR Validation

| NFR | Target | Verification path |
|---|---|---|
| 性能 `/skill/<name>` | < 200ms p95 | filesystem read，单文件 < 50KB → < 200ms 易达成；可在 T038 加 timing assertion |
| 性能 `/api/memory/patterns` | < 500ms p95 | filesystem scan ~10-100 文件 + frontmatter parse → < 500ms；spec 021 已验证 |
| 性能 `/api/events/heartbeat` | < 1s p95 | SQL view query indexed `(timestamp, event_type)` 1k 行 < 5ms + cursor encode 微秒级 → < 1s |
| 内存 | < +50MB | 3 aggregator deque（24h 滑窗）+ filesystem cache 无 → 增量可忽略 |
| 并发 | external agent 5-10 并发 | FastAPI async + aggregator threading.Lock → OK |
| 可观测性 | 3 新 gauge + OTel span | T002/T003/T014 → ✓ |
| 可回滚 | git revert + restart | spec.md Reversibility 段 9 项详细 → ✓ |
| 安全 | 全 read-only + API_KEY auth | 无 write endpoint + Depends(verify_api_key) 注册（T013/T026） → ✓ |

## 5. Risks（plan 阶段已识别）

| Risk | Mitigation | Task |
|---|---|---|
| 🟡 path migration blast radius（spec 019 ~50 fixture 引用） | grep + sed 一次性改 + T019 验证 + 单独 commit | T018 + T019 |
| 🟡 arena serve restart 时 cycle 偏移 | ship 时选 cycle 间隔内（17min - 57min after :00）restart | （ship 时 manual timing） |
| 🟡 ETH "ghost hold" 已知问题不在本 spec 范围 | enrich_verdict_context 加 OKX position sync 是 spec 023+ 工作 | OOS by design |
| 🟢 scheduler silent miss（P0 incident 5/18 18:57-19:57） | 加 watchdog 是 spec 023+ 工作；OCO server-side 保护已生效（ETH SL 自动平仓证明）| OOS by design |

## 6. Recommendations

### Critical / Important
无

### Optional
- [ ] **T031 pre-commit hook**：plan 阶段可决定使用 `pre-commit` framework 还是 `lefthook`（当前未明示）— 默认推荐 pre-commit（生态成熟）
- [ ] **T025 skills.py 加 OTel span**：除 T027 aggregator 外可加 `skill.fetch.served` OTel span 含 `skill_name` attr — 边际改动，可进 implementation 阶段判断
- [ ] **T011 SQL view migration**：plan 阶段可决定是 Alembic migration 还是 raw SQL（spec 015 模式优先）— 不影响 spec 范围
- [ ] **demo client 加 retry**：T034 demo client 当前是单次 fetch，可加 1 次 retry 应对 transient network — 不影响 SC-Q1 但更 robust

## 7. MVP

- ✓ **MVP scope（最小可交付）**：T004-T006（Phase 3 US-A3 = patterns endpoint）
  - 闭 spec 021 T021 outstanding gap
  - 单独 deploy 即可让外部 agent 看到 trilogy 产出
  - 0 路径迁移风险
- ✓ **Layer 2**：+ T001-T003 + T007-T015（Phase 4 US-A2 heartbeat）
- ✓ **Layer 3**：+ T016-T028（Phase 5 US-A1 SKILL.md，含路径迁移）
- ✓ **Final**：+ T029-T043（Phase 6 OpenAPI + Phase 7 polish）
- ✓ 4 commit 切分清晰（C1-C4 详 tasks.md Commit 切分表）

## 8. 并行机会评估

- **Phase 2**：3 个 task 全并行（T001/T002/T003，不同文件）
- **Phase 3 US-A3**：T006 tests 可与 T004/T005 并行（但 T004 → T005 串行）
- **Phase 4 US-A2**：T007/T008 可并行（journal.py 同文件但不同函数）
- **Phase 5 US-A1**：T020-T024 = 5 个 SKILL.md 全并行（不同文件）+ T019 + T028 tests 并行
- **Phase 6 US-A4**：T032 + T033 tests 并行
- **Phase 7 polish**：T034 + T035 并行

总并行机会：~12 个 task 可在不同时间窗口并发执行（约 28% 任务可加速）。

## Conclusion

plan + tasks 完整覆盖、无 red flag、MVP 切分清晰、NFR target 现实可达。25 FR / 10 SC / 8 edge case 100% 覆盖。可进入 `/speckit-implement`。

**Ready for implementation**: Yes

**No P0 / P1 issues**：✓
