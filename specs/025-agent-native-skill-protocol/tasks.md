---

description: "Task list for spec 022 — Agent-Native Skill Protocol Layer"
---

# Tasks: Agent-Native Skill Protocol Layer

**Branch**: `025-agent-native-skill-protocol` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Data Model**: [data-model.md](data-model.md) | **Contracts**: [contracts/](contracts/)

## Format: `[ID] [P?] [Story] Description`

- **[P]**：可并行（不同文件 + 无依赖）
- **[Story]**：归属 user story（US1=US-A1 / US2=US-A2 / US3=US-A3 / US4=US-A4）
- 描述含确切文件路径

## Path Conventions

`src/` + `tests/` + `agent_skills/` + `docs/api/` + `scripts/` 在 repo root（单 monorepo 项目）。

---

## Phase 1: Setup（无新依赖）

复用既有 FastAPI / Pydantic v2 / SQLAlchemy 2.x / Prometheus client / OpenTelemetry SDK / pyyaml / pytest / httpx — 全已存在。

---

## Phase 2: Foundational（blocking prerequisites for all stories）

- [X] T001 [P] 创建 `agent_skills/_external/` 子目录（5 个空 SKILL.md placeholder：cryptotrader / verdict-feed / market-intel / evolution-insights / execution-replay）— 后续 task 填内容
- [X] T002 [P] 在 `src/cryptotrader/observability/heartbeat_metrics.py` 新增模块 — 3 个 sliding-window aggregator（`HeartbeatPollAggregator` / `HeartbeatPollLagAggregator` / `ExternalSkillFetchAggregator`），复用 spec 020a `CacheMetricsAggregator` 模式（deque + threading.Lock）
- [X] T003 [P] 在 `src/api/routes/metrics.py` 注册 3 个新 Prometheus Gauge（`events_heartbeat_poll_count_24h` / `events_heartbeat_poll_lag_seconds` / `external_skill_fetch_count_24h`）+ `/metrics` lazy update from aggregator

---

## Phase 3: User Story 3 — Patterns API（P1，闭 spec 021 T021）

**Goal**：`GET /api/memory/patterns` 返回 PatternRecord 列表 + 闭 spec 021 SC-P3 outstanding gap。

**Independent Test**：`curl /api/memory/patterns` 返回 `total >= 3` + items 含 maturity 字段 + 支持 `?agent=tech` 过滤。

- [X] T004 [US3] 在 `src/api/routes/memory.py` 加 `PatternRecordResponse` Pydantic v2 schema（复用 `cryptotrader.learning.memory.PatternRecord`，仅做 ApiResponse wrapping）
- [X] T005 [US3] 在 `src/api/routes/memory.py` 加 `GET /api/memory/patterns?agent=&maturity=&limit=` handler — 扫 `agent_memory/{tech,chain,news,macro}/patterns/*.md` → 调 `_load_pattern_record()` (spec 021 helper) → 应用 query filter → 返回 `{items, total}`
- [X] T006 [P] [US3] 创建 `tests/test_api_memory_patterns.py` — 5 用例：(a) total >= 3（基于 spec 021 已落地 3 patterns）；(b) `?agent=tech` 过滤；(c) `?maturity=observed` 过滤；(d) `?limit=2` 截断；(e) `?agent=news` 空目录返回 `items=[], total=0` 不 500

**Checkpoint**：完成后可单独测试 — 闭 spec 021 T021。

---

## Phase 4: User Story 2 — Heartbeat 事件订阅（P1）

**Goal**：`GET /api/events/heartbeat` 让外部 agent pull 事件流（verdict / trade / rejection / phase1_rejected / evolution / oco_state_change）。

**Independent Test**：(a) trigger trading cycle 产生 ≥ 4 events；(b) poll with `since=<cycle_start>` 返回 ≥ 4 events；(c) 续 cursor 返回 0；(d) `?types=verdict,trade` filter 仅 2 类。

- [X] T007 [US2] 在 `src/cryptotrader/nodes/journal.py` 加 `record_phase1_rejection(trace_id, pair, reason, payload)` write helper — 写入 `journal` 表 with `event_type="phase1_rejected"`
- [X] T008 [US2] 在 `src/cryptotrader/nodes/journal.py` 加 `record_evolution_event(event_subtype, artifact_name, payload)` write helper — 写入 `journal` 表 with `event_type="evolution_event"`
- [X] T009 [US2] 修改 `src/cryptotrader/nodes/verdict.py` — 在 Phase 1 hard-reject 路径（`low_rr` / `stop_too_tight` / `missing_sl_tp` / `direction_inverted`）调用 `journal.record_phase1_rejection()`
- [X] T010 [US2] 修改 `src/cryptotrader/ops/daemon.py` — 在 3 个钩子调用 `journal.record_evolution_event()`：(a) `_action_pattern_extraction` 创建新 pattern 后；(b) `_action_skill_proposal` 写 `.draft` 后；(c) `_action_pareto_rerank` rerank 完成后
- [X] T011 [US2] 创建 SQL view `events_heartbeat` migration script — 投影 `journal` 表的 6 个 event_type（verdict_decision / journal_trade_committed / risk_gate_rejected / phase1_rejected / evolution_event / oco_state_change）排序 `(timestamp DESC, trace_id DESC)`
- [X] T012 [US2] 创建 `src/api/routes/events.py` 新模块 — `HeartbeatEvent` Pydantic v2 schema + `GET /api/events/heartbeat?since=&cursor=&types=&limit=` handler + cursor encode/decode helper（base64-url `timestamp|trace_id` 编码）
- [X] T013 [US2] 在 `src/api/main.py` 注册 events router with `Depends(verify_api_key)`
- [X] T014 [US2] events handler 内集成 `HeartbeatPollAggregator.record(client_identifier)` + OTel span `events.heartbeat.poll` with `client_identifier` attr（API_KEY hash 后 8 位）
- [X] T015 [P] [US2] 创建 `tests/test_api_events_heartbeat.py` — 6 用例：(a) 基础 poll；(b) cursor 续传 → 0 events；(c) `since` + `cursor` 同传时 cursor 优先；(d) `?types=verdict,trade` filter；(e) 未来 `since` 返回空不报错；(f) limit 上限

**Checkpoint**：完成后可独立 demo heartbeat 端点。

---

## Phase 5: User Story 1 — SKILL.md 自我发现集成（P1）

**Goal**：外部 agent 一行 prompt `Read /skill/cryptotrader` 即可秒级集成 — bootstrap + 5 child skill markdown + `/skill/<name>` endpoint。

**Independent Test**：`scripts/demo_external_client.py` 端到端 fetch SKILL.md → child skill → patterns → heartbeat 全链 < 30s。

### Skill 路径迁移（atomic）

- [X] T016 [US1] git mv `agent_skills/tech/SKILL.md` → `agent_skills/_internal/tech/SKILL.md`（同样 chain/news/macro 共 4 个文件）
- [X] T017 [US1] 修改 `src/cryptotrader/learning/evolving_skill_provider.py` — `SKILL_DIR` (或类似) const 改 `agent_skills/_internal/`
- [X] T018 [US1] grep + 修改既有测试 fixture 中所有 `agent_skills/{tech,chain,news,macro}/` 路径引用到 `agent_skills/_internal/...`
- [X] T019 [P] [US1] 创建 `tests/test_skill_provider_internal_path.py` — 验证 EvolvingSkillProvider 从新路径正确 load 4 skills + spec 019 既有行为不回归

### 5 个 External SKILL.md

- [X] T020 [P] [US1] 写 `agent_skills/_external/cryptotrader/SKILL.md` — bootstrap：YAML frontmatter（`name: cryptotrader-ai` + description）+ install-locally curl 段落 + child skill 路由表（5 行）+ auth 说明（API_KEY Bearer） + future per-agent JWT 设计存档段落
- [X] T021 [P] [US1] 写 `agent_skills/_external/verdict-feed/SKILL.md` — use case + endpoint examples（`GET /api/verdicts/recent` + heartbeat subscribe pattern） + curl snippets + 引用 `docs/api/verdict.yaml` + `docs/api/events.yaml`
- [X] T022 [P] [US1] 写 `agent_skills/_external/market-intel/SKILL.md` — use case + endpoint examples（`GET /api/snapshot/{pair}` + `GET /api/agents/{name}/results/recent`） + curl snippets + 引用 `docs/api/market.yaml`
- [X] T023 [P] [US1] 写 `agent_skills/_external/evolution-insights/SKILL.md` — use case + endpoint examples（`GET /api/memory/skills` + `GET /api/memory/patterns` + `GET /api/memory/skill-proposals`） + curl snippets + 引用 `docs/api/memory.yaml`；含 PatternRecord / SkillRecord schema 示例
- [X] T024 [P] [US1] 写 `agent_skills/_external/execution-replay/SKILL.md` — use case + endpoint examples（`GET /api/journal/events` + OCO 状态查询） + curl snippets + 引用 `docs/api/execution.yaml`

### /skill/<name> endpoint

- [X] T025 [US1] 创建 `src/api/routes/skills.py` — `SkillRecord` Pydantic v2 schema + `GET /skill/{name}?format=markdown|json` handler — filesystem read `agent_skills/_external/<name>/SKILL.md` → split frontmatter / body → 按 format 返回 markdown 或 JSON
- [X] T026 [US1] 在 `src/api/main.py` 注册 skills router with `Depends(verify_api_key)`
- [X] T027 [US1] skills handler 内集成 `ExternalSkillFetchAggregator.record(skill_name, client_identifier, response_status)`
- [X] T028 [P] [US1] 创建 `tests/test_skills_endpoint.py` — 5 用例：(a) `/skill/cryptotrader` 返回 markdown；(b) `?format=json` 返回 SkillRecord；(c) 不存在 skill 返回 404；(d) auth 失败返回 401 + WWW-Authenticate header；(e) frontmatter 解析正确（name + description）

**Checkpoint**：完成后可单独跑 SKILL.md 自我发现流程。

---

## Phase 6: User Story 4 — OpenAPI 静态化 + Observability（P2）

**Goal**：5 + 1 yaml 静态文件 + pre-commit hook auto-regen + 3 Prometheus gauge 验证（gauge 已在 T002/T003 落地，本阶段补 observability tests）。

**Independent Test**：5 yaml 存在 + OpenAPI 3.0+ 校验 PASS + pre-commit hook regen + `/metrics` 输出含 3 gauge。

- [ ] T029 [US4] 创建 `scripts/export_openapi.py` — in-process import FastAPI app + 调 `app.openapi()` → 按 router tag 拆 5 个分 yaml + 1 combined → 写入 `docs/api/`
- [ ] T030 [US4] 跑 `python scripts/export_openapi.py` 生成 `docs/api/{verdict,market,events,execution,memory,openapi}.yaml`（6 文件 checked-in）
- [ ] T031 [US4] 修改 `.pre-commit-config.yaml` 或新增 hook config — 检测 `src/api/routes/*.py` 变化时自动跑 `scripts/export_openapi.py` + `git add docs/api/*.yaml`（不强制 block，仅警告）
- [ ] T032 [P] [US4] 创建 `tests/test_export_openapi.py` — 3 用例：(a) 6 yaml 文件生成且 valid OpenAPI 3.0+；(b) routes 变更后 regen 反映新 endpoint；(c) deprecated route 含 `deprecated: true` 字段
- [ ] T033 [P] [US4] 创建 `tests/test_observability_heartbeat.py` — 3 用例：(a) 触发 ≥ 1 次 heartbeat poll 后 `events_heartbeat_poll_count_24h` ≥ 1；(b) 触发 ≥ 1 次 skill fetch 后 `external_skill_fetch_count_24h` ≥ 1；(c) lag gauge 有合理值

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T034 [P] 创建 `scripts/demo_external_client.py`（~80 行 Python） — 模拟外部 agent：(a) GET /skill/cryptotrader；(b) GET /skill/verdict-feed；(c) GET /api/memory/patterns；(d) poll /api/events/heartbeat — 全程 < 30s；含 `API_KEY` 未设友好提示（不 traceback）
- [ ] T035 [P] 创建 `tests/test_e2e_skill_protocol.py` — 端到端：fetch SKILL.md → fetch /openapi.yaml → call /api/memory/patterns → poll /api/events/heartbeat 全链 PASS（验证 SC-Q1）
- [ ] T036 跑 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -3` 验证 ≥ 2476 pass / 0 fail（SC-Q7 — spec 021 baseline 不回归）
- [ ] T037 跑 `uv run ruff check src/api/routes/events.py src/api/routes/skills.py src/api/routes/memory.py src/cryptotrader/nodes/journal.py src/cryptotrader/ops/daemon.py src/cryptotrader/learning/evolving_skill_provider.py src/cryptotrader/observability/heartbeat_metrics.py src/api/routes/metrics.py scripts/export_openapi.py scripts/demo_external_client.py tests/test_api_memory_patterns.py tests/test_api_events_heartbeat.py tests/test_skills_endpoint.py tests/test_skill_provider_internal_path.py tests/test_export_openapi.py tests/test_observability_heartbeat.py tests/test_e2e_skill_protocol.py` 全 clean
- [ ] T038 跑 `time uv run python scripts/demo_external_client.py`（API_KEY 已设 + arena serve up） — 验证 SC-Q1（< 30s exit 0）
- [ ] T039 跑 `curl -H "Authorization: Bearer $API_KEY" http://localhost:8003/api/memory/patterns` 返回 `total >= 3` — **验证 SC-Q3 / 闭 spec 021 T021**
- [ ] T040 跑 `curl /api/events/heartbeat?since=2026-05-18T00:00:00Z` 返回 ≤ 50 events + valid `next_cursor` — 验证 SC-Q4
- [ ] T041 跑 `openapi-spec-validator docs/api/*.yaml` 全 PASS — 验证 SC-Q5
- [ ] T042 跑 `curl /metrics | grep -E "events_heartbeat_poll_count_24h|events_heartbeat_poll_lag_seconds|external_skill_fetch_count_24h"` 3 行 ≥ 0 — 验证 SC-Q8
- [ ] T043 跑 `git log --oneline main..025-agent-native-skill-protocol | wc -l` ≤ 5（4 impl + 1 docs）— 验证 SC-Q10

---

## Dependencies

```
Phase 1 Setup (无新依赖) ─────────────────────────────────────────────────────┐
                                                                              │
Phase 2 Foundational                                                          │
  T001 (_external/ placeholder) ──┐                                           │
  T002 (heartbeat_metrics module) ─┤                                          │
  T003 (3 Prometheus gauge) ───────┴────────────────────────────────┐         │
                                                                    │         │
Phase 3 US-A3 Patterns API（P1，C1 commit 闭 T021）                  │         │
  T004 (PatternRecord schema) ─→ T005 (handler) ─→ T006 (tests) ────┘         │
                                                                              │
Phase 4 US-A2 Heartbeat（P1，C2 commit）                                       │
  T007 / T008 (journal helpers) ──→ T009 (verdict hook) ──→ T010 (daemon hooks)│
  T011 (SQL view) ──→ T012 (events.py) ──→ T013 (router register) ──→ T014    │
  T015 (tests) ── 依赖 T002/T003/T012 ──────────────────────────────────────┘ │
                                                                              │
Phase 5 US-A1 SKILL.md（P1，C3 commit）                                        │
  T016 (skill mv) ──→ T017 (path const) ──→ T018 (fixture path) ──→ T019 (tests)
  T020-T024 [P] (5 _external SKILL.md) — 可并行                               │
  T025 (skills.py) ──→ T026 (router register) ──→ T027 (aggregator) ──→ T028 │
                                                                              │
Phase 6 US-A4 OpenAPI + Observability（P2，C3 commit）                         │
  T029 (export_openapi.py) ──→ T030 (run + checked-in 6 yaml) ──→ T031 (hook)│
  T032 [P] (export tests) / T033 [P] (observability tests) — 可并行          │
                                                                              │
Phase 7 Polish & E2E（C4 commit）                                              │
  T034 [P] (demo client) / T035 [P] (e2e test) — 可并行                       │
  T036-T043 (gate steps) — 串行                                                │
```

## Parallel Execution Examples

**Phase 2 — 3 个 task 全可并行**（不同文件）：
```bash
# Spawn 3 agents simultaneously
agent: "T001 create agent_skills/_external/ placeholders"
agent: "T002 implement heartbeat_metrics.py aggregators"
agent: "T003 register Prometheus gauge in metrics.py"
```

**Phase 5 — 5 个 SKILL.md 全可并行**（T020-T024）：
```bash
agent: "T020 write cryptotrader/SKILL.md bootstrap"
agent: "T021 write verdict-feed/SKILL.md"
agent: "T022 write market-intel/SKILL.md"
agent: "T023 write evolution-insights/SKILL.md"
agent: "T024 write execution-replay/SKILL.md"
```

**Phase 6 — 2 个 tests 可并行**（T032 + T033）。

## Commit 切分（spec.md C1-C4 对应）

| Commit | Tasks | 内容 |
|---|---|---|
| **C1** — skill 重组 + patterns endpoint | T004-T006, T016-T019 | 闭 spec 021 T021 + 路径迁移 |
| **C2** — events heartbeat + journal extends | T001-T003, T007-T015 | events endpoint + journal + 3 gauge + SQL view |
| **C3** — OpenAPI 静态化 + 5 SKILL.md + /skill/<name> | T020-T033 | 协议层 + 静态 yaml + observability tests |
| **C4** — E2E + demo client + Gate | T034-T043 | 端到端 + 验证全 SC |

总 commits ≤ 4（SC-Q10）。

## Implementation Strategy（MVP-first incremental delivery）

**MVP 切片**（仅 US-A3 = C1 commit 即可独立交付）：
- 闭 spec 021 T021（最高 ROI 单点价值）
- 可独立部署 + 验证 SC-Q3 → 用户已可通过 `curl /api/memory/patterns` 看 trilogy 产出

**Layer 2 切片**（追加 US-A2 = C2 commit）：
- 添加 heartbeat 事件流
- 外部 agent 可 pull 实时事件

**Layer 3 切片**（追加 US-A1 = C3 commit）：
- 添加完整 SKILL.md 协议层
- 单条 prompt 集成

**Layer 4 切片**（追加 US-A4 = C3 commit 同 PR）：
- OpenAPI 静态化 + observability

**Final 切片**（C4 commit）：
- E2E 验证 + demo client + ruff/pytest gate
