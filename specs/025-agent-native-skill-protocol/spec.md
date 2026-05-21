# Feature Specification: Agent-Native Skill Protocol Layer

**Feature Branch**: `025-agent-native-skill-protocol`
**Created**: 2026-05-18
**Status**: Draft
**Input**: 参考完整 brainstorm `brainstorm/10-spec-022-agent-native-skill-protocol.md`

## User Scenarios & Testing *(mandatory)*

### User Story 1 — SKILL.md 自我发现集成（Priority: P1）

作为 cryptotrader-ai 的所有者，我希望我自己用 Codex/Cursor/Claude Code 帮我调试或分析交易时，只需一行 prompt `Read http://localhost:8003/skill/cryptotrader and register`，agent 就能自动 fetch bootstrap SKILL.md → 学到 child skill 路由 → 按需 fetch 子 skill → 直接消费 verdict / patterns / events — 不再每次手动告知接口。

**Why this priority**：日常工作流痛点最高 — 当前每次 Codex 接入都要手动 brief「去 `/api/snapshot` 拉数据」、「verdict 在 `routes/verdict.py`」等碎片信息；本 spec 一次性解决。

**Independent Test**：跑 `scripts/demo_external_client.py` 模拟外部 agent — (a) GET `/skill/cryptotrader` 拿到 bootstrap markdown；(b) GET `/skill/verdict-feed` 拿到 child skill；(c) GET `/api/verdicts/recent` 实际数据返回；(d) 全程 < 30s 完成。断言 ≥ 5 个 SKILL.md fetch 成功 + ≥ 1 个真实 endpoint 返回数据。

**Acceptance Scenarios**：

1. **Given** arena serve 已启动 + API_KEY 已设置，**When** `curl -H 'Authorization: Bearer $API_KEY' http://localhost:8003/skill/cryptotrader`，**Then** 返回有效 markdown（YAML frontmatter + body）且 ≤ 2s。
2. **Given** agent 已 fetch bootstrap SKILL.md，**When** agent 按 SKILL.md 中路由表 fetch 任一 child skill（如 `/skill/verdict-feed`），**Then** 拿到 self-contained markdown 含 endpoint examples + curl snippets + OpenAPI yaml 引用。
3. **Given** agent 已 fetch child skill，**When** agent 按 skill 文档调用真实 endpoint（如 `/api/verdicts/recent`），**Then** 返回合法 JSON + Pydantic schema 一致。

---

### User Story 2 — Heartbeat 事件订阅（Priority: P1）

作为外部 agent（Codex/Cursor），我希望通过 `GET /api/events/heartbeat?since=<timestamp>&types=[verdict,trade,rejection,evolution]` 持续 poll cryptotrader-ai 的事件流，及时知道 verdict 决策、真单 fill、risk 拒绝、进化事件 — 不需 webhook endpoint。

**Why this priority**：external agent 多数无公网 endpoint，pull 模式是 agent 时代事件订阅的事实模式（参考 AI-Trader heartbeat skill 设计）。

**Independent Test**：(a) trigger trading cycle 跑完产生 ≥ 4 events；(b) poll heartbeat with `since=<cycle_start>` 返回 ≥ 4 events；(c) 第 2 次 poll with `since=<last_event_ts>` 返回 0 events（cursor 正确）；(d) `?types=verdict,trade` filter 仅返回这 2 类。

**Acceptance Scenarios**：

1. **Given** trading cycle 刚跑完产生 4+ events（含 1 verdict + 1 trade + 1 rejection + 1 evolution），**When** `curl /api/events/heartbeat?since=<cycle_start_ts>`，**Then** 返回 `items` 含 ≥ 4 个 HeartbeatEvent + `next_cursor` non-null。
2. **Given** 上次 poll 返回了 `next_cursor=X`，**When** `curl /api/events/heartbeat?cursor=X`，**Then** 返回 `items=[]` 且 `next_cursor=null`（无新事件时）。
3. **Given** 同时传入 `since` + `cursor`，**When** 调用 heartbeat，**Then** `cursor` 优先，`since` 忽略。
4. **Given** `?types=verdict,trade` filter，**When** 调用 heartbeat，**Then** 返回 items 仅含 `event_type ∈ {verdict, trade}`。

---

### User Story 3 — Patterns API 暴露（Priority: P1，闭 spec 021 T021）

作为外部 agent，我希望 `GET /api/memory/patterns` 返回 trilogy 进化系统产出的 PatternRecord 列表（含 maturity / pnl_track / source_cycles / regime_tags / created_at / agent / applied_text），让我能学习 cryptotrader-ai 历史决策模式。

**Why this priority**：闭 spec 021 SC-P3 outstanding gap；spec 018/019/020a/b/c/021 全部产出现在外部 0 visibility。

**Independent Test**：(a) 至少 3 patterns 存在于 disk（spec 021 已落地）；(b) `curl /api/memory/patterns` 返回 `total >= 3`；(c) items 含 `maturity="observed"` 字段；(d) 支持 `?agent=tech` query filter 返回该 agent 的 patterns subset。

**Acceptance Scenarios**：

1. **Given** `agent_memory/tech/patterns/sma-breakdown-short.md` 等 ≥ 3 patterns 文件存在（spec 021 已落地），**When** `curl /api/memory/patterns`，**Then** 返回 `{items: [...], total: 3}` 且每项含 maturity/pnl_track/source_cycles 等字段。
2. **Given** `?agent=tech` query param，**When** 调用 patterns，**Then** items 仅含 `agent="tech"` 的 patterns。
3. **Given** `?maturity=observed&limit=2`，**When** 调用，**Then** items ≤ 2 且全部 `maturity="observed"`。
4. **Given** `agent_memory/news/patterns/` 为空（spec 021 状态），**When** `curl /api/memory/patterns?agent=news`，**Then** 返回 `{items: [], total: 0}` 不报 500。

---

### User Story 4 — OpenAPI 静态化 + Observability（Priority: P2）

作为 SRE，我希望 OpenAPI schema 是 checked-in 静态文件而不是 runtime 生成，外部 agent / 文档工具能离线读 schema；同时希望 heartbeat / external skill fetch 有 Prometheus gauge 可观测。

**Why this priority**：spec 020a observability backbone 已建立，本 spec 加 3 个 gauge 是边际成本；OpenAPI 静态化让 agent 不依赖 server 就能 codegen / 类型检查。

**Independent Test**：(a) `docs/api/{verdict,market,events,execution,memory}.yaml` 5 文件存在 + valid OpenAPI 3.0+；(b) pre-commit hook 检测 `routes_*.py` 变化自动 regen；(c) `curl /metrics` 输出含 `events_heartbeat_poll_count_24h` / `events_heartbeat_poll_lag_seconds` / `external_skill_fetch_count_24h` 3 gauge。

**Acceptance Scenarios**：

1. **Given** `scripts/export_openapi.py` 跑完，**When** `ls docs/api/*.yaml`，**Then** 5 个分文件 + 1 个 combined 共 6 文件存在且 OpenAPI 3.0+ 校验 PASS。
2. **Given** 修改 `src/api/routes/events.py` 后 git commit，**When** pre-commit hook 跑，**Then** `docs/api/events.yaml` 自动 regen + git stage（不强制 block，仅警告）。
3. **Given** API 跑过 ≥ 1 次 heartbeat poll + ≥ 1 次 external skill fetch，**When** `curl /metrics`，**Then** 输出含 3 新 gauge 且值 ≥ 1。

---

### Edge Cases

- 外部 agent fetch SKILL.md 时 API_KEY 错误 → 返回 401 + `WWW-Authenticate` header 提示
- heartbeat poll `since` parameter 是未来时间 → 返回空 list，不报错
- heartbeat poll 同时 `since` + `cursor` → `cursor` 优先，`since` 忽略
- patterns endpoint 时 `agent_memory/<agent>/patterns/` 不存在 → 返回 `items=[]`，不 500
- spec 019 EvolvingSkillProvider 在路径迁移后（atomic mv 到 `_internal/`）仍正常 load skills
- arena serve 重启后所有 `_external/SKILL.md` 在 ≤ 5s 可 fetch
- OpenAPI export 时 FastAPI app 含 deprecated route → 仍 export 但加 `deprecated: true` 字段
- demo client 跑 `python scripts/demo_external_client.py` 在 `API_KEY` 未设环境变量时 → 友好错误提示而非 traceback

---

## Requirements *(mandatory)*

### Functional Requirements

#### 外部 SKILL.md 协议层

- **FR-022-1**：系统 MUST 将 `agent_skills/{tech,chain,news,macro}/SKILL.md` 重组到 `agent_skills/_internal/{tech,chain,news,macro}/SKILL.md`（git mv atomic 单次原子提交）
- **FR-022-2**：系统 MUST 在 `agent_skills/_external/` 子目录提供 5 个 SKILL.md：`cryptotrader/SKILL.md`（bootstrap）/ `verdict-feed/SKILL.md` / `market-intel/SKILL.md` / `evolution-insights/SKILL.md` / `execution-replay/SKILL.md`
- **FR-022-3**：所有 SKILL.md MUST 用 YAML frontmatter（`---\nname: ...\ndescription: ...\n---`），复用 spec 019 既有规范
- **FR-022-4**：bootstrap SKILL.md MUST 含 (a) install-locally curl 段落；(b) child skill 路由表；(c) auth 说明（单 API_KEY Bearer 模式）
- **FR-022-5**：每个 child SKILL.md MUST self-contained：含 use case / endpoint examples / curl snippets / response schema reference 到 OpenAPI yaml
- **FR-022-6**：`EvolvingSkillProvider`（spec 019）MUST 更新路径常量到 `agent_skills/_internal/`（迁移 atomic + 测试 fixture 同步改）
- **FR-022-X1**：系统 MUST 暴露 `GET /skill/<skill_name>` endpoint 返回对应 `_external/<skill_name>/SKILL.md` 内容（Content-Type: `text/markdown`）

#### OpenAPI 静态化

- **FR-022-7**：`scripts/export_openapi.py` MUST 提供 — 在 in-process FastAPI app 调 `app.openapi()` → 按 router tag 拆 → 写 5 个分文件 + 1 combined
- **FR-022-8**：系统 MUST 生成 5 个分文件 `docs/api/{verdict,market,events,execution,memory}.yaml` + 1 个 `docs/api/openapi.yaml` combined
- **FR-022-9**：pre-commit hook MUST 检测 `src/api/routes/*.py` 变化时自动 regen openapi yaml + git stage（不强制 CI block，仅警告）

#### Heartbeat /events 端点

- **FR-022-10**：`src/api/routes/events.py` MUST 提供 `GET /api/events/heartbeat?since=<ISO8601>&limit=50&types=[verdict,trade,rejection,evolution]&cursor=<token>`
- **FR-022-11**：系统 MUST 复用现有 `journal` 表 + 新建 SQL view `events_heartbeat` 投影 4 类 must + 2 类 opt event
- **FR-022-12**：response schema MUST 是 `{items: HeartbeatEvent[], next_cursor: str | null}`；`HeartbeatEvent = {timestamp, trace_id, event_type, pair, payload}`
- **FR-022-13**：cursor-based pagination — `next_cursor` MUST 编码 `(timestamp, trace_id)` 字典序，后续 poll 用 `?cursor=<token>` 续传
- **FR-022-14**：OTel span `events.heartbeat.poll` MUST 含 `client_identifier` attr（API_KEY hash 后 8 位）
- **FR-022-15**：`src/cryptotrader/nodes/journal.py` MUST 加 `record_phase1_rejection()` + `record_evolution_event()` 两个新 write helper
- **FR-022-16**：spec 020b daemon MUST 调用 `record_evolution_event()` 在 new pattern created / new skill drafted / Pareto rerank done 三个钩子

#### Patterns API（闭 spec 021 T021）

- **FR-022-17**：`src/api/routes/memory.py` MUST 加 `GET /api/memory/patterns` handler
- **FR-022-18**：handler MUST 复用 `cryptotrader.learning.memory` 模块的 `_load_pattern_record()` helper（spec 021 已实装）
- **FR-022-19**：response schema MUST 是 `{items: PatternRecord[], total: int}` 与 `/api/memory/skills` 对齐
- **FR-022-20**：handler MUST 支持 query params `?agent=<tech|chain|news|macro>&maturity=<observed|stable|active|archived>&limit=<int>`

#### Observability + Monitoring

- **FR-022-21**：`src/cryptotrader/observability/heartbeat_metrics.py` MUST 提供 3 个 aggregator：`HeartbeatPollAggregator` (24h) / `HeartbeatPollLagAggregator` (24h) / `ExternalSkillFetchAggregator` (24h)，复用 spec 020a `CacheMetricsAggregator` 模式
- **FR-022-22**：`src/api/routes/metrics.py` MUST 注册 3 新 Prometheus gauge：`events_heartbeat_poll_count_24h` / `events_heartbeat_poll_lag_seconds` / `external_skill_fetch_count_24h`；`/metrics` 触发前 lazy update from aggregator

#### Demo / 验证 client

- **FR-022-23**：`scripts/demo_external_client.py` MUST 提供 — 模拟外部 agent 完整流程：fetch SKILL.md → fetch /openapi.yaml → call /api/memory/patterns → poll /api/events/heartbeat；~80 行 Python
- **FR-022-24**：demo client MUST 友好处理 `API_KEY` 未设环境变量时打印 setup 指引（不是 traceback）

### Key Entities

- **HeartbeatEvent**：单条事件记录。属性：`timestamp` (ISO8601), `trace_id` (UUID), `event_type` (verdict / trade / rejection / evolution / oco_state / phase1_rejected), `pair` (e.g. SOL/USDT:USDT, optional), `payload` (event-specific dict)。
- **PatternRecord**：trilogy 进化系统产出的模式记录（spec 018 定义，spec 021 创建）。属性：`agent` (tech/chain/news/macro), `slug`, `applied_text`, `maturity` (observed/stable/active/archived), `pnl_track` (list of float), `source_cycles` (list of cycle_id), `regime_tags` (list), `created_at`, `last_updated_at`。
- **SkillRecord (external)**：暴露给外部 agent 的 skill 描述。属性：`name`, `description`, `frontmatter` (YAML dict), `body` (markdown), `last_modified_at`。
- **ExternalSkillFetchEvent**：每次 `/skill/<name>` 被外部 agent 调用的访问记录。属性：`timestamp`, `skill_name`, `client_identifier` (API_KEY hash 后 8 位), `response_status`。

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-Q1**：`scripts/demo_external_client.py` 端到端跑通 < 30s，fetch SKILL.md → fetch patterns → poll heartbeat 全 PASS（验证 US-A1）
- **SC-Q2**：5 个 `_external/<skill>/SKILL.md` 文件存在 + 通过 markdown lint + YAML frontmatter valid（验证 FR-022-2/3）
- **SC-Q3**：`curl http://localhost:8003/api/memory/patterns` 返回 `total >= 3`（**闭 spec 021 T021 SC-P3**，验证 US-A3）
- **SC-Q4**：`curl /api/events/heartbeat?since=2026-05-18T00:00:00Z` 返回最近事件 ≤ 50 + valid `next_cursor`（验证 US-A2）
- **SC-Q5**：`docs/api/{verdict,market,events,execution,memory}.yaml` 5 文件 valid OpenAPI 3.0+ + combined `docs/api/openapi.yaml`（验证 FR-022-8）
- **SC-Q6**：pre-commit hook 检测 `src/api/routes/*.py` 变化自动 regen openapi yaml + git stage（验证 FR-022-9）
- **SC-Q7**：spec 014/015/017a/17b/018/019/020a/b/c/021 既有测试不回归 — baseline 2476 → ≥ 2476 pass / 0 fail
- **SC-Q8**：3 个新 Prometheus gauge 在 `/metrics` 可见 + 值 ≥ 0（验证 FR-022-22）
- **SC-Q9**：`/spex:review-spec` + `/spex:review-plan` 无 P0 / P1 issues + REVIEW-PLAN.md 生成
- **SC-Q10**：单 PR ≤ 4 commits（C1 skill 重组 + patterns endpoint / C2 events heartbeat + journal extends / C3 OpenAPI export + 5 SKILL.md / C4 E2E + demo client + gate）

---

## Dependencies

**Upstream**：
- spec 010（OpenTelemetry tracing 基建）
- spec 015（metrics endpoint）
- spec 018（Memory Evolution — PatternRecord schema 定义）
- spec 019（Skill Evolution — EvolvingSkillProvider 路径迁移目标）
- spec 020a（Trilogy Ops — observability aggregator 模式复用）
- spec 020b（Evolution Daemon — 加 record_evolution_event 调用钩子）
- spec 021（Pattern Cold-Start — patterns 已落地，本 spec 暴露之）

**Downstream**：
- spec 023（A/B Experiment 框架 — 可消费本 spec events stream 作为 A/B 数据源）

**External tooling**：
- Anthropic Agent Skills 规范（SKILL.md frontmatter）
- OpenAPI 3.0+ 工具链（Stoplight / Swagger UI / Postman 自动消费 `docs/api/*.yaml`）

---

## Assumptions

- spec 020a sliding window aggregator 模式可直接复用（不需新建抽象）
- spec 019 `EvolvingSkillProvider` 路径迁移 atomic mv + 测试 fixture 同步改不破坏其他 spec
- spec 021 已落地 ≥ 3 patterns 在 `agent_memory/{tech,chain}/patterns/` — SC-Q3 验证基础
- arena serve 重启会 reload skill provider — 本 spec ship 时配合重启吸收 path migration
- 单 API_KEY auth 足够（本 spec 不实现 per-agent JWT，B 方案仅 design 存档于 cryptotrader/SKILL.md future migration path 段落）
- 外部 agent 多数能跑 Python（demo client 用 Python 而非 Node.js）
- pre-commit hook 已配置 `.pre-commit-config.yaml` 或 `lefthook.yml`（如未配，本 spec C3 加）

---

## Out of Scope

- ❌ A/B Experiment 框架（→ spec 023）
- ❌ Polymarket 信号源接入（→ 独立 spec）
- ❌ Data prefetch worker 拆分（→ 独立 spec）
- ❌ Agent 贡献度归因 leaderboard（→ 独立 spec）
- ❌ Social copy-trade / tradesync / heartbeat 上推（**永久 out of scope**，不做 social 方向）
- ❌ Per-agent JWT + register endpoint 完整实现（仅在 `cryptotrader/SKILL.md` design 文档化 future migration path）
- ❌ Public hosting / open-source release（保持 self-hosted 私有部署）
- ❌ OpenClaw / Mastra / autogen 任何 agent framework 绑定（framework-agnostic via 标准 HTTP + Anthropic Agent Skills 规范）
- ❌ Write endpoint（external agent 全 read-only — 不允许下单 / 改配置 / 写 patterns / drafted skills）

---

## Reversibility

本 spec 落地后可通过 git revert 单 PR + docker compose restart 回退（无 schema 变更，无破坏性数据迁移）：

- `agent_skills/_internal/` 改回 `agent_skills/`（git mv 回退）+ provider 路径常量回退
- `agent_skills/_external/` 删除（不影响生产）
- `src/api/routes/events.py` 删除（不影响生产）
- `src/api/routes/memory.py` 移除 `/api/memory/patterns` handler（不影响生产）
- `src/cryptotrader/observability/heartbeat_metrics.py` 删除
- SQL view `events_heartbeat` drop（不影响 `journal` 表原数据）
- `docs/api/*.yaml` 删除（runtime `openapi.json` 仍可用）
- `scripts/{export_openapi,demo_external_client}.py` 删除（不影响生产）
- `/metrics` 移除 3 新 gauge（不影响 spec 020a 既有 gauge）

---

## Implementation Outline（单 PR 切 4 commit）

**C1 — skill 重组 + memory/patterns endpoint（闭 spec 021 T021）**：
- `agent_skills/{tech,chain,news,macro}/SKILL.md` → `agent_skills/_internal/...` (git mv)
- `src/cryptotrader/learning/evolving_skill_provider.py` 路径常量改
- `src/api/routes/memory.py` 加 `GET /api/memory/patterns` handler
- spec 019 测试 fixture path 同步改
- 单测 `tests/test_api_memory_patterns.py`

**C2 — events heartbeat endpoint + journal extends**：
- `src/api/routes/events.py` 新增
- `src/cryptotrader/nodes/journal.py` 加 `record_phase1_rejection()` + `record_evolution_event()`
- spec 020b daemon 加 evolution event 写入钩子
- `src/cryptotrader/observability/heartbeat_metrics.py` 新增
- `src/api/routes/metrics.py` 注册 3 gauge
- SQL view `events_heartbeat` migration
- 单测 `tests/test_api_events_heartbeat.py`

**C3 — OpenAPI 静态化 + 5 SKILL.md + /skill/<name> endpoint**：
- `scripts/export_openapi.py` 新增
- pre-commit hook 配置（更新 `.pre-commit-config.yaml`）
- 生成 6 yaml 提交（5 分 + 1 combined）
- `agent_skills/_external/{cryptotrader,verdict-feed,market-intel,evolution-insights,execution-replay}/SKILL.md` 5 文件
- `src/api/routes/skills.py` 新增 `GET /skill/<name>` handler

**C4 — E2E + demo client + 最终 Gate**：
- `scripts/demo_external_client.py` 新增（80 行 Python）
- `tests/test_e2e_skill_protocol.py`（fetch SKILL.md → fetch patterns → poll heartbeat 全链）
- ruff / pytest / SC-Q* 全 gate
