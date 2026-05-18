# Spec 022 — Agent-Native Skill Protocol Layer（外部 SKILL.md 协议 + OpenAPI 静态化 + Heartbeat 事件端点）

**关联 spec**：[019](../specs/020-skill-evolution/) / [020a](../specs/021-trilogy-ops/) / [020b](../specs/022-evolution-daemon/) / [021](../specs/024-pattern-cold-start/)
**Date**: 2026-05-18
**Status**: brainstorm 完成，待 `/speckit-specify`

---

## 问题陈述

trilogy 进化系统 + spec 021 数据链补完后，**评估系统从 humans-only docs 升级为 agents-can-read-it docs 的对外接口** 成为下一个 ROI 最高的工作。

参考 HKUDS/AI-Trader（18k stars，2026-05-13 推送）的核心创新「SKILL.md as universal integration」— agent 通过单条 `Read https://.../SKILL.md and register` 消息就能秒级集成。cryptotrader-ai 当前 `agent_skills/`（spec 019）已实现 SKILL.md 协议的「内部 prompt 注入」一半，但**没用它做对外接口**。

具体痛点：
1. **你自己用 Codex/Cursor 时**，每次都要手动告知 AI "去 /api/snapshot 拉数据"、"看看 verdict 怎么决策的" — agent 无法自我发现接口
2. **trilogy 产物孤岛化**：spec 018/019/020a/b/c/021 产出的 patterns / skills / Pareto frontier 全在 disk 上，外部 agent 0 visibility
3. **spec 021 T021 单点遗留**：`/api/memory/rules` endpoint 在 src/ 不存在（spec drift），3 个 patterns disk 上 working 但 API 不可见

spec 022 一次性解决这 3 个问题 — 单点 fix 升级为系统化协议能力。

---

## 与 HKUDS/AI-Trader 的差异定位

| 维度 | HKUDS/AI-Trader | 本 spec 022 |
|---|---|---|
| 用户模型 | 多用户 social platform（agents 之间互相 follow / copy-trade） | 单用户私有部署（你 1 人 + 外部 helper agents） |
| Auth | per-agent JWT + register flow | 单 API_KEY（spec 022 不实现 register） |
| Skills 数量 | 5+ social-focused（copytrade/tradesync/heartbeat/market-intel/polymarket） | 5 trading-engine-focused（cryptotrader/verdict-feed/market-intel/evolution-insights/execution-replay） |
| 写能力 | tradesync 是 write 接口（上传 positions） | **全 read-only** — 不允许外部 agent 改状态 |
| 协议形式 | YAML frontmatter SKILL.md + URL 自描述 | 同（复用 Anthropic Agent Skills 规范） |
| 重点 | social graph 协议 | trading-engine 决策链 visibility |

**关键：复用 SKILL.md 协议形式 + 拒绝 social / write 概念**。

---

## Scope（P0 四项）

### P0-1：外部 SKILL.md 协议层
重组 `agent_skills/`：拆 `_internal/`（保留现有 spec 019 4 agent prompt skills）+ `_external/`（新增对外协议层）。

`_external/` 含 5 个 child skill：

| Skill | 职责 | 关联 routes |
|---|---|---|
| `cryptotrader/SKILL.md` | bootstrap — auth + 路由到 child skills（含 install-locally 建议） | 元数据 |
| `verdict-feed/SKILL.md` | 暴露 verdict 决策流 | `GET /api/verdicts/recent` + heartbeat 订阅 |
| `market-intel/SKILL.md` | 暴露 4 agent 输出 + snapshot 数据 | `GET /api/snapshot/{pair}` + `GET /api/agents/{name}/results/recent` |
| `evolution-insights/SKILL.md` | 暴露 trilogy 产出 | `GET /api/memory/skills` (已存在) + **`GET /api/memory/patterns` (新增, 闭 T021)** + `GET /api/memory/skill-proposals` (已存在) |
| `execution-replay/SKILL.md` | 暴露 journal events + OCO 状态查询 | `GET /api/journal/events` + `GET /api/execution/oco-status` |

复用 spec 019 既有 SKILL.md frontmatter 格式（`---\nname: ...\ndescription: ...\n---` YAML head + markdown body）。

### P0-2：OpenAPI 静态化
- 加 CI/pre-commit hook 把运行时 `/openapi.json` dump 到 checked-in 文件
- 按 router 拆分 5 个 yaml（与 SKILL.md scope 大致对齐）+ 1 个 combined
- SKILL.md 引用这些静态 yaml 作为 stable schema 锚点（agent 读 SKILL.md 时不需 server 跑起来）

### P0-3：Heartbeat /events 端点
```
GET /api/events/heartbeat?since=<ISO8601>&limit=50&types=[verdict,trade,rejection,evolution]
```
- 复用现有 `journal` 表 + 新建 SQL view `events_heartbeat`（聚合 verdict/trade/rejection/evolution 4 类）
- cursor-based pagination（按 `(timestamp, trace_id)` 字典序）
- OTel `events.heartbeat.poll` span 含 `client_identifier` attr（API_KEY hash 后 8 位）
- 复用 spec 020a `CacheMetricsAggregator` 模式：`HeartbeatPollAggregator` (24h sliding window) + Prometheus gauge `events_heartbeat_poll_count_24h` / `events_heartbeat_poll_lag_seconds`

### P0-4（衔接 spec 021）：闭 T021 — `GET /api/memory/patterns`
- 复用 `cryptotrader.learning.memory` 模块的 PatternRecord 加载器
- 返回 `{items: PatternRecord[], total: int}` 与 `/api/memory/skills` schema 对齐
- 支持 `?agent=tech&maturity=observed&limit=50` query params
- evolution-insights/SKILL.md 引用此 endpoint

---

## 关键决策（9 Q 全部 resolved，标注 Decision + 理由）

### Q1：内部 / 外部 skill 物理路径 → **Decision: A**

`agent_skills/_internal/` + `agent_skills/_external/`

**理由**：
- spec 019 `EvolvingSkillProvider` 已 hard-code `agent_skills/<agent>/` 路径，方案 A 改动最小（单 const 改 + 测试 fixture 路径加 `_internal/` 前缀）
- `_` 前缀按 Python 惯例信号「structural / private classification」— 不是新增 "下划线开头的 skill"
- 单根目录便于 git diff + permission 锁定
- 不取 C（renames `agent_skills/` → `skills/`）因为 spec 019 测试 / agent_memory 等全部引用 `agent_skills/`，blast radius 大

**migration**：spec 022 C1 一次性原子 commit 完成（mv 旧 4 个 SKILL.md → `_internal/{tech,chain,news,macro}/SKILL.md` + update `EvolvingSkillProvider` 路径 const + update 现有测试）。按用户偏好「直接删旧不留 fallback」无 alias。

### Q2：External skill 数量 + 边界 → **Decision: 5（如 P0-1 表）**

**理由**：
- 5 个 skill 每个 self-contained，agent 单次 task 只 fetch 1-2 个 child skill（不会 over-fetch）
- 不合并 verdict-feed + market-intel：verdict 是决策（含 thesis + scale），market-intel 是原始输入（4 agent direction/confidence）— 概念维度不同
- 不拆 OCO-status / risk-state 独立：execution-replay 自然包含 OCO + journal events，太细容易碎片化
- 命名风格借鉴 AI-Trader `copytrade` / `tradesync`：动宾结构，agent 读 description 即懂用途

### Q3：Auth 模型 → **Decision: A（保留单 API_KEY）+ B 设计存档**

**理由**：
- 你是 1 人 + 自用 helper agents（Codex/Cursor）场景，per-agent JWT 是 overkill
- AI-Trader 走 B 因 multi-tenant SaaS，你不需要
- SKILL.md 中写 `Include header: Authorization: Bearer <YOUR_API_KEY>` — 协议上看起来等价 JWT，未来若需要多租户只需替 API_KEY 为 JWT generation logic 即可
- spec 022 仅在 cryptotrader/SKILL.md 文档化 B 的 future migration path，**不实装**

### Q4：Heartbeat event sources → **Decision: 4 must + 2 opt**

**必含 4 类**：
1. `verdict_decision`（action/scale/thesis/cf/divergence）
2. `journal_trade_committed`（algo_id/sz/sl/tp/entry/exit）
3. `risk_gate_rejected`（pair/check_name/reason）
4. `phase1_rejected`（pair/reason: low_rr/stop_too_tight/missing_sl_tp/direction_inverted）

**可选 2 类**（默认 off，`?types=` 显式 enable）：
5. `evolution_event`（new pattern created / new skill drafted / Pareto rerank done）
6. `oco_state_change`（algo placed / cancelled / triggered）

**排除**：
- ❌ `cycle_pair_start`（太频繁 — 4 pair × 1h cycle = 96/day，纯噪音）
- ❌ raw agent_result（4 agent × 4 pair = 16/cycle，应通过 market-intel/agents/{name}/results endpoint 拉）
- ❌ debate_round（中间态，verdict 已是终态）

### Q5：Heartbeat 持久化层 → **Decision: A（复用 journal + view）**

**理由**：
- journal 表已存在（spec 014/015）+ 已写 verdict/trade/rejection 全部事件
- 新建 SQL view `events_heartbeat` 投影 4 类 must + 2 类 opt — 0 migration / 0 新表
- Phase 1 rejects 当前只 log 不进 journal — spec 022 C2 加一行 `journal.record_phase1_rejection()` 即可
- evolution_event 当前在 daemon log — spec 022 加 `journal.record_evolution_event()` 写入 — 配合 spec 020b daemon 已有钩子
- 性能：journal 表只读 query 用 indexed `(timestamp, event_type)`，1k 行 < 5ms

### Q6：OpenAPI 拆分粒度 → **Decision: B（按 router 拆 5 yaml + combined）**

**5 个分文件**：
- `docs/api/verdict.yaml`（routes/verdict.py 出 paths）
- `docs/api/market.yaml`（routes/snapshot.py + routes/agents.py）
- `docs/api/events.yaml`（routes/events.py 新增）
- `docs/api/execution.yaml`（routes/journal.py + routes/execution.py）
- `docs/api/memory.yaml`（routes/memory.py 含 patterns 新 endpoint）

**1 个 combined**：`docs/api/openapi.yaml`（工具链兼容如 Postman/Insomnia/Stoplight）

**生成方式**：`scripts/export_openapi.py` 启动一次 FastAPI app（in-process，不监听端口）调 `app.openapi()` → 按 router tag 拆 → 写 yaml。pre-commit hook + CI 都跑。

### Q7：兼容性 / 迁移策略 → **Decision: 一次性原子 migration**

spec 019 `EvolvingSkillProvider` 路径常量改 `agent_skills/` → `agent_skills/_internal/`，对应 4 个 SKILL.md 文件 git mv 到新路径。spec 019 现有 testing fixture 同步 mv。

无 alias / 无 fallback（用户偏好）。一次 commit 完成。

**风险评估**：
- spec 019 测试 ~50 处 path 引用（搜 `agent_skills/`）— 全 grep+sed 一次性改
- 生产 agent_memory/ 不受影响（不同顶级目录）
- arena serve 重启会 reload skill provider — 配合 spec 022 ship 流程的 restart 一次性吸收

### Q8：Demo / 验证 client → **Decision: A（提供 scripts/demo_external_client.py）**

**理由**：
- 单文件 ~80 行 Python：fetch SKILL.md → fetch /openapi.yaml → call /api/memory/patterns → poll /api/events/heartbeat
- 作为 spec 022 SC-Q1 验证（"reference client 端到端 < 30s 完成")
- 双倍价值：你自己接 Codex/Cursor 时这个 demo 就是 prompt template
- AI-Trader 没提供 reference client（只给 curl examples），我们做得比它好

### Q9：`/api/memory/patterns` 实现 → **Decision: B（复用 memory.py）**

**理由**：
- patterns 是 PatternRecord schema（spec 018 定义），不同于 SkillRecord
- `cryptotrader.learning.memory` 模块已有 `_load_pattern_record(path: Path)` helper（spec 021 实装）
- 加新 route handler ~20 行：扫 `agent_memory/{tech,chain,news,macro}/patterns/*.md` → 调 helper parse frontmatter → Pydantic schema → 返回
- 不走 EvolvingSkillProvider（那是 spec 019 skill 路径，schema 不同）
- 不写 events 表（patterns 已经文件持久化，不需双写）

---

## 约束（继承+新增）

- ✅ 不破坏 spec 014/015/017a/17b/018/019/020a/b/c/021 公开 API（_internal/ 迁移 atomic + provider 路径常量改）
- ✅ 不引入新 runtime 依赖
- ✅ **全 read-only** — 不开放任何 write endpoint 给外部 agent
- ✅ 不暴露 secrets（API_KEY 不在 SKILL.md，credentials 不在 journal events，agent_memory/ 已 gitignore）
- ✅ Markdown 简体中文（CLAUDE.md 规则）
- ✅ 直接删旧不留 fallback
- ✅ 不引入 social / copy-trade / leaderboard 概念

---

## Out of Scope（→ 进 spec 023+）

- ❌ A/B Experiment 框架（→ spec 023）
- ❌ Polymarket 信号源接入（→ 独立 spec）
- ❌ Data prefetch worker 拆分（→ 独立 spec）
- ❌ Agent 贡献度归因 leaderboard（→ 独立 spec）
- ❌ Social copy-trade / tradesync（**永久 out of scope**）
- ❌ Per-agent JWT + register endpoint 完整实现（仅 design 文档化）
- ❌ Public hosting / open-source release（保持 self-hosted）
- ❌ OpenClaw / Mastra / autogen 任何 agent framework 绑定（framework-agnostic）
- ❌ Write endpoint（不允许外部 agent 改 cryptotrader-ai 状态）

---

## Success Criteria（10 条预估）

- SC-Q1：scripts/demo_external_client.py 端到端跑通（< 30s），fetch SKILL.md → fetch patterns → poll heartbeat 全 PASS
- SC-Q2：5 个 _external/SKILL.md 文件存在 + 通过 markdown lint + YAML frontmatter valid
- SC-Q3：`curl /api/memory/patterns` 返回 `total >= 3`（闭 spec 021 T021 SC-P3）
- SC-Q4：`curl /api/events/heartbeat?since=2026-05-18T00:00:00Z` 返回最近事件 ≤ 50（cursor 正确）
- SC-Q5：`docs/api/{verdict,market,events,execution,memory}.yaml` 5 文件 valid OpenAPI 3.0+ + combined `docs/api/openapi.yaml`
- SC-Q6：pre-commit hook 检测 routes_*.py 变化时自动 regen openapi yaml + git stage
- SC-Q7：spec 019 既有测试不回归（baseline ≥ 2476 pass / 0 fail，post-spec 021）
- SC-Q8：3 个新 Prometheus gauge（`events_heartbeat_poll_count_24h` / `_lag_seconds` / `external_skill_fetch_count_24h`）在 `/metrics` 可见
- SC-Q9：`/spex:review-spec` + `/spex:review-plan` 无 P0/P1 issues
- SC-Q10：单 PR ≤ 4 commits（C1 skill 重组 + memory route / C2 events route + journal extends / C3 OpenAPI export + SKILL.md / C4 E2E + demo client + gate）

---

## 范围预估

- 1-2 周
- ~12 个新文件：
  - `agent_skills/_external/cryptotrader/SKILL.md`
  - `agent_skills/_external/verdict-feed/SKILL.md`
  - `agent_skills/_external/market-intel/SKILL.md`
  - `agent_skills/_external/evolution-insights/SKILL.md`
  - `agent_skills/_external/execution-replay/SKILL.md`
  - `src/api/routes/events.py`（新增 heartbeat endpoint）
  - `src/api/routes/memory.py` 改（+ patterns endpoint）
  - `src/cryptotrader/observability/heartbeat_metrics.py`（HeartbeatPollAggregator）
  - `scripts/export_openapi.py`
  - `scripts/demo_external_client.py`
  - `docs/api/{verdict,market,events,execution,memory,openapi}.yaml`（6 yaml）
- ~150-200 行新 Python 代码
- ~1000 行 SKILL.md markdown（含 examples + curl snippets）

---

## 商业 / 战略价值

即使是 1 人项目：
1. **你自己用 Codex/Cursor 立即受益** — 一行 prompt（`Read https://localhost:8003/skill/cryptotrader`）替代每次手动告知 agent 接口
2. **架构成熟度 signal** — agents-can-read-it docs > humans-only docs，这是 trilogy 进化系统的逻辑终点
3. **trilogy 产物变现** — spec 018/019/020a/b/c/021 产出的 patterns / skills / Pareto frontier 现在 agent 可消费，否则 disk 上孤岛化
4. **闭 spec 021 T021** — 单点 fix 升级为完整 protocol，避免单独开 spec-021-followup
5. **不锁定生态** — Anthropic Agent Skills 规范 + 标准 OpenAPI + standard HTTP，任何 agent framework 都能用
6. **未来可逆 / 可扩** — 加 per-agent JWT 只需 API key resolver 改 2 行；加 SSE/WebSocket push 只需在 heartbeat endpoint 上层做

---

## 实施 outline（单 PR 切 4 commit）

**C1：skill 重组 + memory/patterns endpoint（闭 T021）**
- `agent_skills/{tech,chain,news,macro}/SKILL.md` → `agent_skills/_internal/{tech,chain,news,macro}/SKILL.md` (git mv)
- `src/cryptotrader/learning/evolving_skill_provider.py` 路径常量改
- `src/api/routes/memory.py` 加 `GET /api/memory/patterns` handler
- spec 019 测试 fixture path 同步改
- 复用 spec 020a aggregator 模式：`PatternFetchAggregator` (24h)

**C2：events heartbeat endpoint + journal extends**
- `src/api/routes/events.py` 新增（heartbeat handler + cursor pagination）
- `src/cryptotrader/nodes/journal.py` 加 `record_phase1_rejection()` + `record_evolution_event()`
- `src/cryptotrader/observability/heartbeat_metrics.py` 新增
- `src/api/routes/metrics.py` 注册 3 新 gauge
- SQL view `events_heartbeat` migration

**C3：OpenAPI 静态化 + 5 SKILL.md**
- `scripts/export_openapi.py`（in-process FastAPI app → openapi() → 拆 yaml）
- 加 pre-commit hook（`.pre-commit-config.yaml` 或 `lefthook.yml`）
- 生成 6 yaml 提交（按 router）
- `agent_skills/_external/{cryptotrader,verdict-feed,market-intel,evolution-insights,execution-replay}/SKILL.md` 5 文件

**C4：E2E + demo client + gate**
- `scripts/demo_external_client.py`（80 行 reference）
- `tests/test_e2e_skill_protocol.py`（fetch SKILL.md → fetch patterns → poll heartbeat 全链）
- `tests/test_api_memory_patterns.py`（闭 T021 单测）
- `tests/test_api_events_heartbeat.py`（cursor / types filter / limit）
- ruff / pytest / SC-Q* 全 gate

---

## 待用户确认 / 可重定向决策

虽然 9 Q 已全部给出 recommendation + Decision，以下 3 点对最终设计影响较大，欢迎 reject：

1. **Q3 Auth** — 是真的只用 API_KEY？还是想顺便上 per-agent token（增 ~3 天工作量）？
2. **Q7 migration** — spec 019 路径 atomic mv 还是保留旧路径 alias 一段时间（增 ~1 天 + 1 个 deprecation tracker）？
3. **Q8 demo client** — Python 还是 Node.js（外部 agent 多用 Node 接 LangChain.js 等）？

如全部接受 → 可以直接进 `/speckit-specify` 写 spec.md。
