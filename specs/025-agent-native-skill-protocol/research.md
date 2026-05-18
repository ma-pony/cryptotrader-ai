# Phase 0 Research: Agent-Native Skill Protocol Layer

**Date**: 2026-05-18
**Spec**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md)

## 目标

记录 spec 022 在 brainstorm 阶段已 resolved 的 9 个关键决策，作为后续 task / 实现的设计依据。所有 NEEDS CLARIFICATION 已在 brainstorm 阶段消除（`brainstorm/10-spec-022-agent-native-skill-protocol.md`）。

---

## Decision 1：内部 / 外部 skill 物理路径

**Decision**：A — `agent_skills/_internal/` + `agent_skills/_external/`（同根目录，下划线前缀分类）

**Rationale**：
- spec 019 `EvolvingSkillProvider` 已 hard-code `agent_skills/<agent>/` 路径，方案 A 改动最小（单 const 改 + 测试 fixture 路径加 `_internal/` 前缀）
- `_` 前缀按 Python 惯例信号「structural / private classification」— 不是新增 "下划线开头的 skill"
- 单根目录便于 git diff + permission 锁定
- 拒 C（`skills/` 顶级 rename）因 blast radius 大（spec 019 测试 / agent_memory 等全部引用 `agent_skills/`）

**Alternatives considered**：
- B：`agent_skills/` + 新建 `external_skills/` 顶级目录 — 顶级分离更清晰但 git churn 更多
- C：`skills/internal/` + `skills/external/` AI-Trader 风格 — rename 既有目录风险大

**实现影响**：
- spec 019 `EvolvingSkillProvider` 一处 PATH const 改
- 现有 4 个 `agent_skills/{tech,chain,news,macro}/SKILL.md` git mv 一次 atomic
- 测试 fixture path 同步改（grep + sed）

---

## Decision 2：External skill 数量 + 边界

**Decision**：5 个（cryptotrader bootstrap + verdict-feed + market-intel + evolution-insights + execution-replay）

**Rationale**：
- 5 个 skill 每个 self-contained，agent 单次 task 只 fetch 1-2 个 child skill（不会 over-fetch）
- 不合并 verdict-feed + market-intel：verdict 是决策（含 thesis + scale），market-intel 是原始输入（4 agent direction/confidence）— 概念维度不同
- 不拆 OCO-status / risk-state 独立：execution-replay 自然包含 OCO + journal events，太细容易碎片化
- 命名风格借鉴 AI-Trader `copytrade` / `tradesync`：动宾结构，agent 读 description 即懂用途

**Alternatives considered**：
- 更少（3 个）：合并 verdict-feed + market-intel，合并 evolution-insights + execution-replay — 信息混杂，agent prompt 难裁剪
- 更多（7+ 个）：拆 OCO-status / risk-state / Pareto-frontier / skill-proposals 独立 — 碎片化、维护成本高

---

## Decision 3：Auth 模型

**Decision**：A — 保留单 API_KEY（B 方案 per-agent JWT 仅 design 文档化于 cryptotrader/SKILL.md future migration path）

**Rationale**：
- 单用户场景 + 自用 helper agents，per-agent JWT 是 overkill
- AI-Trader 走 B 因 multi-tenant SaaS，cryptotrader-ai 不需要
- SKILL.md 中写 `Include header: Authorization: Bearer <YOUR_API_KEY>` — 协议上看起来等价 JWT，未来扩 multi-tenant 只需替 API_KEY 为 JWT generation logic 即可
- 不实装 register endpoint（无 multi-user 用例）

**Alternatives considered**：
- B：完整 per-agent JWT + register endpoint — 增 ~3 天工作量，但 0 实际收益
- C：分级（API_KEY admin + read-only token）— 单用户不需要分级

---

## Decision 4：Heartbeat event sources

**Decision**：4 must + 2 opt + 3 排除

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
- ❌ raw `agent_result`（4 agent × 4 pair = 16/cycle，应通过 market-intel/agents endpoint 拉）
- ❌ `debate_round`（中间态，verdict 已是终态）

**Rationale**：信号 / 噪音平衡。must 4 类是 agent 决策必看；opt 2 类是 advanced use case；排除的是过频或中间态。

---

## Decision 5：Heartbeat 持久化层

**Decision**：A — 复用现有 `journal` 表 + 新建 SQL view `events_heartbeat`

**Rationale**：
- journal 表已存在（spec 014/015）+ 已写 verdict/trade/rejection 全部事件
- 新建 SQL view `events_heartbeat` 投影 4 类 must + 2 类 opt — 0 migration / 0 新表
- Phase 1 rejects 当前只 log 不进 journal — 加一行 `journal.record_phase1_rejection()` 即可
- evolution_event 当前在 daemon log — 加 `journal.record_evolution_event()` 写入 — 配合 spec 020b daemon 已有钩子
- 性能：journal 表只读 query 用 indexed `(timestamp, event_type)`，1k 行 < 5ms

**Alternatives considered**：
- B：新增 `events` typed column 表 — 结构化更好但需 migration，单用户不值得
- C：仅 OTel trace export — 延迟高 + 依赖 OTel backend 不可用时 fallback 困难

---

## Decision 6：OpenAPI 拆分粒度

**Decision**：B — 按 router 拆 5 yaml + 1 combined

**5 个分文件**：
- `docs/api/verdict.yaml`
- `docs/api/market.yaml`
- `docs/api/events.yaml`（新增）
- `docs/api/execution.yaml`
- `docs/api/memory.yaml`（含 patterns 新 endpoint）

**1 个 combined**：`docs/api/openapi.yaml`（工具链兼容如 Postman/Insomnia/Stoplight）

**生成方式**：`scripts/export_openapi.py` 启动 in-process FastAPI app（不监听端口）调 `app.openapi()` → 按 router tag 拆 → 写 yaml。pre-commit hook + CI 都跑。

**Rationale**：
- 匹配 AI-Trader 模式（`docs/api/openapi.yaml` + `docs/api/copytrade.yaml`）
- SKILL.md 引用稳定 per-skill yaml 路径（不会被 unrelated route 改动 churn）
- 拒 A（单文件）— route 列表长后 diff 噪音大；拒 C（per-skill yaml）— skill ↔ router 不是 1:1

---

## Decision 7：兼容性 / 迁移策略

**Decision**：一次性原子 migration（spec 019 路径常量改 + git mv 4 个 SKILL.md + 测试 fixture path 同步改 = 1 commit）

**Rationale**：
- 用户偏好「直接删旧不留 fallback」
- 无 alias / 无 deprecation tracker — atomic 简单
- 一次 commit 完成 → 单次 PR review

**风险**：
- spec 019 测试 ~50 处 path 引用（搜 `agent_skills/`）— grep+sed 一次性改，单测先跑过
- 生产 agent_memory/ 不受影响（不同顶级目录）
- arena serve 重启会 reload skill provider — 配合 spec 022 ship 时重启吸收

---

## Decision 8：Demo / 验证 client

**Decision**：A — 提供 `scripts/demo_external_client.py`（Python，~80 行）

**Rationale**：
- 单文件 reference 价值大：fetch SKILL.md → fetch /openapi.yaml → call /api/memory/patterns → poll /api/events/heartbeat
- 作为 spec 022 SC-Q1 验证（"reference client 端到端 < 30s 完成"）
- 双倍价值：你自己接 Codex/Cursor 时这个 demo 就是 prompt template
- AI-Trader 没提供 reference client（只给 curl examples）— cryptotrader-ai 做得比它好
- Python 而非 Node.js：项目本身 Python 栈 + httpx 已存在，0 新依赖

---

## Decision 9：`/api/memory/patterns` 实现方式

**Decision**：B — 复用 `cryptotrader.learning.memory` 模块（不通过 EvolvingSkillProvider）

**Rationale**：
- patterns 是 PatternRecord schema（spec 018 定义），不同于 SkillRecord
- `cryptotrader.learning.memory` 模块已有 `_load_pattern_record(path: Path)` helper（spec 021 实装）
- 新 route handler ~20 行：扫 `agent_memory/{tech,chain,news,macro}/patterns/*.md` → 调 helper parse frontmatter → Pydantic schema → 返回
- 不走 EvolvingSkillProvider（那是 spec 019 skill 路径，schema 不同）
- 不写 events 表（patterns 已经文件持久化，不需双写）

**Alternatives considered**：
- A：filesystem-only 直接读 .md 文件 — 但 frontmatter parse 逻辑会重复（已存于 memory.py）
- C：daemon push 写 events 表 — 引入双写一致性问题 + 现有 distill 流程不写 db

---

## 决策汇总表

| Q# | 主题 | Decision | 实现影响 |
|---|---|---|---|
| Q1 | skill 物理路径 | A — `agent_skills/_internal/` + `_external/` | 1 const 改 + git mv + fixture path 改 |
| Q2 | external skill 数量 | 5 个（不合并不细拆） | 5 个新 SKILL.md 文件 |
| Q3 | Auth 模型 | A — 保留单 API_KEY | 0 代码改（SKILL.md doc-only） |
| Q4 | Heartbeat events | 4 must + 2 opt | journal 加 2 helper + daemon 加钩子 |
| Q5 | Heartbeat 持久化 | A — 复用 journal + SQL view | 1 SQL view migration |
| Q6 | OpenAPI 拆分 | B — 5 yaml + combined | 1 export 脚本 + pre-commit hook |
| Q7 | migration | atomic 一次性 | 1 commit |
| Q8 | demo client | A — Python 80 行 | 1 script |
| Q9 | patterns 实现 | B — 复用 memory.py | 1 route handler ~20 行 |

无未决问题。可直接进入 Phase 1 设计 + Phase 2 任务拆分。
