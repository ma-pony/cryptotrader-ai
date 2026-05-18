# Implementation Plan: Agent-Native Skill Protocol Layer

**Branch**: `025-agent-native-skill-protocol` | **Date**: 2026-05-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/025-agent-native-skill-protocol/spec.md`

## Summary

外部 SKILL.md 协议 + OpenAPI 静态化 + Heartbeat 事件端点，同时闭 spec 021 T021 outstanding gap（`/api/memory/patterns` endpoint 缺失）。让外部 AI agent（Codex/Cursor/Claude Code）通过单条 `Read /skill/cryptotrader` 消息秒级集成。技术方案：FastAPI 新增 3 路由（`/api/memory/patterns`、`/api/events/heartbeat`、`/skill/<name>`）+ 5 个外部 SKILL.md（YAML frontmatter + 自描述 markdown）+ in-process OpenAPI 导出脚本 + 复用 spec 020a sliding window aggregator 模式加 3 个 Prometheus gauge + 80 行 Python demo client 验证端到端。

## Technical Context

**Language/Version**: Python 3.12+（uv 管理；`_compat.py` shim 已存 spec 020c）
**Primary Dependencies**: FastAPI / Pydantic v2 / SQLAlchemy 2.x / Prometheus client / OpenTelemetry SDK / pyyaml — 全部已存在
**Storage**: 复用现有 PostgreSQL（journal 表）+ 文件系统（`agent_skills/_external/*.md` + `agent_memory/<agent>/patterns/*.md`）；新建 SQL view `events_heartbeat`（无 schema migration）
**Testing**: pytest + pytest-asyncio + httpx TestClient（unit + integration + e2e 覆盖）
**Target Platform**: Linux server（macOS dev OK）；arena serve 监听 :8003
**Project Type**: web-service backend + 5 SKILL.md markdown 文档 + 2 scripts/* 工具
**Performance Goals**:
- `/skill/<name>` 端点 < 200ms p95（filesystem read）
- `/api/memory/patterns` < 500ms p95（filesystem scan + frontmatter parse）
- `/api/events/heartbeat` < 1s p95（SQL view query + cursor 编码）
- demo client 端到端 < 30s（4 fetch + 1 poll）
**Constraints**:
- 全 read-only（外部 agent 不允许 write）
- 单 API_KEY auth（per-agent JWT 仅 design 文档化）
- 无新 runtime 依赖
- 不破坏 spec 014-021 公开 API
**Scale/Scope**: 单用户 + 5-10 个 helper agents 并发；patterns ~10-100 文件 + journal 表 ~1000-10000 行/天

## Constitution Check

*Constitution 是 template-only（`.specify/memory/constitution.md` 含全 placeholder）。无定制 governance gates 可执行。本 spec 仍遵循项目通用约定（CLAUDE.md）：*

- ✅ Markdown 简体中文
- ✅ 直接删旧不留 fallback
- ✅ 3-phase approval workflow（Spec → Plan → Tasks → Implementation）— 当前 Plan 阶段
- ✅ Spec-kit + spex workflow（brainstorm 已 done）

Re-check post Phase 1：N/A（无 constitution gates）

## Project Structure

### Documentation (this feature)

```text
specs/025-agent-native-skill-protocol/
├── plan.md              # 本文件
├── spec.md              # 已生成
├── checklists/
│   └── requirements.md  # 已生成
├── research.md          # Phase 0 — 9 Q decision 溯源
├── data-model.md        # Phase 1 — HeartbeatEvent / PatternRecord / SkillRecord entities
├── quickstart.md        # Phase 1 — SC 验证步骤
├── contracts/           # Phase 1 — 3 个 OpenAPI yaml（skill / events / patterns）
└── tasks.md             # Phase 2（/speckit-tasks 生成）
```

### Source Code (repository root)

```text
src/
├── api/routes/
│   ├── events.py                ← 新增（heartbeat endpoint）
│   ├── memory.py                ← 修改（+ patterns endpoint）
│   ├── skills.py                ← 新增（/skill/<name> markdown 端点）
│   └── metrics.py               ← 修改（+ 3 Prometheus gauge）
├── cryptotrader/
│   ├── learning/
│   │   ├── evolving_skill_provider.py  ← 修改（路径常量 → _internal/）
│   │   └── memory.py            ← 复用 _load_pattern_record（spec 021）
│   ├── nodes/journal.py         ← 修改（+ record_phase1_rejection + record_evolution_event）
│   ├── observability/
│   │   └── heartbeat_metrics.py ← 新增（3 aggregator）
│   └── ops/daemon.py            ← 修改（加 record_evolution_event 钩子）

agent_skills/
├── _internal/                   ← 重组：原 4 agent prompt skills
│   ├── tech/SKILL.md            ← git mv from agent_skills/tech/
│   ├── chain/SKILL.md
│   ├── news/SKILL.md
│   └── macro/SKILL.md
└── _external/                   ← 新增：对外协议层
    ├── cryptotrader/SKILL.md       ← bootstrap
    ├── verdict-feed/SKILL.md
    ├── market-intel/SKILL.md
    ├── evolution-insights/SKILL.md
    └── execution-replay/SKILL.md

scripts/
├── export_openapi.py            ← 新增（in-process FastAPI app → 拆 5 yaml）
└── demo_external_client.py      ← 新增（80 行 reference client）

docs/api/
├── verdict.yaml                 ← 生成
├── market.yaml                  ← 生成
├── events.yaml                  ← 生成
├── execution.yaml               ← 生成
├── memory.yaml                  ← 生成（含 patterns endpoint）
└── openapi.yaml                 ← 生成（combined）

tests/
├── test_api_memory_patterns.py        ← 新增（闭 T021 单测）
├── test_api_events_heartbeat.py       ← 新增（cursor / types / limit）
├── test_e2e_skill_protocol.py         ← 新增（端到端）
└── test_skill_provider_internal_path.py ← 新增（路径迁移验证）

.pre-commit-config.yaml          ← 修改（加 openapi auto-regen hook）
```

**Structure Decision**: 复用项目既有 monolithic FastAPI 结构（`src/api/routes/*.py` 一对一路由）。无新模块层级。`agent_skills/_internal/` + `_external/` 分类纯 organizational，不引入新 import 路径。

## Complexity Tracking

无 NEEDS CLARIFICATION，无 constitution violations。9 个关键设计 Q 全在 brainstorm 阶段 resolved（详见 `research.md`）。

---

## Implementation Phases Reference

- **Phase 0**：研究 + 决策溯源 → `research.md`
- **Phase 1**：设计实体 + 接口契约 → `data-model.md` / `contracts/*.yaml` / `quickstart.md`
- **Phase 2**：任务拆分 → `tasks.md`（由 `/speckit-tasks` 生成）
- **Phase 3**：实施 → 4 commit（C1-C4 见 spec.md Implementation Outline）
- **Phase 4**：评审 → `/spex:review-code` + `/spex:stamp`
