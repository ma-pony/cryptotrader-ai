# Implementation Plan: Spec 020a — Trilogy Ops

**Branch**: `021-trilogy-ops` | **Date**: 2026-05-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/021-trilogy-ops/spec.md`

## Summary

Trilogy（spec 017a/b + 018 + 019）已合并 main，本 spec 收尾 Ops 子域。落地 5 项工作：

1. `scripts/staging_validate.py`：1-key staging smoke check 脚本
2. `docs/rollback-trilogy.md`：trilogy 3 spec 的 rollback runbook
3. `log_llm_usage()` 加 cache_creation_input_tokens + 写 OTel span attr
4. IVE `classify_case` async 化（`llm.invoke` → `await llm.ainvoke`）
5. SkillsGrid triggers_keywords badges + propose_new_skill `inference_failed` 持久化
6. `/metrics` 加 2 个 Prometheus 指标（cache hit rate / IVE failure rate），不告警

技术路径：复用既有基建（spec 010 OTel / spec 015 metrics endpoint / spec 018 IVE / spec 019 SkillsGrid），单 PR 4 commit 切分（C1 文档 + C2 后端 + C3 前端 + C4 E2E）。

## Technical Context

**Language/Version**: Python 3.12+ (后端) / TypeScript 5.9 (前端)
**Primary Dependencies**: LangChain 1.2+ / langchain-openai 1.1+ / OpenTelemetry SDK / FastAPI 0.135+ / React 19.2 / Vite 7
**Storage**: 不涉及（本 spec 无 schema 变更）
**Testing**: pytest 8.x + pytest-asyncio (后端) / Vitest (前端)
**Target Platform**: Linux server (生产) / macOS (dev)
**Project Type**: Web application（backend Python + frontend React + 共享 metrics endpoint）
**Performance Goals**:
- staging_validate.py 全程 ≤ 60s（mocked LLM）
- IVE classify_case async 化后单 cycle event loop 阻塞时间从 ~5s 降到 < 100ms
- OTel span attr 写入开销 ≤ 1ms / call
**Constraints**:
- 不引入新 runtime 依赖
- 不破坏 spec 014/15/17a/17b/18/19 公开 API
- 单 PR 4 commit
- 落地 ~3-5 天
**Scale/Scope**:
- 后端：~250 LOC 新增（staging_validate）+ ~150 LOC 修改（log_llm_usage / IVE / skill_metadata_inference / propose_new_skill）
- 前端：~80 LOC 新增（SkillsGrid badge row + 2 metric panel）
- 测试：~300 LOC 新增（staging_validate / cache attr / IVE async / SkillsGrid / metrics panel）

## Constitution Check

`.specify/memory/constitution.md` 不存在，跳过。

与 CLAUDE.md 既定规则对齐：
- ✓ Markdown 简体中文
- ✓ 直接删旧不留 fallback
- ✓ 不破坏既有 API
- ✓ 不引入新 runtime 依赖
- ✓ Spec-Plan-Tasks-Implementation 4 阶段流程

**Gate**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/021-trilogy-ops/
├── plan.md              # This file
├── spec.md              # Feature specification（已生成）
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── REVIEW-SPEC.md       # 已生成
├── tasks.md             # Phase 2 output（/speckit.tasks 生成）
└── checklists/
    └── requirements.md  # 已生成
```

### Source Code (repository root)

```text
src/cryptotrader/
├── agents/
│   └── base.py                            # MODIFY: log_llm_usage 加 cache_creation + OTel span attr
└── learning/
    ├── evolution/
    │   ├── ive.py                         # MODIFY: classify_case async 化
    │   └── skill_metadata_inference.py    # MODIFY: 失败路径写 inference_failed: True
    └── skill_proposal.py                  # MODIFY: propose_new_skill 写 frontmatter inference_failed

src/api/routes/
└── metrics.py                              # MODIFY: 加 2 个 Prometheus 指标（cache hit rate / IVE failure rate）

scripts/
└── staging_validate.py                     # NEW: staging smoke check 脚本

docs/
└── rollback-trilogy.md                     # NEW: trilogy 3 spec rollback runbook

web/src/pages/
├── memory/components/
│   └── SkillsGrid.tsx                     # MODIFY: 加 triggers_keywords badge row
└── metrics/
    └── index.tsx                           # MODIFY: 加 2 个 panel（cache hit rate / IVE failure rate）

tests/
├── test_staging_validate.py                # NEW
├── test_llm_usage_cache_attr.py            # NEW
├── test_ive_async.py                       # MODIFY: 既有 test_ive.py 改 asyncio
├── test_skill_proposal_metadata_inference.py  # MODIFY: 加 test_llm_failure_writes_flag
├── test_metrics_endpoint_cache.py          # NEW
└── test_e2e_trilogy_ops.py                 # NEW: mocked cycle 验证 OTel cache attr
```

**Structure Decision**：沿用既有 web application 结构（backend Python + frontend React + 共享 `/metrics` endpoint），无新模块层级；FR 全部 surgical 修改既有文件 + 新增 2 个文档/脚本。

## Complexity Tracking

无 Constitution 违规，无需填写。

## 实施约束 + 边界

- **C1 commit**：基础设施 + 文档（staging_validate.py + rollback-trilogy.md），纯新增，无 behavior 变化
- **C2 commit**：后端 algorithm + telemetry（log_llm_usage / IVE async / skill_metadata_inference / propose_new_skill 调用方），与既有 IVE 调用路径联动改动
- **C3 commit**：前端 + dashboard（SkillsGrid badge row + 2 metric panel），独立 frontend 改动
- **C4 commit**：E2E 测试 + 最终 gate（test_e2e_trilogy_ops + ruff + grep / wc）

落地后用户验证：
- 1 命令跑 `python scripts/staging_validate.py --dry-run` 全 PASS
- 1 mocked cycle 跑 OTel trace 含 cache attr ≥ 4 处 LLM call
- 前端 `/memory` 页 SkillsGrid 卡片显示 triggers_keywords badges
- 前端 `/metrics` 页含 2 个新 panel
