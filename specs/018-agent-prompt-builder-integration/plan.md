# 实施计划：Agent Prompt Builder Integration（spec 017b）

**Branch**: `018-agent-prompt-builder-integration` | **Date**: 2026-05-08 | **Spec**: [spec.md](spec.md)
**Input**: 来自 `specs/018-agent-prompt-builder-integration/spec.md`

## Summary

完成 spec 017a 的"集成切换"工作 —— 4 agent（tech / chain / news / macro）从硬编码 ROLE 路径切到 spec 017a 的 PromptBuilder 基建。整合 3 块旧路径：

1. `base.py` BaseAgent / ToolAgent 重构（删 `role_description` / `ANALYSIS_FRAMEWORK` / `_build_prompt`）
2. `config.py` AgentsConfig 重构（删 `_resolve_role` / `_resolve_skills` / `prompt_template`）
3. `agents/skills/middleware.py` 删除（由 DefaultSkillProvider + PromptBuilder._render_skills 替代）

技术路径：直接删旧不留 fallback。3 commit 单 PR：C1 纯新增 / C2 atomic 切换 / C3 E2E + gate。

## Technical Context

**Language/Version**: Python 3.12+（项目既定）
**Primary Dependencies**: PyYAML（已有）/ LangChain 1.2+（既有 SystemMessage / HumanMessage / create_agent）/ spec 017a `prompt_builder.py`（PromptBuilder + Provider Protocol + TokenBudgetEnforcer）
**Storage**: 文件系统 — `config/agents/*.md`（4 新文件）/ `agent_skills/<id>/SKILL.md`（spec 014 既有）/ `agent_memory/cases/*.md`（spec 014 既有，per-cycle 全局，路径修复推迟 spec 018）
**Testing**: pytest + pytest-asyncio（既有）；新增 2 个测试文件 + 4 个 agent 测试更新
**Target Platform**: Linux server（生产）+ macOS（开发）
**Project Type**: Single project — 后端 Python 库 + CLI / FastAPI
**Performance Goals**: 1 cycle 4 次 PromptBuilder.build() 总耗时 < 200ms（不含 LLM 推理）
**Constraints**:
- 不引入新 runtime 依赖
- 不修改 spec 017a PromptBuilder 公开 API（仅向后兼容地新增 `experience: str = ""` 参数）
- 不修改 spec 014 `agent_skills/` / `agent_memory/` 目录结构
- 不修改 spec 010 OpenTelemetry tracing 集成
- DefaultMemoryProvider 路径修复显式不在本 spec 范围（推迟 spec 018）
**Scale/Scope**:
- 4 个 agent config 文件（每个 ~120-180 行 markdown）
- 1 个新模块（`snapshot_renderer.py`，~150 行预估）
- 2 个新测试模块（`test_snapshot_renderer.py` ~150 行 / `test_e2e_prompt_externalization.py` ~200 行）
- 4 个 agent test 更新（每个 ~50 行 diff）
- 删除文件 1 个（`agents/skills/middleware.py`）
- 修改文件 ~10 个
- 总 diff 估算 ~2000 行；3 commit 单 PR

## Constitution Check

`.specify/memory/constitution.md` 为模板占位符，无 gate。本 spec 已对齐项目 CLAUDE.md：
- ✓ Markdown 简体中文
- ✓ 不修改 CLAUDE.md
- ✓ 文件改动范围：`config/agents/*` / `src/cryptotrader/agents/*` / `src/cryptotrader/config.py` / `src/cryptotrader/nodes/agents.py` / `src/cryptotrader/security.py` / `tests/*` / `pyproject.toml` / 本 spec 目录
- ✓ 不引入新依赖
- ✓ 不替换 spec 014 / 015 / 017a 任何 invariant

**Constitution Check 状态**：PASS

## Project Structure

### Documentation (this feature)

```text
specs/018-agent-prompt-builder-integration/
├── plan.md              # 本文件
├── spec.md              # 已存在
├── REVIEW-SPEC.md       # stage 2 输出
├── checklists/
│   └── requirements.md  # 已存在
├── research.md          # Phase 0 输出
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── contracts/
│   └── promptbuilder-experience-extension.md
├── tasks.md             # Phase 2 输出（speckit-tasks 生成）
└── REVIEW-PLAN.md       # Stage 5 输出
```

### Source Code (repository root)

```text
config/agents/
├── tech.md      # NEW — C1
├── chain.md     # NEW — C1
├── news.md      # NEW — C1
└── macro.md     # NEW — C1

src/cryptotrader/agents/
├── base.py                          # 重构 — C2
├── prompt_builder.py                # 重构 — C2
├── snapshot_renderer.py             # NEW — C1
├── tech.py / chain.py / news.py / macro.py  # 重构 — C2
└── skills/
    └── middleware.py                # 删除 — C2

src/cryptotrader/
├── config.py                        # 重构 — C2
└── security.py                      # 修注释 — C2

src/cryptotrader/nodes/
└── agents.py                        # 重构 — C2

tests/
├── fixtures/
│   ├── agent_configs/example.md     # spec 017a 已有
│   ├── memory/                      # 新增 mock（用于 e2e）
│   └── skills/                      # 新增 _test_shared / _test_tech（SC-Y13）
├── test_snapshot_renderer.py        # NEW — C1
├── test_tech_agent.py / test_chain_agent.py / test_news_agent.py / test_macro_agent.py  # 更新 — C2
└── test_e2e_prompt_externalization.py  # NEW — C3

pyproject.toml                       # 更新 — C3（如需 RUF001-003 ignores）
```

**Structure Decision**: Single project — 沿用 spec 017a 落地的目录结构。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |

无复杂度违反项。
