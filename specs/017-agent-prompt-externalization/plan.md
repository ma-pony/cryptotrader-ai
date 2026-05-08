# 实施计划：Agent Prompt Builder Foundation（spec 017a，scope-narrowed）

**Branch**: `017-agent-prompt-externalization` | **Date**: 2026-05-08 | **Spec**: [spec.md](spec.md)
**Input**: 来自 `specs/017-agent-prompt-externalization/spec.md`

## Summary

交付 `PromptBuilder` 基建模块（`src/cryptotrader/agents/prompt_builder.py`）+ 单元测试，使 spec 017b（4 agent 迁移）与 spec 018（skill / memory 进化算法）可基于同一基建开发。本 spec 仅覆盖基建层，不修改 4 个 analysis agent 文件、不改 `base.py` / `config.py` / `nodes/agents.py` / `graph.py`。

### Scope Split History（2026-05-08）

实施过程中发现 `base.py` BaseAgent/ToolAgent 的 `role_description` 路径 + `config.py` AgentRegistry 的 `_resolve_role` / `_resolve_skills` + `agents/skills/middleware.py:SkillsInjectionMiddleware` 都需要整合，整合范围比原 6-task 估算大 2-3 倍。决策：拆分为 017a（基建）+ 017b（集成），本 plan 仅覆盖 017a。

技术路径基于 spec 016 决策：D-PA-01（Markdown frontmatter）、D-PA-02（system / user-tail 槽位）、D-MW-01（patterns + cases 混合）、D-MW-02 / D-MW-03（默认 Provider 实现）。

## Technical Context

**Language/Version**: Python 3.12+（项目既定）
**Primary Dependencies**: PyYAML（已有）、LangChain 1.2+（既有，提供 SystemMessage / HumanMessage）、`agents/base.py:create_llm()`（既有 LLM 工厂）
**Storage**: 文件系统 — `config/agents/*.md`（配置）、`agent_skills/<id>/SKILL.md`（spec 014 既有）、`agent_memory/<agent_id>/{patterns.md, cases.jsonl}`（spec 014 既有）
**Testing**: pytest + pytest-asyncio（既有）；新增 4 个测试文件（`test_prompt_builder.py` / `test_token_budget.py` / `test_config_loader.py` / `test_e2e_prompt_externalization.py`）+ 4 个 agent 测试更新
**Target Platform**: Linux server（生产）+ macOS（开发）— 与现有运行环境一致
**Project Type**: Single project — 后端 Python 库 + CLI / FastAPI（现有 `src/cryptotrader/` 单体）
**Performance Goals**: PromptBuilder.build() 单次调用 < 50ms（不含 LLM 推理）；token 估算误差 < 10% vs tiktoken
**Constraints**:
- 不引入新 runtime 依赖（不引入 tiktoken）
- 不修改 `agent_skills/` / `agent_memory/` schema（spec 018 才动）
- 不修改 verdict / debate / risk gate 节点
- 现有 OpenTelemetry tracing 体系内挂载新字段（spec 010 既有基础设施）
- Token 估算复用 spec 014 `_estimate_tokens()`（CJK-aware）
**Scale/Scope**（spec 017a，已完成）：
- 1 个新模块（`prompt_builder.py`，666 行实际）
- 3 个测试文件（44 用例 PASS）
- 1 个 fixture 文件（`tests/fixtures/agent_configs/example.md`）
- 1 个空目录（`config/agents/`）
- 0 个修改 4 agent 源码（隔离性 gate SC-X13）
- 1 个 commit（基建一次性提交）

## Constitution Check

`.specify/memory/constitution.md` 为模板占位符（无具体原则定义），无可强制 gate。本 spec 已对齐项目级 CLAUDE.md 规则：
- ✓ Markdown 内容简体中文
- ✓ 不修改 CLAUDE.md
- ✓ 仅在 `specs/017-*/` + `src/cryptotrader/agents/` + `config/agents/` + `tests/` 范围内创建/修改文件
- ✓ 不引入新 runtime 依赖
- ✓ 不替换 spec 014 任何 FR（仅替换 4 个 agent 的内部 prompt 拼接路径）

**Constitution Check 状态**：PASS（无具体 gate 可违反）。

## Project Structure

### Documentation (this feature)

```text
specs/017-agent-prompt-externalization/
├── plan.md              # 本文件
├── spec.md              # 已存在
├── REVIEW-SPEC.md       # 已存在
├── checklists/
│   └── requirements.md  # 已存在
├── research.md          # Phase 0 输出
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── contracts/           # Phase 1 输出
│   ├── prompt-builder.md
│   ├── memory-provider.md
│   ├── skill-provider.md
│   └── agent-config-schema.md
└── tasks.md             # Phase 2 输出（speckit-tasks 生成）
```

### Source Code (repository root)

```text
src/cryptotrader/agents/
├── __init__.py                     # 更新导出（增加 PromptBuilder / Protocol）
├── base.py                         # 既有（不动；create_llm() 已经 model 由配置驱动）
├── prompt_builder.py               # NEW — T1
├── tech.py                         # 重构 — T2（删 ROLE，构造器改必填 prompt_builder）
├── chain.py                        # 重构 — T3
├── news.py                         # 重构 — T4
└── macro.py                        # 重构 — T5

src/cryptotrader/nodes/
└── agents.py                       # 更新 — T6（4 处 agent 实例化注入 prompt_builder）

src/cryptotrader/
└── graph.py                        # 更新 — T6（如有 agent 实例化也同步）

config/agents/
├── tech.md                         # NEW — T2
├── chain.md                        # NEW — T3
├── news.md                         # NEW — T4
└── macro.md                        # NEW — T5

tests/
├── test_prompt_builder.py          # NEW — T1（≥7 用例）
├── test_token_budget.py            # NEW — T1（≥5 用例）
├── test_config_loader.py           # NEW — T1
├── test_tech_agent.py              # 更新 — T2
├── test_chain_agent.py             # 更新 — T3
├── test_news_agent.py              # 更新 — T4
├── test_macro_agent.py             # 更新 — T5
└── test_e2e_prompt_externalization.py  # NEW — T6
```

**Structure Decision**: Single project — 沿用项目现有 `src/cryptotrader/` 包结构。新增 `config/agents/` 顶层目录用于 agent 配置（与已有 `agent_skills/` / `agent_memory/` 平行）。

## Complexity Tracking

无复杂度违反项需要 justify — Constitution 为模板占位无 gate，spec 范围聚焦（4 个 agent + 1 个新模块），无第 4 项目层、无非常规模式引入。

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |
