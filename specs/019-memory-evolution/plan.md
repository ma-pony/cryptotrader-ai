# 实施计划：Memory Evolution（spec 018）

**Branch**: `019-memory-evolution` | **Date**: 2026-05-09 | **Spec**: [spec.md](spec.md)
**Input**: 来自 `specs/019-memory-evolution/spec.md`

## Summary

完成 trilogy 第 3 段的 Memory 子域：基于 spec 016 决策（D-MW-01..03 / D-EV-02..04 / D-EVAL-01）落地 EvolvingMemoryProvider。3 块整合：(1) DefaultMemoryProvider 路径 bug 修复 + 数据迁移；(2) FSM/Pareto/IVE 算法栈；(3) cases schema 扩展（Trade Execution + Causal Chain + IVE Classification）+ 前端 `/memory` 页面。

技术路径：直接删旧不留 fallback。4 commit 单 PR：C1 迁移工具 + schema 字段 / C2 算法层 / C3 Provider+nodes atomic 切换 / C4 API+前端+E2E。

## Technical Context

**Language/Version**: Python 3.12+（项目既定）/ TypeScript 5.9（Web）
**Primary Dependencies**: spec 014 既有 `learning/memory.py` IO + `agents/skills/schema.py` 既有 dataclass + spec 017a `MemoryProvider` Protocol + FastAPI（既有，新加 router）+ React 19.2 + React Query
**Storage**: 文件系统 — `agent_memory/cases/<cycle_id>.md`（spec 014 既有）+ `agent_memory/<agent>/patterns/*.md`（spec 014 既有）+ `agent_memory/<agent>/patterns/.archived/`（本 spec 新增）
**Testing**: pytest + pytest-asyncio + vitest（web）；新增 7 个测试文件 + 1 vitest + 1 API
**Target Platform**: Linux server（生产）+ macOS（开发）
**Project Type**: Single project（python）+ frontend（react/vite，既有 web/）
**Performance Goals**: EvolvingMemoryProvider.get_recent_memory < 200ms / cycle / agent；evaluate_node 全 cycle 末段 < 5s
**Constraints**:
- 不引入新 runtime 依赖
- 不破坏 spec 017a/b 公开 API
- 不修改 spec 014 既有 Maturity Literal（沿用 4 状态 + 加 archived 终态）
- 不修改 spec 014 既有 verbal_reinforcement 流转
- spec 020 推迟项显式 OOS
**Scale/Scope**:
- ~80 case 历史数据迁移
- 7 新测试文件 + 1 vitest + 1 API
- 修改文件 ~15 个
- 总 diff 估算 ~3700 行
- 4 commit 单 PR

## Constitution Check

`.specify/memory/constitution.md` 为模板占位符。本 spec 已对齐 CLAUDE.md：
- ✓ Markdown 简体中文
- ✓ 不修改 CLAUDE.md
- ✓ 文件改动范围合理（spec 目录 + scripts/ + src/cryptotrader/{agents,learning,nodes}/ + src/api/ + web/ + tests/）
- ✓ 不引入新依赖
- ✓ 不替换 spec 014/15/17a/17b 任何 invariant

**Constitution Check 状态**：PASS

## Project Structure

### Documentation (this feature)

```text
specs/019-memory-evolution/
├── plan.md              # 本文件
├── spec.md              # 已存在
├── REVIEW-SPEC.md       # stage 2 输出
├── checklists/
│   └── requirements.md  # 已存在
├── research.md          # Phase 0 输出
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── contracts/
│   ├── evolving-memory-provider.md
│   └── memory-api-routes.md
├── tasks.md             # Phase 2 输出
└── REVIEW-PLAN.md       # Stage 5 输出
```

### Source Code (repository root)

```text
scripts/
└── migrate_017_to_018.py            # NEW — C1

src/cryptotrader/agents/skills/
└── schema.py                        # MODIFY — C1（PatternRecord/CaseRecord 加新字段，Maturity 加 archived）

src/cryptotrader/agents/
└── prompt_builder.py                # MODIFY — C3（删 DefaultMemoryProvider class）

src/cryptotrader/learning/evolution/
├── __init__.py                      # NEW — C2
├── fsm.py                           # NEW — C2（evaluate_transitions）
├── pareto.py                        # NEW — C2（rank_rules）
├── ive.py                           # NEW — C2（classify_case）
└── provider.py                      # NEW — C3（EvolvingMemoryProvider）

src/cryptotrader/nodes/
├── evolution.py                     # NEW — C3（evaluate_node）
├── journal.py                       # MODIFY — C3
├── execution.py                     # MODIFY — C3
└── agents.py                        # MODIFY — C3

src/cryptotrader/
└── graph.py                         # MODIFY — C3（_build_full_graph 插入 evaluate）

src/api/
├── main.py                          # MODIFY — C4（register router）
└── routes/
    └── memory.py                    # NEW — C4（4 endpoints）

web/src/
├── pages/memory/
│   ├── MemoryPage.tsx               # NEW — C4
│   ├── components/
│   │   ├── RulesGrid.tsx            # NEW — C4
│   │   ├── CasesTimeline.tsx        # NEW — C4
│   │   ├── ArchivedRules.tsx        # NEW — C4
│   │   └── RecentTransitions.tsx    # NEW — C4
│   └── queries.ts                   # NEW — C4
├── components/layout/sidebar.tsx    # MODIFY — C4
├── App.tsx                          # MODIFY — C4
└── i18n/{zh-CN,en-US}.ts            # MODIFY — C4

tests/
├── fixtures/memory_old_format/      # NEW — C1
├── test_migrate_017_to_018.py       # NEW — C1
├── test_fsm.py                      # NEW — C2
├── test_pareto.py                   # NEW — C2
├── test_ive.py                      # NEW — C2
├── test_evolving_memory_provider.py # NEW — C3
├── test_evolution_node.py           # NEW — C3
├── test_api_memory.py               # NEW — C4
├── test_e2e_memory_evolution.py     # NEW — C4
└── web/test_memory_page.tsx         # NEW — C4（vitest）
```

**Structure Decision**: Single project + existing web/ frontend。沿用 spec 014/17a/17b 既有目录结构，新增 `src/cryptotrader/learning/evolution/` 子包 + `web/src/pages/memory/` 子目录。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |

无复杂度违反项。
