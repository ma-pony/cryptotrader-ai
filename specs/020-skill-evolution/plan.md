# 实施计划：Skill Evolution（spec 019）

**Branch**: `020-skill-evolution` | **Date**: 2026-05-09 | **Spec**: [spec.md](spec.md)
**Input**: 来自 `specs/020-skill-evolution/spec.md`

## Summary

完成 trilogy 第 3 段的 Skill 子域：基于 spec 016 D-DS-01 + D-RT-01 + D-MW-01 决策落地 EvolvingSkillProvider。3 块整合：(1) Skill schema 升级（6 字段）+ 5 现有 SKILL.md 数据迁移；(2) D-RT-01 两层检索算法（regime_tags 预过滤 + IDF 加权，不含 embedding）；(3) load_skill_tool factory 改造 + propose_new_skill LLM 推断 + 前端 /memory 加 SkillsGrid section。

技术路径：直接删旧不留 fallback。4 commit 单 PR：C1 迁移 + schema / C2 算法层（IDF + LLM 推断模块）/ C3 Provider + 集成 atomic / C4 API + 前端 + E2E。

## Technical Context

**Language/Version**: Python 3.12+（项目既定）/ TypeScript 5.9（Web）
**Primary Dependencies**: spec 014 既有 `agents/skills/{schema,loader,tool}.py` + `learning/skill_proposal.py` + spec 017a `SkillProvider` Protocol + FastAPI（既有）+ React 19.2 + React Query
**Storage**: 文件系统 — `agent_skills/<name>/SKILL.md`（spec 014 既有，本 spec 加 6 frontmatter 字段）+ `agent_skills/<name>/SKILL.md.draft`（spec 014 既有 propose_new_skill 写入路径）
**Testing**: pytest + pytest-asyncio + vitest（web）；新增 9 测试文件
**Target Platform**: Linux server（生产）+ macOS（开发）
**Project Type**: Single project（python）+ frontend（既有 web/）
**Performance Goals**: EvolvingSkillProvider.get_available_skills < 50ms / cycle / agent；IDF 计算在 5 skill + 50 字段 snapshot 上 < 30ms
**Constraints**:
- 不引入新 runtime 依赖（IDF 用 pure Python，**不引入** sentence-transformers / sklearn / nltk）
- 不破坏 spec 017a/b/18 公开 API（PromptBuilder.build / SkillProvider Protocol 签名）
- 不修改 spec 014 既有 Maturity Literal（不加到 Skill）
- 不修改 spec 018 EvolvingMemoryProvider（同 module-level singleton 中并存）
- spec 020 推迟项（Anthropic cache / daemon / git lineage）显式 OOS
**Scale/Scope**:
- 5 个现有 SKILL.md 数据迁移（含硬编码 mapping）
- 9 新测试文件 + vitest 扩展
- 修改文件 ~12 个
- 总 diff 估算 ~2800 行
- 4 commit 单 PR

## Constitution Check

`.specify/memory/constitution.md` 为模板占位符。已对齐 CLAUDE.md：
- ✓ Markdown 简体中文
- ✓ 不修改 CLAUDE.md
- ✓ 文件改动范围（spec 目录 + scripts/ + src/cryptotrader/{agents,learning}/ + src/api/ + web/ + tests/）
- ✓ 不引入新依赖
- ✓ 不替换 spec 014/15/17a/17b/18 任何 invariant

**Constitution Check 状态**：PASS

## Project Structure

### Documentation (this feature)

```text
specs/020-skill-evolution/
├── plan.md              # 本文件
├── spec.md              # 已存在
├── REVIEW-SPEC.md       # stage 2 输出
├── checklists/requirements.md
├── research.md          # Phase 0 输出
├── data-model.md        # Phase 1 输出
├── quickstart.md        # Phase 1 输出
├── contracts/
│   ├── evolving-skill-provider.md
│   └── skill-api-routes.md
├── tasks.md             # Phase 2 输出
└── REVIEW-PLAN.md       # Stage 5 输出
```

### Source Code (repository root)

```text
scripts/
└── migrate_018_to_019.py            # NEW — C1（含 5 skill 硬编码 mapping）

src/cryptotrader/agents/skills/
├── schema.py                        # MODIFY — C1（Skill 加 6 字段 default）
├── loader.py                        # 不动（spec 014/17b 既有）
└── tool.py                          # MODIFY — C3（_make_load_skill_tool 加 provider 参数）

src/cryptotrader/agents/
└── prompt_builder.py                # MODIFY — C3（删 DefaultSkillProvider class）

src/cryptotrader/learning/evolution/
├── idf.py                           # NEW — C2（pure Python IDF 算法）
├── skill_metadata_inference.py      # NEW — C2（LLM 推断 prompt + parse）
└── skill_provider.py                # NEW — C3（EvolvingSkillProvider）

src/cryptotrader/learning/
└── skill_proposal.py                # MODIFY — C3（propose_new_skill 加 LLM 推断 metadata）

src/cryptotrader/nodes/
└── agents.py                        # MODIFY — C3（_get_or_build_pb 切到 EvolvingSkillProvider；load_skill_tool wire）

src/api/routes/
└── memory.py                        # MODIFY — C4（加 4 个 skills endpoints）

web/src/pages/memory/
├── MemoryPage.tsx                   # MODIFY — C4（加 SkillsGrid section）
├── components/SkillsGrid.tsx        # NEW — C4
└── queries.ts                       # MODIFY — C4（加 4 hooks）

web/src/locales/zh-CN/memory.json    # MODIFY — C4（加 Skills section 文案）
web/src/locales/en-US/memory.json    # MODIFY — C4

tests/
├── fixtures/skills_old_format/      # NEW — C1
├── test_migrate_018_to_019.py       # NEW — C1
├── test_idf.py                      # NEW — C2
├── test_skill_metadata_inference.py # NEW — C2
├── test_evolving_skill_provider.py  # NEW — C3
├── test_load_skill_tool.py          # NEW — C3
├── test_skill_proposal_metadata_inference.py  # NEW — C3
├── test_api_memory_skills.py        # NEW — C4
├── test_e2e_skill_evolution.py      # NEW — C4
└── web/test_memory_page.tsx         # MODIFY — C4（加 4 用例）
```

**Structure Decision**: Single project + existing web/ frontend。沿用 spec 014/17/18 既有目录结构；新增 `src/cryptotrader/learning/evolution/{idf,skill_metadata_inference,skill_provider}.py` 在 spec 018 既有 `evolution/` 子包中。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | — | — |

无复杂度违反项。
