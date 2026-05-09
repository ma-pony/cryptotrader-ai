# Spec 评审：Skill Evolution（spec 019）

**关联 spec**：[specs/020-skill-evolution/spec.md](spec.md)
**评审时间**：2026-05-09
**评审人**：Claude（spex:review-spec，ship pipeline stage 2）

## 总体评估

**状态**：✅ SOUND

**摘要**：spec 由完整 brainstorm 7 决策 + 4 项 spot-check 后生成；32 条 FR + 19 条 SC + 6 user stories 全部具体可测；Out of Scope 显式分"移至 spec 020" / "本 spec 不动"两类，trilogy 边界清晰。无 P0 / P1 issues，2 条 P3 advisory。

## 完整性：5/5

- ✓ 强制 section 全部完成
- ✓ 推荐 section 完整（Edge Cases / Dependencies / Out of Scope / Reversibility / Implementation Outline）
- ✓ 32 条 FR-W 覆盖 8 子模块（Schema/Migration / Provider / load_skill_tool / skill_proposal / API / Frontend / Telemetry / Migration Tooling）
- ✓ 10 条 Edge Cases 显式列出（含 IDF 失败 / regime_tags=[] 视为 match all / .draft 写入语义等）

## 清晰度：5/5

- ✓ 无 [NEEDS CLARIFICATION] markers
- ✓ 全部 MUST / MUST NOT
- ✓ 接口签名（PatternRecord vs Skill 区分 / Maturity 不加到 Skill / propose_new_skill 写 .draft）明确
- ✓ Commit 序列（C1/C2/C3/C4）边界清楚

## 可实现性：5/5

- ✓ 4 commit 单 PR 计划具体（每 commit 文件范围 + diff 估算）
- ✓ FR-W3 含 5 skill 完整硬编码 mapping（无需 implement 时再次 LLM 推断）
- ✓ Upstream / Downstream 依赖完整（spec 017a/b/14/15/10/16/18）
- ✓ Out of Scope 显式排除"sentence-transformers embedding"（违反"不引入新依赖"约束）

## 可测试性：5/5

- ✓ SC-W1..W19 含具体阈值（"≥ 8 用例 PASS" / "≥ 12 用例 PASS" / "返回空" / "通过基线 ≥ 2254"）
- ✓ User Story Acceptance Scenarios 全部 Given/When/Then 格式
- ✓ E2E 测试条件具体（含 D-RT-01 4 telemetry 字段 + propose_new_skill 7 telemetry 字段）

## Constitution 对齐

无 `.specify/memory/constitution.md` 项目原则定义。按 CLAUDE.md：
- ✓ Markdown 中文
- ✓ 不修改 CLAUDE.md
- ✓ 仅在 `specs/020-*/` + 既有 `src/` / `web/` 范围内创建/修改文件
- ✓ 不引入新依赖（IDF 用 pure Python）
- ✓ 与 spec 014/15/17a/17b/18 既有架构契约兼容（Skill dataclass 加 default 字段；SkillProvider Protocol 不变；EvolvingMemoryProvider 同 singleton 并存）

## 跨 Spec 一致性

- ✓ 与 spec 017a 决策对齐：保留 SkillProvider Protocol；DefaultSkillProvider class 删除
- ✓ 与 spec 017b 决策对齐：load_skill_tool 改造（不再直接读文件，走 Provider）
- ✓ 与 spec 018 决策对齐：EvolvingMemoryProvider 不动；同 module-level singleton 中并存 EvolvingSkillProvider
- ✓ 与 spec 016 决策对齐：D-DS-01 子集采纳（Skill 适配，不加 pattern-only 字段如 maturity FSM/pnl_track）；D-RT-01 子集（无 embedding）；D-MW-01 全采纳（importance/access_count/last_accessed_at）
- ✓ 与 spec 014 既有架构对齐：propose_new_skill 写 .draft 语义保留；discover_skills_for_agent 复用作 scope 第一层；load_skill_tool factory 模式适配

## 推荐改进

### Critical（必须修复）

- 无

### Important（应当修复）

- 无

### Optional（建议）

1. **SC-W6 graph 节点位置不需要验证**：spec 019 不动 graph.py（Q4 决策"仅 retrieval"），SC-W6 已隐含但可加显式 SC：`grep -n "evaluate_skills\|skill_evolution" src/cryptotrader/graph.py` 返回空（确认未误加节点）。**严重程度**：P3
2. **FR-W26 i18n 文件路径不确定**：spec 现写"`web/src/i18n/{zh-CN,en-US}.ts` 或对应 i18n 文件（spec 018 实际可能放 `web/src/locales/zh-CN/memory.json`）"含两种可能，落地前需 grep 确认。**严重程度**：P3。implement subagent 可在改 i18n 时自查 spec 018 落地后实际路径

## 结论

spec 019（目录 020）准备就绪，可进入 plan 阶段。

**是否就绪**：是
