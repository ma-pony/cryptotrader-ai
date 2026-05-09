# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联前置研究**：[spec 016](../016-research-skill-evolution-prior-art/) / [spec 017a](../017-agent-prompt-externalization/)
**关联 brainstorm**：[brainstorm/03-spec-017b-prompt-builder-integration.md](../../brainstorm/03-spec-017b-prompt-builder-integration.md)
**Date**: 2026-05-08

## 概述

本 spec 的 6 项关键设计决策已在 brainstorm 阶段（2026-05-08）系统讨论。本文档不重复决策推导，仅记录最终决定 + 实施细节研究。

## Technical Context 中无 NEEDS CLARIFICATION 项

Brainstorm 6 项决策 + 4 项 spot-check 已消除全部 ambiguity。

## 6 项关键决策

| # | 决策 | 来源 |
|---|---|---|
| Q1 整合力度 | C 激进 — 删 SkillsInjectionMiddleware / AgentRegistry.prompt_template / ANALYSIS_FRAMEWORK 常量 / role_description 字段 | 用户 17a 决策延续："直接删旧不留 fallback" |
| Skill-Q | A verbatim — DefaultSkillProvider 用 scope filter，PromptBuilder 渲染完整 SKILL.md body | 行为零回归优先 |
| Q2 ANALYSIS_FRAMEWORK 落点 | B 拆两段 — discipline → system_prompt × 4，JSON schema → output_schema × 4 | output_schema 在 017a 强制保留有架构 protection |
| Q3 _build_prompt 处理 | D 独立 renderer — 新建 `snapshot_renderer.py` 模块 | crypto 领域逻辑隔离，PromptBuilder 保持 generic |
| Q4 backtest_mode | A 保留 — ToolAgent 二分支不变 | spec 014 / 015 backtest 测试是 regression gate |
| Q5 PromptBuilder 注入 | B nodes/agents.py module-level singleton | config.py 是基础模块，反向依赖会破坏依赖图 |
| Q6 迁移颗粒度 | B 三 commit 单 PR | atomic 切换无 fallback，但 review 友好 |

## 4 项 spot-check 结果（2026-05-08）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | DefaultMemoryProvider patterns/cases 路径 | ❌ 不匹配 spec 014 实际：017a 期望 `<id>/cases.jsonl`，实际是 `cases/<cycle_id>.md`（全局）。决策 Option-2：本 spec 不修，experience: str 参数走直接路径，DefaultMemoryProvider 修推迟 spec 018 |
| 2 | prompt_template in config/ | ✓ 无匹配，安全删除 |
| 3 | graph.py agent 实例化 | ✓ 无匹配，FR-Y35 NOOP |
| 4 | regime_tags 流转 | ⚠ nodes/agents.py:42-56 有传参；隐含在 Q5 决策中清理 |

## 实施细节决策

### Decision 1：experience 参数走直接路径

**Decision**：FR-Y6b 在 PromptBuilder.build() 加 `experience: str = ""` 参数；非空时跳过 MemoryProvider，直接作为 `recent_memory` section 内容。

**Rationale**：
- spec 014 `verbal_reinforcement` 节点输出 experience 字符串到 state，是现有的 verbal reinforcement 流转路径
- 017a DefaultMemoryProvider 路径错误（cases.jsonl 不存在），不能依赖
- 保持向后兼容：experience 参数已是 BaseAgent.analyze() 现有签名
- spec 018 重写 EvolvingMemoryProvider 时可以替代此路径

**Alternatives considered**：
- 在本 spec 修复 DefaultMemoryProvider 路径（拒绝：扩大范围 ~150 行 diff）
- 完全删除 experience 参数（拒绝：spec 014 verbal_reinforcement 流转会断）

### Decision 2：DefaultSkillProvider scope filter 修正

**Decision**：FR-Y28 把 017a 的 `agent_id in skill.tags` 过滤改为 `discover_skills_for_agent(agent_id)`（spec 014 既有函数）。

**Rationale**：
- spec 014 SKILL.md frontmatter 用 `scope: shared` 或 `scope: agent:<id>` 字段（不是 `tags`）
- 017a 的 tags 过滤是错误假设（实际 SKILL.md frontmatter 没必填 `tags` 字段）
- 复用 spec 014 既有函数，不重复实现 scope 解析

**Alternatives considered**：
- 修改 017a Skill dataclass 加 `tags` 字段（拒绝：spec 014 SKILL.md schema 不动）
- 改 spec 014 SKILL.md schema 加 tags（拒绝：超出 spec 范围）

### Decision 3：ANALYSIS_FRAMEWORK 拆段策略

**Decision**：FR-Y2 / FR-Y3 把 ANALYSIS_FRAMEWORK 拆为 discipline 段（30 行）+ JSON schema 段（5 行），分别 verbatim 复制到 4 个 config 的 system_prompt / output_schema。

**Rationale**：
- 不引入 include 机制（保持 017a config schema 稳定）
- output_schema 段在 017a TokenBudgetEnforcer 中强制保留（不可丢/降）
- 4 倍重复成本可控（~140 行 markdown 总量）
- 单点修改 ANALYSIS_FRAMEWORK 时改 4 处 — 但 ANALYSIS_FRAMEWORK 进化频率低（spec 014 / 015 落地后未变）

**Alternatives considered**：
- 全部复制到 system_prompt 单段（拒绝：output_schema 段失去 architectural protection）
- `_shared/analysis_framework.md` + include 机制（拒绝：违反 017a 稳定契约）

### Decision 4：load_skill_tool 显式注入

**Decision**：FR-Y34 把 spec 014 既有 `load_skill_tool` 通过 nodes/agents.py 显式 import 并加到 ToolAgent.tools；不依赖 SkillsInjectionMiddleware.tools 类变量。

**Rationale**：
- SkillsInjectionMiddleware 删除后，其类变量 `tools = [load_skill_tool]` 也消失
- ToolAgent 的 LangChain create_agent 循环仍需要 load_skill_tool（让 LLM 按需 load skill 详情）
- 显式 import + 注入比隐式类变量更清晰

**Alternatives considered**：
- 完全删除 load_skill_tool（拒绝：丢失现有 tool-call 能力，且 spec 018 才是合适落点）
- 保留 SkillsInjectionMiddleware 仅为 tools 类变量（拒绝：违反 Q1 C 决策）

### Decision 5：snapshot_renderer 模块边界

**Decision**：FR-Y11/Y12/Y13 `render_crypto_snapshot(snapshot, experience="")` 单纯字符串渲染，不计算 indicators；TechAgent.compute_indicators 计算逻辑仍在 tech.py，agent.analyze() 内 merge 进 snapshot dict 后传给 PromptBuilder。

**Rationale**：
- 单一职责：renderer 不涉及 pandas / 计算
- TechAgent 依赖 pandas 不变（spec 014 / 015 已落地）
- spec 018 可在 PromptBuilder 加 `snapshot_renderer` 构造参数注入新 renderer，无破坏

**Alternatives considered**：
- snapshot_renderer 内部调 `compute_indicators`（拒绝：模块依赖图反向）
- TechAgent override 在 PromptBuilder 注册 indicator hook（拒绝：复杂化）

### Decision 6：3 commit 序列内部顺序

**Decision**：C2 commit 内文件改动顺序无关（atomic）；但建议先改 base.py（核心定义）→ 4 agent 类（实现）→ config.py（注入）→ nodes/agents.py（wiring）→ snapshot_renderer 接入（PromptBuilder 内）→ 删 middleware → 4 agent test 更新。

**Rationale**：
- 实施 subagent 应按"definition → implementation → wiring → tests"自然顺序
- atomic commit 保证 git history 干净，bisect 颗粒度不被破坏
- 防止 subagent drift（如 spec 017a 实施时误改 memory.py）：每段改动后立即跑相关测试，发现回归立即停

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。
