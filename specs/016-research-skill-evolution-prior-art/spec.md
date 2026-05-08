# Feature Specification: Skill / Memory Evolution Prior-Art Research

**Feature Branch**: `016-research-skill-evolution-prior-art`
**Created**: 2026-05-08
**Status**: Draft
**Input**: User description: 系统性研究 8 个开源 LLM agent 项目（SkillClaw / MetaClaw / OpenClaw-RL / Hermes Agent Self-Evolution / EvoSkill / EvoSkills / skill-evolution / autoresearch），提取它们的 skill 进化、memory 固化、动态检索、prompt 组装等设计模式，为本项目的 spec 017（prompt 外置）和 spec 018（skill 进化 v2）提供决策依据。本 spec 仅产出研究文档，不修改运行时代码。

## Background

本项目的 skill / memory 架构（spec 014）只在生产运行了 1-2 天，初步观察暴露了几个待优化点：硬编码 prompt 难维护、skill 命中率偏低、memory 进化逻辑未实战验证。8 个外部开源项目针对同类问题给出了不同方案。本 spec 系统性吸收这些项目的设计经验，研究采取**开放心态**：当前的 5 层防过拟合算法、PnL maturity FSM、单写者反射模型、SKILL.md 文件协议都不预设为"必须保留"——若研究证明有更优做法，018 spec 可基于本研究的结论替换任何当前组件。

研究分两段交付：

- **Phase 1 速读** —— 8 项目的"prompt 组装 + memory 拼接"机制速读，解锁 spec 017 的 brainstorm
- **Phase 2 全面深读** —— 8 角度（进化算法 / Skill 数据结构 / 检索机制 / Memory↔Skill 关系 / Prompt 组装 / Evaluation / Agent↔Skill 边界 / 工程实现）对所有 8 项目完整覆盖，解锁 spec 018 的 brainstorm

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Phase 1 解锁 spec 017 brainstorm (Priority: P1)

作为本项目的架构师，在启动 spec 017（agent-prompt-externalization）的 brainstorm 之前，我需要先看到 8 个外部项目如何把 agent role / skills / memory / context 组装成最终 LLM 输入，避免 spec 017 提交后才发现更优做法。

**Why this priority**: spec 017 是机械重构（ROLE 字符串外置 + memory 拼接），但"组装结构"这个细节直接决定文件格式与可扩展性。Phase 1 是 spec 017 的硬前置条件——没有这部分研究，spec 017 brainstorm 启动就有"先做完再发现要改"的风险。

**Independent Test**: 仅凭 Phase 1 的产物（synthesis.md 中"Prompt Assembly"主题章节 + decisions.md 中相关决策），架构师能够独立做出 spec 017 的核心设计选择（例如：YAML / TOML / Markdown 哪种格式、字符串拼接 / Jinja 模板 / 结构化 schema 哪种组装方式、static / dynamic 何时刷新）。

**Acceptance Scenarios**:

1. **Given** 8 个项目的 prompt-assembly 机制已记录, **When** 架构师阅读 synthesis.md 的"Prompt Assembly"章节, **Then** 应该能在 30 分钟内列出 3-5 个具体设计选项及各自的取舍点
2. **Given** Phase 1 输出已就绪, **When** 用户确认 SC-R6 满足, **Then** spec 017 brainstorm 立即可启动（无需等 Phase 2）

---

### User Story 2 - Phase 2 解锁 spec 018 brainstorm (Priority: P1)

作为本项目的架构师，在启动 spec 018（skill-evolution-v2）的 brainstorm 之前，我需要看到 8 个项目在所有 8 个研究角度的设计权衡矩阵——尤其是进化算法 / 检索机制 / evaluation 这三块，避免"凭直觉重新发明"。

**Why this priority**: spec 018 是本次重构的重头戏，可能替换 spec 014 的核心组件（5 层防过拟合、PnL maturity FSM、单写者反射模型）。决策需要建立在对外部经验的系统性理解之上，否则替换后果不可控。

**Independent Test**: 仅凭 Phase 2 的产物（comparison-matrix.md + 8 份 projects/<name>.md + synthesis.md 全部 8 章节 + decisions.md），架构师能在 4 小时内完成 spec 018 的初步 brainstorm 起草，且每个核心设计决策都能引用至少一个外部项目作为证据。

**Acceptance Scenarios**:

1. **Given** Phase 2 全部产物已就绪, **When** 架构师查询"如何度量 skill 命中率", **Then** comparison-matrix.md 中"Evaluation"列对 8 个项目都有明确答案
2. **Given** synthesis.md 的 ≥10 条具体建议已写完, **When** 用户与架构师 review, **Then** 每条建议都能引用至少 1 个项目 + 至少 1 个本项目组件 + 1 个明确行动方向
3. **Given** SC-R7 的所有判定满足, **When** 用户确认完成, **Then** spec 018 brainstorm 可启动

---

### User Story 3 - Reviewer 审查每条研究建议 (Priority: P2)

作为 reviewer，需要看到每条研究建议的完整链路（来源项目 + 具体算法 + 我们的对应组件 + 行动方向），才能判断是否采纳。

**Why this priority**: 研究产物会驱动 017/018 的设计，但研究本身可能有偏差。Reviewer 必须能从单条建议追溯到源码引用，独立验证而不是被动接受。

**Independent Test**: 随机抽取 synthesis.md 中 3 条建议，每条都能在 5 分钟内通过引用文件路径跳到外部项目源码并验证建议成立。

**Acceptance Scenarios**:

1. **Given** synthesis.md 第 5 条建议引用 SkillClaw 的某算法, **When** reviewer 点击文件路径, **Then** 能直接跳到对应源码位置
2. **Given** 某条建议被 reject, **When** reviewer 查 decisions.md, **Then** 能看到完整 ADR 段落（Status / Context / Decision / Consequences / Reject 理由）

---

### User Story 4 - Maintainer 验证 license 合规 (Priority: P2)

作为后续维护者，需要 license 信息齐全，避免后续从 016 借鉴的代码引入与本项目 license 冲突的开源条款（例如不慎引入 GPL 代码）。

**Why this priority**: 一旦 spec 018 决定 fork 某项目的代码片段，license 兼容性是合规硬约束。研究阶段记录 license 比落实代码后再回查代价低 100x。

**Independent Test**: 8 个 projects/<name>.md 的 frontmatter 都含 `license:` 字段；decisions.md 中所有"建议 fork"的条目都有显式的 license 兼容性结论。

**Acceptance Scenarios**:

1. **Given** 某项目 license 是 GPL-3.0, **When** decisions.md 中存在引用其代码的建议, **Then** 该建议必须额外注明"本项目 license 与 GPL-3.0 兼容性问题"或被显式 reject

---

### Edge Cases

- **某 GitHub repo 被删除或转私有**：研究 MUST 在 projects/<name>.md 中明确记录"仓库不可访问 + 最后访问日期"，并尝试通过 web archive 恢复。若仍不可得，该项目可在 8 个里面被替换为相邻项目（需用户审批）。
- **license 文件缺失**：projects/<name>.md 的 frontmatter `license:` 字段写 `Unknown`，并在文档主体记录调研结果（GitHub 显示？README 注明？）。
- **Tier 1 项目代码极差或弃维**：可降级为 Tier 2，但 decisions.md MUST 记录降级理由。
- **某角度对某项目完全 not applicable**（例如 autoresearch 没有 skill 概念）：comparison-matrix.md 该格写"N/A — <一句解释>"，不留空。
- **Phase 1 与 Phase 2 之间发现新的相关项目**：可在 Phase 2 范围内选择性添加；不强制研究第 9 个项目，但 decisions.md MUST 记录"未研究的相关项目"以便未来 spec 引用。
- **研究 LLM 上下文窗口不足以读完一个 Tier 1 项目**：分多次读取，每次记录"本次覆盖了哪些文件"以避免遗漏。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-R1**: 研究 MUST 对以下 8 个 GitHub 项目分别完成：SkillClaw（AMAP-ML）、MetaClaw（aiming-lab）、OpenClaw-RL（Gen-Verse）、Hermes Agent Self-Evolution（NousResearch）、EvoSkill（sentient-agi）、EvoSkills（EvoScientist）、skill-evolution（hao-cyber）、autoresearch（uditgoenka）。

- **FR-R2**: 研究 MUST 覆盖 8 个角度，对每个项目按角度分别记录：进化算法 / Skill 数据结构 / 检索机制 / Memory↔Skill 关系 / Prompt 组装 / Evaluation / Agent↔Skill 边界 / 工程实现。

- **FR-R3**: 项目优先级分 3 档（仅决定阅读深度，不跑测试 / 复现）：
  - **Tier 1**（深读）：SkillClaw、Hermes Agent Self-Evolution、OpenClaw-RL —— 主代码路径全读，关键算法画流程图，所有 8 角度独立小节
  - **Tier 2**（细读）：MetaClaw、EvoSkill、EvoSkills、skill-evolution —— 关键源码理解 + 8 角度填表（不要求每角度独立小节）
  - **Tier 3**（扫读）：autoresearch —— README + 代表性 source file + 8 角度填表中的 1-2 句话

- **FR-R4**: 研究方法 MUST 包含：(a) WebFetch repo README + docs + 关键源文件；(b) 完整 git clone 到本地 → 代码深读（含测试代码作为算法注解，但不实际执行）；(c) WebSearch 寻找 paper / blog / 官方介绍文章作补充。**MUST NOT 跑任何 demo / 测试 / 复现实验**，所有理解通过源码静态分析得出。

- **FR-R5**: 研究产物 MUST 写到 `specs/016-research-skill-evolution-prior-art/research/` 目录，分 4 类文档：
  - `projects/<name>.md`（8 个）—— 每个项目独立文档：架构总览 / 关键算法 / 代码片段引用 / 取舍点 / 借鉴建议
  - `comparison-matrix.md`（1 个）—— 8 项目 × 8 角度矩阵；每格一句话 + 引用文件路径
  - `synthesis.md`（1 个）—— 按主题/角度横向对比，给出针对本项目的具体建议（≥ 10 条）
  - `decisions.md`（1 个）—— 决策日志：每个被采纳 / reject / defer 的方案附 ADR

- **FR-R6**: 研究分两段交付：
  - **Phase 1**（解锁 spec 017）：8 项目的"prompt 组装 + memory 拼接"机制速读，至少占 `synthesis.md` 中 prompt-assembly 主题的完整章节 + decisions.md 中相关决策。Phase 1 不要求其他 7 个角度完整。
  - **Phase 2**（解锁 spec 018）：所有 FR-R2 列出的 8 角度对所有 8 项目完整覆盖。

- **FR-R7**: 每个 `projects/<name>.md` MUST 在 frontmatter 含 `license:` 字段（值取自该项目 LICENSE 文件，如 `MIT`、`Apache-2.0`、`GPL-3.0`、`Custom (see file)`、`Unknown`）。`decisions.md` 中如果建议"fork 该项目的代码片段" MUST 验证 license 兼容性（与本项目 license 不冲突）。

- **FR-R8**: 研究 MUST NOT 修改 `src/`、`config/`、`agent_skills/`、`agent_memory/` 任何文件；MUST NOT 引入新 runtime 依赖；MUST NOT 替换 spec 014 的任何 FR。所有产物限定在 `specs/016-research-skill-evolution-prior-art/` 目录内。

- **FR-R9**: 研究 MUST NOT 在产物文档中复制粘贴 > 50 行的开源代码——超过 50 行的引用 MUST 用文件路径 + line range 链接形式。

### Key Entities

- **Research Project**: 一个被研究的外部 GitHub 项目；属性包括 `name`、`url`、`license`、`tier`、`last_accessed_at`。
- **Research Angle**: 8 个固定研究角度之一（进化算法 / Skill 数据结构 / 检索机制 / Memory↔Skill 关系 / Prompt 组装 / Evaluation / Agent↔Skill 边界 / 工程实现）。
- **Project Document**: `projects/<name>.md` 文件；frontmatter + 按角度组织的章节。
- **Comparison Matrix Cell**: 8 项目 × 8 角度矩阵中的一格；每格至少含 1 句话定性 + 1 个文件路径或 commit 引用。
- **Synthesis Recommendation**: synthesis.md 中针对本项目的一条具体建议；至少含来源、影响组件、行动方向。
- **Architecture Decision Record (ADR)**: decisions.md 中记录的方案条目；至少含 Status（Accepted / Rejected / Deferred）+ 简短理由。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-R1**: `specs/016-research-skill-evolution-prior-art/research/projects/` 包含 8 个 `<name>.md` 文件，分别对应 8 个项目。每个文件深度由 FR-R3 的 Tier 决定，没有强制行数下限——但内容应当能让读者凭文档 alone 理解该项目的 skill / memory 设计（无需重读源码）。

- **SC-R2**: `comparison-matrix.md` 是 8 × 8 矩阵（8 项目 × 8 角度）。每格非空（"N/A — <理由>" 也算填了），格内至少包含一句话定性 + 一个文件路径或 commit 引用，方便 reviewer 快速跳到源码。

- **SC-R3**: `synthesis.md` 给出 ≥ 10 条针对本项目的具体建议。建议格式自由，但每条至少包含：(a) 借鉴的来源（项目名 + 大致位置）；(b) 在本项目中影响什么（哪个组件 / 哪个 spec FR）；(c) 大致行动方向（采纳 / 替换 / 调整 / 推迟）。过于含糊的建议（"考虑使用 RL"无具体落点）不计入这 10 条。

- **SC-R4**: `decisions.md` 用 ADR 风格记录被考虑的方案，每个方案至少含 Status + 简短理由。没有最低数量——但若与 SC-R3 严重不匹配（例如建议 10 条但 decisions 只有 1 条），视作研究深度不足。

- **SC-R5**: 每个 `projects/<name>.md` 头部 frontmatter MUST 含 `license:` 字段。

- **SC-R6**: **Phase 1 完成判定（解锁 spec 017）**：(a) `synthesis.md` 中"Prompt Assembly"主题覆盖 8 项目（不要求每个独立长段落，但每个都要被提到）；(b) `decisions.md` 含 prompt 组装相关决策若干条；(c) 用户在 brainstorm 流程中确认 Phase 1 已就绪。

- **SC-R7**: **Phase 2 完成判定（解锁 spec 018）**：(a) SC-R1 ~ SC-R5 满足；(b) `synthesis.md` 8 角度都有覆盖；(c) 通过 `spex:review-spec` 评审无 P0 / P1 issues。

## Assumptions

- 8 个 GitHub repo 在研究期间保持可访问（若某个被删/转私有，按 Edge Cases 第 1 条处理）。
- 用户接受研究产物全部用 Markdown 形式（不强制结构化数据库或图谱表达）。
- 研究 agent 拥有网络访问 GitHub 与执行 git clone 的能力。
- 本机磁盘空间能容纳 8 个 repo 的 git history（粗估 < 5GB 总）。
- 本项目自身 license（待确认；研究阶段假设是 permissive 类如 MIT/Apache，若实际是 GPL 则部分 fork 建议会失效）。

## Dependencies

**Upstream（输入）**: 无。016 是 leaf research，不依赖任何其他 spec 完成，可独立启动。

**Downstream（被本 spec 解锁）**:

- **Spec 017 (agent-prompt-externalization)** 的 brainstorm 启动 MUST 在 016 Phase 1 完成（SC-R6 满足）之后。
- **Spec 018 (skill-evolution-v2)** 的 brainstorm 启动 MUST 在 016 Phase 2 完成（SC-R7 满足）之后。

**External tooling assumed available**: git；网络访问 GitHub / web；本机磁盘空间。

## Out of Scope

- 运行时代码修改（`src/`、`config/`、`agent_skills/`、`agent_memory/` 任何文件）。
- 新 runtime 依赖引入（不为研究而 `pip install` 新包；研究产物的代码引用以路径 + 行号形式呈现）。
- 自动化抓取流水线 / "auto-summarize repo" 类脚本（研究依赖人工 + LLM 辅助阅读）。
- 跑测试 / 复现 / demo（FR-R4 已禁止）。
- License 修改建议（仅记录原项目 license，不评估 / 改动本项目自身 license）。
- 集成 PR / 设计草案（这些是 spec 017 / 018 的产物）。
- N1 long-signal asymmetry / N6 drawdown_pct discrepancy / N8 regime detection 等 spec 015 标记为 OOS 的研究主题（除非某项目设计明显与之相关）。
- LLM 模型选型 benchmark（关心 skill / memory 架构，不关心模型选型）。

## Reversibility

016 完全可逆：spec 文档可独立删除，删除后无运行时影响。8 个 repo 的本地 clone 可在 Phase 2 完成后归档 / 删除（研究产物文档已包含必要的代码引用）。
