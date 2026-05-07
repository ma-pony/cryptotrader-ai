# Spec Review v3 (Fresh)

**Reviewer**: spex:review-spec
**Date**: 2026-05-06
**Subject**: [spec.md](./spec.md) (commit 1c5da42)
**Verdict**: 🟡 **APPROVE WITH 6 CRITICAL FIXES + 5 MINOR**（架构稳，但 v2→v3 升级遗留多处不一致）

---

## Soundness（设计合理性）

### ✅ 通过

- 双层架构（memory gitignored + skills git）边界清晰
- `scope: shared | agent:<id>` 替代硬编码 mapping，正确
- 4 层防过拟合（无时间衰减）保留交易场景 know-how
- Anthropic Skills 协议合规：name + description 必填，扩展字段（scope / manually_edited）由协议 additionalProperties 允许
- Out of Scope 段防止 scope creep

### ⚠️ Critical 问题（必须 fix）

#### CRITICAL-1：Status header 仍是 v2

第 5 行：`**Status**: Draft (v2 — 双层架构修订)`

实际已经经过 v3 dynamic discovery 重写。**应改为 `Draft (v3 — 动态发现 + scope 字段)`**。

#### CRITICAL-2：多处"恒为 5"残留与 FR-004 矛盾

| 位置 | 原文 | 与 FR-004 的矛盾 |
|---|---|---|
| 第 12 行 User Story 1 | "高层 `agent_skills/`（git 跟踪，**仅** 5 个 SKILL.md）" | FR-004 说"**至少**含 5 个" |
| 第 53 行 User Story 3 标题 | "Skills 层：5 个高层能力包" | 应为"≥5" |
| 第 60 行 Independent Test 1 | "`ls agent_skills/` 列出**恰好** 5 个目录" | 用户加新 skill 后此 test fail |
| 第 68 行 Acceptance Scenario 1 | "显示 5 个目录且总文件数 ≤ 6" | 同上 |
| 第 169 行 Section header | "#### Skills 层（5 个 SKILL.md）" | 应为"≥5，动态发现" |
| 第 212 行 Key Entities | "Skill：高层能力包（5 个之一）" | 应为"≥5" |
| 第 240 行 Assumption | "`agent_skills/` 5 个 SKILL.md 总规模 ≤ 50KB" | 应限定为 initial state |

**修复**：全部改为 "initial 5，运行时可增长"或"≥5"。

#### CRITICAL-3：FR-006 case 文件粒度未定义（per-agent vs per-cycle）

第 155 行：

> 每个 trading cycle 完成时（含 journal_trade 节点）MUST 写入 `agent_memory/<agent>/cases/<cycle_id>.md`（**每个 agent 独立一份**）

矛盾点：
- "**每个 agent 独立一份**"暗示 1 cycle × 4 agents = 4 个 case 文件
- 但 verdict 是跨 4 agent 综合产物，每份 case 包含的 verdict_reasoning 是相同的吗？
- 蒸馏时如果 4 份合 1 个 cycle，会重复计算还是去重？

**需要明确**：
- 选项 A：**每 cycle 4 个 case 文件**（per-agent），各含该 agent 的 analysis + 全 cycle 的 verdict + 该 agent 引用的 applied patterns
- 选项 B：**每 cycle 1 个 case 文件**（per-cycle），含 4 个 agent analyses + verdict + 全部 applied patterns
- 选项 C：**两层**：`cases/<cycle_id>.md`（cycle 级）+ `agents/<agent>/contributions/<cycle_id>.md`（agent 级）

我推荐 **选项 B**（per-cycle 单文件），更简单 + 反思时容易关联。

#### CRITICAL-4：FR-013 锁机制与 research.md R4 不一致

- spec 第 167 行：`fcntl.flock 排他锁`
- research.md R4：决议为 `threading.Lock`（理由：单进程足够，fcntl 是 over-engineering）

**修复**：spec FR-013 改为 `threading.Lock`（或显式说明跨进程时升级为 fcntl.flock，但本期单进程）。

#### CRITICAL-5：跨 agent `applied:` 引用约定不明确

FR-026 + Edge Case 第 136 行：

> verdict 节点显式用 agent 前缀如 `applied: tech::X`

但 4 个 analysis agent（在自己的 prompt 里）调用 patterns 时呢？SKILL.md body 里 patterns 是 bare name（如 `funding_squeeze_long`）。agent 在 reasoning 里写 `applied: funding_squeeze_long` 还是 `applied: tech::funding_squeeze_long`？

**需要明确规则**：
- 在 self-agent context（4 个 analysis agent）：bare name 默认 = 自身 agent
- 在 cross-agent context（verdict 综合 4 agent）：要求显式 `<agent>::<name>` 形式
- 反思解析时：bare name 跨 agent 模糊 → 按发起 agent 解析；歧义跳过 + warning

**修复**：在 FR-026 增补这套规则。

#### CRITICAL-6：`agent_memory/` 目录初始化的 agent 列表硬编码

Edge Case 第 129 行：

> 系统 MUST 自动创建目录树（含 4 个 agent 子目录...

虽然 FR-004b 删除了 `SKILL_NAME_BY_AGENT`，但 4 个 agent 名（tech / chain / news / macro）在 `agent_memory/` 目录初始化时仍是硬编码。

这其实是合理的——4 agent 是项目级常量（verdict / risk gate / nodes/agents.py 等多处假设）。**需要在 spec 显式承认**：

> `VALID_AGENT_IDS = {"tech", "chain", "news", "macro"}` 是项目级常量（与 nodes/agents.py 一致），不通过 scope 字段动态化；本 feature **不修改 agent 列表**

否则读 spec 的人会问"FR-004b 不是不允许硬编码吗？"

---

## Completeness（缺漏）

### Minor 缺漏

#### MINOR-1：SC-001 baseline measurement 未在 spec 内定义

> SC-001：迁移完成后单个 trading cycle 中 4 个 agent 的 prompt token 数总和较旧系统下降 ≥ 30%

"较旧系统"——基线值哪里来？research.md R5 说"实施前测 5 次"，spec 没要求。

**建议**：spec 增补"实施前 MUST 测得旧系统 baseline 并记录在 PR 描述"，或在 plan.md 充分。

#### MINOR-2：SC-009 观察 tool 调用次数的指标系统未定义

> SC-009：`load_skill` tool 在 ≥ 14 天 reflection 运行后被实际调用（频率 > 0）

如何收集？spec 没说要加 metrics 计数器。

**建议**：FR 中增加"`load_skill` tool MUST 通过 metrics_collector 增加 calls counter"。

#### MINOR-3：propose-new 输入数据范围不明确

FR-016a 说"分析 active patterns 找出共同 regime/theme 的子集"——

- 单个 agent 的 patterns？跨 4 agents？
- shared scope 提议是从 4 agents 找跨域共性，还是从 trading-knowledge 现有内容？

**建议**：增补"`--scope agent:<id>` 仅分析该 agent；`--scope shared` 跨 4 agents"。

#### MINOR-4：middleware mtime 缓存机制：可选 / 必须

FR-019a 说"或启用 mtime 缓存时"——可选还是必须？

**建议**：本期改为"**进程内 LRU 缓存**（最大 N 个 SKILL.md 解析结果，由 mtime 失效）"，明确语义。

#### MINOR-5：AgentSkillSet entity 描述未升级

第 215 行：

> AgentSkillSet：单 agent 一次 cycle 加载的 skill 集合（含 own SKILL.md + trading-knowledge SKILL.md）

v3 后是动态 list（可能多于 2 个）。**修复**：改为"动态 list[Skill]，含所有 `scope: shared` 与 `scope: agent:<self>` 的 skills"。

---

## Implementability（可实施性）

### ✅ 全部通过

- LangChain `AgentMiddleware` + `wrap_model_call` 验证存在（plan 阶段已 inspect.signature 确认）
- 文件 IO + threading.Lock 在单进程单户场景充分
- Anthropic 协议格式与 .claude/skills/speckit-* 实例对齐

---

## Anthropic Skills 协议合规

| 协议要求 | spec 现状 | 通过？ |
|---|---|---|
| frontmatter 含 `name` | FR-014 | ✅ |
| frontmatter 含 `description` | FR-014 | ✅ |
| body 是 markdown | FR-014 | ✅ |
| 文件命名为 `SKILL.md` | FR-004 | ✅ |
| 目录结构 `<skill-name>/` | FR-004（扁平）| ✅ |
| 扩展字段允许（如 metadata, argument-hint） | additionalProperties | ✅ |
| Skills 数量"小"（high-level capabilities） | initial 5 | ✅ |

---

## Internal Consistency

### Cross-references audit

- FR-004 ↔ User Story 3 Independent Test #1：**矛盾**（CRITICAL-2）
- FR-004 ↔ Acceptance Scenario 1（line 68）：**矛盾**（CRITICAL-2）
- FR-013 ↔ research R4：**矛盾**（CRITICAL-4）
- FR-026 ↔ Edge Case L136：**部分覆盖**（CRITICAL-5 需补完）

---

## 总结：12 个 fix 项

| # | 严重度 | 改 spec 哪段 |
|---|---|---|
| CRITICAL-1 | 🔴 | Status header 改 v3 |
| CRITICAL-2 | 🔴 | 7 处"恒为 5"改"initial 5/≥5" |
| CRITICAL-3 | 🔴 | FR-006 明确 case 粒度（建议 per-cycle 单文件） |
| CRITICAL-4 | 🔴 | FR-013 锁机制改 threading.Lock |
| CRITICAL-5 | 🔴 | FR-026 补完 bare name vs `<agent>::<name>` 规则 |
| CRITICAL-6 | 🔴 | 增补"VALID_AGENT_IDS 是项目级常量"声明 |
| MINOR-1 | 🟡 | SC-001 增补 baseline measurement 要求 |
| MINOR-2 | 🟡 | FR 增补 metrics counter for load_skill |
| MINOR-3 | 🟡 | FR-016a 明确 propose-new 输入范围 |
| MINOR-4 | 🟡 | FR-019a 明确缓存机制（"进程内 LRU + mtime 失效"）|
| MINOR-5 | 🟡 | AgentSkillSet entity 描述改动态 list |
| 一致性 | 🟢 | （以上修完后再 grep 一次确认）|

## 决策

修完 6 个 critical 后 spec 进入 plan 阶段无阻塞。5 个 minor 同时修工作量小（每个 1-2 行改动）。

总改动约 **30 行 spec 文本编辑**，~10 分钟完成。
