# Review Guide: Agent Prompt Externalization

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-05-08

---

## What This Spec Does

把 4 个分析 agent（tech / chain / news / macro）的"角色提示词"从 Python 代码字符串搬到独立的 Markdown 配置文件，由统一的 `PromptBuilder` 在运行时把"角色 + 历史记忆 + 可用技能 + 当前快照 + 组合状态 + 输出 schema"拼成 LLM 输入。同时引入 token 预算控制器，超长时按声明的优先级丢弃次要 section。落地后修改 prompt 不必改 Python 代码。

**In scope：**
- 4 个 analysis agent 的 prompt 配置外置（system_prompt 等 5 个 section）
- 统一 PromptBuilder + ConfigLoader + TokenBudgetEnforcer + 默认 Memory/Skill Provider
- 4 个 agent 文件内 ROLE 常量与硬编码拼接代码删除（无向后兼容路径）
- 8 字段 telemetry 挂到现有 OpenTelemetry tracing
- 单元 + E2E 测试

**Out of scope（值得审视的边界）：**
- Skill / Memory 进化算法（GEPA / Reflective Mutation / 5-signal FSM 等）→ spec 018
- SKILL.md schema 升级 → spec 018
- Verdict / Debate / Risk gate 节点的 prompt 外置 → 单独 spec
- 配置热重载（运行时 reload）→ 留待后续；本 spec 仅"重启服务后生效"
- Anthropic prompt cache 配置 → spec 018

## Bigger Picture

本 spec 是 trilogy（016 研究 / 017 基建 / 018 进化）的中间环节。016 已完成 8 个开源项目（SkillClaw / Hermes / OpenClaw-RL / EvoSkill / EvoSkills / MetaClaw / skill-evolution / autogen）的全角度研究，结论汇总在 [research/synthesis.md](../016-research-skill-evolution-prior-art/research/synthesis.md) 与 [decisions.md](../016-research-skill-evolution-prior-art/research/decisions.md)。本 spec 落地的 `MemoryProvider` / `SkillProvider` 协议是 spec 018 的对接点 —— spec 018 的"进化版 Provider"将直接 plug 进本 spec 的 PromptBuilder。

外部参考：Hermes Agent Self-Evolution（NousResearch）和 Anthropic Claude Code skills 都用 Markdown frontmatter 作为 agent 配置载体，本 spec 沿用此模式（决策见 [research/decisions.md D-PA-01](../016-research-skill-evolution-prior-art/research/decisions.md)）。Slot 分配（system 长稳定 prefix vs user_tail 动态短内容）参考 Anthropic prompt cache 命中率最佳实践（D-PA-02 / D-PA-05）。

值得思考的相邻关系：spec 014 的 `agent_skills/` 与 `agent_memory/` 目录结构本 spec **完全不动**，仅复用为 Provider 数据源。如果 spec 014 的目录结构在落地后发现需要 schema 升级，会落到 spec 018，本 spec 不做缓冲层适配 —— 这是一个赌注，值得审视。

---

## Spec Review Guide (30 minutes)

> 30 分钟评审建议把时间花在 4 处：迁移策略激进性、Slot 分配的 cache 假设、Provider 协议接口的稳定性、token 预算的"丢/降"语义。

### Understanding the approach (8 min)

读 [spec.md Purpose](spec.md#purpose) 与 [Implementation Outline](spec.md#implementation-outline) 了解整体路径，然后看 [plan.md Summary](plan.md#summary) 的技术映射。带着这些问题阅读：

- 把 ROLE 字符串从 Python 搬到 Markdown 是否真值得引入一个 `PromptBuilder` 类？是否存在更轻量的方案（比如直接用 Jinja2 模板）—— 见 [research.md Decision 1](research.md#decision-1markdown-frontmatter-作为-agent-配置载体) 的 alternatives
- 4 个 agent 是不是真的足够同构，可以共用一个 PromptBuilder？还是某个 agent（如 NewsAgent 涉及多源新闻聚合）会有独特拼接需求被压抑？
- spec 018 还没设计完，本 spec 现在固化的 `MemoryProvider` / `SkillProvider` Protocol 接口签名（见 [contracts/memory-provider.md](contracts/memory-provider.md) / [contracts/skill-provider.md](contracts/skill-provider.md)）是否会变成 spec 018 的紧身衣？

### Key decisions that need your eyes (12 min)

**直接删旧路径，无向后兼容 fallback** ([spec.md FR-X16, FR-X17](spec.md#migration直接切换无向后兼容))

用户在 brainstorm 时明确选了"直接使用新的，不需要考虑旧代码旧逻辑的兼容"，spec 沿用此决策：T1-T6 完成后 ROLE 常量全部删除，agent 构造器签名改为必填 `prompt_builder` 无默认值。回滚需 git revert，不能运行时切换。

- Question for reviewer：4 个 agent 一次性全切（不分阶段灰度），如果 PromptBuilder 在生产 cycle 出 bug，影响面是 4 个 agent 同时崩。是否值得保留一个"启动期 fallback flag"（比如环境变量 `USE_LEGACY_PROMPT=1`）作为应急回滚？还是相信单测 + E2E 测试足以兜底？

**Slot 分配假设：长稳定 prefix → SystemMessage / 动态 → UserMessage** ([data-model.md PromptBuilder build 数据流](data-model.md#运行期每-cycle-4))

默认把 `system_prompt` / `available_skills` / `output_schema` 入 SystemMessage，把 `recent_memory` / snapshot / portfolio 入 UserMessage。这个分配是为了 Anthropic prompt cache 命中（系统消息越稳定越好）。但是 `available_skills` 是否真的稳定？它由 `SkillProvider.get_available_skills(snapshot=...)` 返回，本 spec 默认实现按 agent_id tag 过滤（snapshot 不影响），但 spec 018 进化版会按 snapshot 做 IDF/match-score 排名，**snapshot 一变 skill 列表就变 → cache miss**。

- Question for reviewer：如果 spec 018 的 EvolvingSkillProvider 让 available_skills 每 cycle 都不同，本 spec 把它放 SystemMessage 反而让 cache 全 miss。是否应该现在就把 `available_skills` 默认放 user_tail？或者 spec 018 接入时再用 `slot_overrides` 调？决策见 [research.md Decision 4](research.md#decision-4slot-分配策略)，目前选了"默认 system / 后期可 override"。

**TokenBudgetEnforcer 的"丢段语义"** ([data-model.md TokenBudgetEnforcer 算法](data-model.md#tokenbudgetenforcer))

超 budget 时按用户在 frontmatter 里声明的 `priority: dict` 从大到小依次丢段，`system_prompt` 与 `output_schema` 强制保留，最后对 `recent_memory` / `available_skills` 截断。

- Question for reviewer：丢段是"丢整个 section"，不是"截断到 N token"。如果 `recent_memory` 含 5 个 cases，被丢就一个不剩。LLM 拿到的 prompt 是"完全没有历史" vs "完全有 5 个 cases" 的二值跳变。是否应该改为渐进截断（先丢最旧的 case，再丢倒数第二）—— 见 [contracts/prompt-builder.md TokenBudgetEnforcer](contracts/prompt-builder.md#class-tokenbudgetenforcer)，目前算法是 `pop` 整个 section，仅对 `recent_memory` / `available_skills` 兜底截断。

**Token 估算继续用 spec 014 的启发式（不引入 tiktoken）** ([research.md Decision 3](research.md#decision-3token-估算继续使用-spec-014-_estimate_tokens))

CJK÷1.5 + ASCII÷4，spec 014 已生产验证误差 < 10%。

- Question for reviewer：审视 [spec.md SC-X3(e)](spec.md#measurable-outcomes)：误差 < 10% 是否足以支持 budget=8000 的精度需求？8000 × 10% = 800 token 的偏差，在 LLM 上下文 200K 上下文里没事，但若误差是单向偏低，prompt 可能实际超 budget 800 token 进 LLM API → 触发 over-budget 报错。是否应该让估算偏向"高估"以留 safety margin？

### Areas where I'm less certain (5 min)

- [spec.md Edge Cases](spec.md#edge-cases) 列了 7 项，但缺一项：**4 个 agent 配置文件加载顺序若有任何一个失败导致 fail-fast，其他 3 个是否已经加载完成？**目前 `nodes/agents.py` 启动期 4 次实例化 PromptBuilder（[contracts/prompt-builder.md 调用契约](contracts/prompt-builder.md#调用契约总结)），第 1 个失败时第 2-4 个根本没创建 → 进程崩。这种语义是否应该改为"4 个一起加载、收集所有错误后再统一报"？目前的 fail-fast 对运维是好（错误清晰）但对调试是差（一次只能看到 1 个错）

- [tasks.md T030](tasks.md#phase-4-user-story-4--剩余-3-个-agent-迁移--role-退役priority-p2) 描述"4 处 agent 实例化注入 prompt_builder"。我假设 `nodes/agents.py` 当前就有 4 处 `TechAgent(...)` / `ChainAgent(...)` / `NewsAgent(...)` / `MacroAgent(...)` 调用 —— 没真正 grep 验证。如果实际是用 dict 动态构造（比如 `{"tech": TechAgent, ...}`），任务描述就要改

- [research.md Decision 9](research.md#decision-9defaultmemoryprovider-的-patterns--cases-拼接格式) 把 patterns + cases 用固定 markdown 子标题分段。如果 cases.jsonl 已经存在的格式与 `_format_case` 期望的字段（case_id / context / outcome / pnl）不完全匹配，T015-T017 的 TechAgent 迁移会跑不动。**spec 014 cases.jsonl 真实 schema 没在本 spec 锁定**

- [plan.md Performance Goals](plan.md#technical-context) 写了 "PromptBuilder.build() < 50ms" 但 [tasks.md](tasks.md) 里**没有性能基准任务**。如果 build() 因 file IO 慢于 50ms，影响每 cycle 200ms（4 agent × 50ms）。审视者可决定是否需要加一个性能 benchmark task

### Risks and open questions (5 min)

- 4 个 agent 一次切换的爆炸半径：测试覆盖度 SC-X4 + SC-X5 是否足够阻止生产事故？是否需要先在 paper trading 跑 N 个 cycle 验证再合 main？参考 [spec.md Reversibility](spec.md#reversibility) — "git revert T1-T6 全部 commit" 是否真的足够快（生产环境 5 个 commit 的 revert + redeploy）？
- 如果 `agent_skills/` 中某个 SKILL.md 的 frontmatter `tags` 当前没有 4 个 agent_id 中的任意一个，spec 018 进化前 [DefaultSkillProvider.get_available_skills](contracts/skill-provider.md#默认实现defaultskillprovider) 会返回空 → `available_skills` section 永远是占位"暂无可用技能"。是否值得在 T032 / T042 加一个"至少有 1 个 skill 被 tech tag 选中"的 sanity check 防止哑火？
- spec 018 接入 `EvolvingMemoryProvider` 时，本 spec 的 `MemoryProvider` Protocol `get_recent_memory(agent_id, snapshot, k=5)` 中 `k` 参数 spec 018 是否会觉得太僵硬？（spec 018 可能想按 token 而非 count 截断）。是否应该把签名改成更通用的 `get_recent_memory(agent_id, snapshot, max_tokens: int)`？参考 [contracts/memory-provider.md 与 spec 018 的协议契约](contracts/memory-provider.md#与-spec-018-的协议契约)
- [spec.md SC-X7](spec.md#measurable-outcomes) "token 差异 < 15%" 是参考性指标（不 fail），[tasks.md T037](tasks.md#phase-5-user-story-3--e2e--telemetry-验证priority-p2) 实现了对比但只 print warning。如果实测差异 25%（注入了大量新模板），是否应该至少写到 commit message？目前 spec 没强制

---
*完整内容见 [spec.md](spec.md)、[plan.md](plan.md)、[tasks.md](tasks.md)。*
