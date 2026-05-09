# Review Guide: Agent Prompt Builder Integration（spec 017b）

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-05-08

---

## What This Spec Does

完成 spec 017a 拆出去的"4 agent 真切换"工作。spec 017a 交付了 PromptBuilder 基建模块，但 4 个 analysis agent（tech / chain / news / macro）当时仍跑硬编码 ROLE 路径。本 spec 把 4 agent 切到 PromptBuilder 路径，同时删除三处旧 prompt 拼接路径：`base.py:ANALYSIS_FRAMEWORK + role_description`、`config.py:_resolve_role + _resolve_skills + prompt_template`、`agents/skills/middleware.py` 整个文件。

**In scope：**
- 4 agent 配置文件（`config/agents/{tech,chain,news,macro}.md`）
- 1 新模块（`snapshot_renderer.py` 把 crypto 领域的 prompt 渲染逻辑物理隔离）
- BaseAgent / ToolAgent / 4 agent 类构造器与 analyze() 重构
- AgentsConfig.build() 重构（删 _resolve_*）
- DefaultSkillProvider 用 `discover_skills_for_agent()` 替代 017a 的错误 tags 过滤
- nodes/agents.py module-level Provider singleton
- E2E 测试 + ROLE/middleware 退役 grep gate

**Out of scope（关键边界）：**
- **DefaultMemoryProvider 路径修复** — spec 017a 设计的 `agent_memory/<id>/{patterns.md, cases.jsonl}` 与 spec 014 实际 `agent_memory/cases/<cycle_id>.md`（全局）不匹配；本 spec 用 `experience: str` 参数旁路解决，DefaultMemoryProvider 路径修复推迟 spec 018
- skill / memory 进化算法 / SKILL.md schema 升级 / verdict-debate-risk prompt 外置 / 配置热重载 / Anthropic prompt cache 配置 / load_skill_tool 删除决策 — 全部移给 spec 018 或独立 spec

## Bigger Picture

spec 017b 是 trilogy（016 研究 / 017a 基建 / 017b 集成 / 018 进化）中"集成切换"环节，必须在 spec 018 启动前合并 main，否则 spec 018 的 EvolvingMemoryProvider 没有"接入点"。spec 017b 落地后 spec 018 仅需替换 `nodes/agents.py:_memory_provider` / `_skill_provider` 的 Default 实例为 Evolving 实例，**无需改本 spec 任何代码**。

外部参考：spec 016 的 8 项目研究（[research/synthesis.md](../016-research-skill-evolution-prior-art/research/synthesis.md)）已确立"PromptBuilder + Provider Protocol"是借鉴 Hermes / autogen 的成熟模式；本 spec 在此基础上完成 4 agent 实际切换。

值得思考的相邻关系：spec 014 的 `verbal_reinforcement` 节点输出 `experience: str` 进 state，本 spec 通过保留 `BaseAgent.analyze(snapshot, experience)` 的 experience 参数 + `PromptBuilder.build(..., experience)` 的扩展，让现有 verbal reinforcement context 注入路径**无回归**。这是"DefaultMemoryProvider 路径推迟修复"得以成立的前提。

## Spec Review Guide (30 分钟)

> 30 分钟 review 建议把时间花在 4 处：atomic C2 commit 风险、ANALYSIS_FRAMEWORK 4 倍重复成本、experience 参数的语义边界、scope filter 修改的回归面。

### Understanding the approach (8 min)

读 [spec.md Purpose](spec.md#purpose) + [tasks.md C1/C2/C3 commit 序列](tasks.md#phase-2-foundational--c1-commitatomic-切换) 了解整体路径。带着这些问题阅读：

- 是否真的需要把"删 SkillsInjectionMiddleware + 删 ANALYSIS_FRAMEWORK + 删 role_description + 4 agent 重构"放在同一个 atomic C2 commit？拆成 2-3 个 commit 让 git bisect 颗粒度更细是否值得？参考 [tasks.md Implementation Strategy](tasks.md#implementation-strategy)
- spec 014 / 015 / 017a 已有的回归测试是否真的能覆盖"4 agent 通过 PromptBuilder 调 LLM 后输出 AgentAnalysis"这一新路径的全部分支？特别是 ToolAgent 的 create_agent 循环。FR-Y10 改了 ToolAgent.analyze() 但未列出每个分支的 test 计划
- 4 agent 文件 < 150 行（SC-Y5）是否可达？TechAgent 现在 174 行（含 ROLE 16 行 + compute_indicators 80 行 + analyze 8 行 + helpers 30 行 + 其他）。删 ROLE 16 行 + 改 analyze 后能到 < 150？compute_indicators 算业务还是渲染前置？

### Key decisions that need your eyes (12 min)

**Atomic C2 commit 体积** ([tasks.md Phase 3](tasks.md#phase-3-user-story-migration--c2-commitatomic-切换))

C2 commit 含 36 个 task，diff 估算 ~1100 行，触及 base.py / 4 agent / config.py / nodes/agents.py / prompt_builder.py / security.py + 4 个 test 更新 + 删 1 个文件。spec 017a 实施时 subagent 在更小的 atomic commit 上 drift（误改 memory.py），本 spec 体积更大。

- Question for reviewer：**是否值得把 C2 拆为 C2a (base.py + prompt_builder + 删 middleware) + C2b (4 agent + nodes/agents.py + 4 test 更新) 两个 commit？** C2a 单独完成时 main 路径还是 broken 的（ROLE 没动但 base.py.role_description 字段消失），所以拆分需要 compat shim — 但用户在 brainstorm Q1 选了 C 激进（无 fallback）。这是矛盾点。

**ANALYSIS_FRAMEWORK 4 倍重复成本** ([research.md Decision 3](research.md#decision-3analysis_framework-拆段策略))

ANALYSIS_FRAMEWORK 35 行 × 4 agent = 140 行 markdown 重复。任何修改都要改 4 处。spec 014 / 015 落地后 ANALYSIS_FRAMEWORK 未变过；但 spec 018 是否会动它（如增 maturity FSM 相关 instruction）？

- Question for reviewer：是否应该在 PromptBuilder.build() 内部硬编码 ANALYSIS_FRAMEWORK 注入（每个 agent 的 system_prompt 自动 prepend），让 4 个 config 文件不重复？trade-off 是 PromptBuilder 失去 generic 性。但反正 PromptBuilder 已经因 `_render_snapshot` 调 `render_crypto_snapshot` 而失去 100% generic 了。

**experience 参数旁路 DefaultMemoryProvider** ([contracts/promptbuilder-experience-extension.md](contracts/promptbuilder-experience-extension.md))

spec 017a 的 DefaultMemoryProvider 现在是僵尸代码（路径错）；本 spec 通过 `experience: str` 参数让 PromptBuilder 跳过 MemoryProvider，直接把 experience 当 recent_memory section 内容。

- Question for reviewer：这是否给 spec 018 留了不必要的复杂性？spec 018 想让 EvolvingMemoryProvider 真正接管 recent_memory section，那时 experience 参数怎么定位？两个机制（参数注入 vs Provider）会不会互相打架？参考 [contracts/promptbuilder-experience-extension.md "Schema 升级路径"](contracts/promptbuilder-experience-extension.md#schema-升级路径) — 但目前的措辞是"两个机制都保留"，这真的是好设计吗？

**DefaultSkillProvider scope filter 修改的回归面** ([data-model.md DefaultSkillProvider](data-model.md#defaultskillprovider-spec-017a-沿用--本-spec-bug-fix))

017a 的 `agent_id in skill.tags` 过滤永远返回 0 skill（因为 SKILL.md frontmatter 用 `scope` 不是 `tags`）。本 spec 改为 `discover_skills_for_agent(agent_id)`，4 agent 立刻"看到"现网所有 scope: shared + scope: agent:<id> 的 SKILL。

- Question for reviewer：现网 `agent_skills/` 目录下有多少 SKILL？每个 skill body 多少 token？4 agent 每次 cycle 都拼这些 body，**token 总量是否会超 budget**？017a TokenBudgetEnforcer 会按优先级丢，但首次跑可能丢的是其他 section（如 snapshot），影响决策质量。建议在 C2 commit 前 sample-check `wc -l agent_skills/*/SKILL.md` 估算 token 占用。

### Areas where I'm less certain (5 min)

- [spec.md FR-Y22](spec.md#config-py-重构) 删 `AgentConfig.prompt_template` 字段。我假设此字段未在生产 TOML 配置中使用（pre-ship spot-check 显示 `grep prompt_template config/` 无匹配）。但 `prompt_template` 字段仍可能在 dataclass 初始化的 default 处被引用（`AgentConfig(agent_id=..., prompt_template="")` 作为 dict 解析的兜底）。需要 C2 实施时 grep 整个 src/ 找 `prompt_template` 引用面。
- [tasks.md T022](tasks.md#base-py-重构) 添加 `BaseAgent._snapshot_to_dict(snapshot: DataSnapshot) -> dict`。我假设 spec 014 `DataSnapshot` 类有所有需要的字段（market.ticker / market.funding_rate / market.volatility / market.ohlcv / news.headlines / onchain.* / macro.*）。如果某字段是 lazy-load 或 method 而非 attribute，T022 会需要适配。**未实际 grep DataSnapshot 类定义**。
- [tasks.md T035](tasks.md#nodes-agents-py-wiring) "load_skill_tool 注入 ToolAgent.tools" 的具体方式留给实施。如果在 `_build_builtin` 内的 lambda 里包装 `tools=[*CHAIN_TOOLS, load_skill_tool]`，需要修改 chain.py / news.py 的现有 CHAIN_TOOLS / NEWS_TOOLS 组装代码 —— 但 chain.py / news.py 在 T028 / T029 也在改构造器。两个 task 的边界需要 careful。
- [spec.md SC-Y5 4 agent < 150 行] 当前 tech.py 174 行；删 ROLE (16 行) + 删 _build_prompt (4 行 override) + 改构造器 (3 行 → 2 行) ≈ 174 - 23 = 151 行。**临界值，可能需要把 compute_indicators 内部 helper 拆出去单独模块**。其他 3 agent 行数未量化。

### Risks and open questions (5 min)

- 实施 subagent 在 C2 atomic commit 期间是否会触及 OOS 范围的文件（如 spec 017a 实施时误改 `learning/memory.py`）？建议在 spawn implement subagent 时 prompt 中明确"以下文件 NOT 在范围内：learning/memory.py / agents/skills/_compat.py / journal/* / portfolio/* / risk/*"（白名单 + 黑名单都给）。tasks.md T036-T038 的 grep cleanup 步骤本身可能误删 docstring 提及（false positive）—— 需要 grep 后人工 review 每个匹配
- experience 参数与 MemoryProvider fallback 的语义在 spec 018 的 EvolvingMemoryProvider 接入时会不会产生歧义？例如：spec 018 用 EvolvingMemoryProvider 自动管理 recent_memory，调用方还可以传 experience 字符串覆盖吗？这种"覆盖语义"是否会让 spec 018 的进化算法失去对 recent_memory section 的 100% 控制？
- C2 commit 后第一次生产 cycle，4 agent 可能 token 占用激增（DefaultSkillProvider 现在真的能加载 skill body）。是否值得在 C2 commit 后 manual 跑 1 个 cycle 确认 prompt 长度 + LLM 响应正常，再 merge main？
- spec 014 SKILL.md 实际 frontmatter `scope` 字段填得是否完整？scope spot-check 没做（spot-check 4 项都是 negative grep）。如果某个 SKILL.md 漏填 scope（默认 shared？还是默认全部 agent？），DefaultSkillProvider 行为会受影响。建议 C1 commit 前跑：`for f in agent_skills/*/SKILL.md; do echo "=== $f ==="; head -10 "$f"; done` 抽查

---
*完整内容见 [spec.md](spec.md)、[plan.md](plan.md)、[tasks.md](tasks.md)。*
