# 架构决策记录（ADR）

**状态**：Phase 1 部分完成 —— 仅记录与 prompt 装配 + 记忆接线相关的 ADR。Phase 2 完成后将增补。

ADR 采用轻量格式：Status / Context / Decision / Consequences。编号约定：
- `D-PROC-NN`：流程 / 范围决策
- `D-PA-NN`：Prompt 装配相关（Phase 1）
- `D-MW-NN`：记忆接线相关（Phase 1）

---

## D-PROC-01：autoresearch 替换为 microsoft/autogen

**状态**：✅ Accepted（2026-05-08）

**背景**：`autoresearch`（uditgoenka，MIT）按 Tier 3 扫读完成。Phase 1 发现：该项目的"技能"和"记忆"概念与 LLM agent 技能演化几乎不挂钩 —— 它本质是基于 git history 的研究迭代框架，含静态 Markdown 程序文件，没有动态技能加载，没有 git log 之外的记忆层。`comparison-matrix.md` 中 autoresearch 行有 4 / 8 列被标记 N/A。

**决策**：将 autoresearch 替换为 **microsoft/autogen**，按 Tier 2 close-read 深度研究。

**为什么选 autogen 而不是 OpenHands/SWE-agent**：
- 多 agent 协作 + per-agent 状态管理直接镜像 cryptotrader-ai 的 4-agent + debate + verdict 架构
- `Memory.update_context()` 注入钩子 + `ChatCompletionContext` 历史变体正是 spec 017 需要设计的抽象
- OpenHands 偏 SWE 场景，间接性更高

**后果**：
- ✅ `autogen.md` 已生成（Tier 2，179 行，MIT license）
- ✅ 原 `autoresearch.md` 重命名为 `_deferred-autoresearch.md`（保留供参考，不在活跃研究集）
- ✅ `comparison-matrix.md` 与 `synthesis.md` 已更新为以 autogen 作为第 8 个项目
- Phase 2 进化算法研究范围不变（autogen 没有技能演化，那部分仍依赖 SkillClaw / EvoSkills / OpenClaw-RL / EvoSkill / skill-evolution）
- Phase 1 范围：8 个项目全覆盖，无 deferred-N/A 行

---

## D-PA-01：agent prompt 改为命名分段 schema（spec 017）

**状态**：Accepted（Phase 1）

**背景**：8 个研究项目中有 4 个（Hermes / MetaClaw / EvoSkills / SkillClaw 部分）使用命名分段而非整段字符串。cryptotrader-ai 当前的 `agents/{tech,chain,news,macro}.py` ROLE 是 30-60 行的 monolithic 常量 —— 难以演进、难以预算、难以复用。

**决策**：采纳命名分段 schema。每个 agent 的 prompt 由配置文件（TOML/YAML/MD frontmatter —— 具体格式由 spec 017 决定）声明，分段如下：
- `agent_role` —— agent 身份 / 角色描述（生命周期最长，最适合缓存）
- `available_skills` —— 当前 skill 目录片段（per-cycle）
- `memory_guidance` —— 如何使用记忆（相对固定）
- `output_schema` —— 必需的 JSON 形状（很稳定）
- `domain_checklist` —— agent 特定的核查规则

运行时 agent prompt = 按顺序拼接每段内容并加显式标题。

**后果**：
- spec 017 必须定义分段 schema + 拼接顺序
- 每段独立可编辑 / 可预算 / 可缓存
- 松耦合允许未来加技能感知段（agentic 段）而不需重写整个 ROLE
- agent 初始化时多解析一次配置文件 —— 可接受的开销

---

## D-PA-02：稳定内容入 system prompt；每 cycle 易变内容入用户消息尾部

**状态**：Accepted（Phase 1）

**背景**：synthesis.md 模式 P4 显示两个阵营：system prompt 注入（Hermes / MetaClaw / SkillClaw）vs 最后用户消息注入（OpenClaw-RL）。cryptotrader-ai 使用 Anthropic prompt cache（spec 004），缓存命中要求 system prompt 前缀稳定。

**决策**：采纳**混合策略**：
- 稳定项（agent_role / available_skills / memory_guidance / output_schema）→ system prompt
- 每 cycle 易变项（最新行情数据、最近 verdict commit hash、当前持仓状态）→ 用户消息尾部

这样既保持 prompt cache 高命中率，又能 per-cycle 更新 context。

**后果**：
- spec 017 必须定义每段属于哪个 prompt slot（system vs user tail）
- 缓存命中率可观测（spec 015 修复后已有相应 telemetry）
- 若某段被错放（例如 `available_skills` 放到 user tail），缓存 100% 不命中 —— 通过 metrics 可发现

---

## D-PA-03：禁止 cycle 内 prompt 变化

**状态**：Accepted（Phase 1）

**背景**：Hermes 显式禁止会话中热替换技能；SkillClaw 仅在 session 间通过 `dashboard sync` 刷新；skill-evolution 用 git 作为切换点。我们 spec 014 的 reflection job 已经在 trading cycle 之间运行。如果不显式写成不变量，未来代码可能误引入热加载，破坏 prompt cache + 让 agent 困惑。

**决策**：写成硬不变量：**agent prompt 在 cycle 开始时组装，cycle 内绝不变化**。spec 017 必须显式记录此约束，实现层应当 assert（例如 prompt builder 每 cycle 调用一次，结果视为 immutable）。

**后果**：
- Reflection job 保持 cycle 间运行（已是现状）
- 外部编辑器修改 agent_skills 文件不影响进行中的 cycle
- prompt cache 行为可预测

---

## D-PA-04：tokenizer-native 模板暂缓（spec 017 brainstorm 时再评估）

**状态**：Deferred

**背景**：OpenClaw-RL 的 tokenizer-native 渲染（HuggingFace Jinja2 chat template）让 prompt 格式自动随模型可移植。cryptotrader-ai 主要用 LangChain ChatOpenAI（model 由 config 选）；LangChain 自身有 chat-template 抽象。

**决策**：spec 017 brainstorm 时评估是否使用 LangChain 的 tokenizer-aware adapter，还是继续 f-string 拼接。决策依据：（a）我们多频繁切换 LLM provider；（b）LangChain template adapter 增加多少摩擦。

**后果**：
- 现在不承诺 —— 待 spec 017 brainstorm 决定
- 若采纳，prompt 渲染更可移植但多一层抽象
- 若推迟，prompt 格式仍 per-provider；切换 provider 需要重写

---

## D-PA-05：在 prompt builder 加 token 预算 enforcer

**状态**：Accepted（Phase 1）

**背景**：MetaClaw 的 20k token 贪心截断是唯一显式做了 prompt 级预算约束的项目。cryptotrader-ai 当前 agent 层无 enforcer（仅 LLM 调用层 retry on overflow，浪费）。

**决策**：spec 017 在 prompt builder 中定义 token 预算 enforcer。可按模型配置（如 glm-5 32k → 24k 预算；claude-opus-4-7 200k → 64k 预算，给响应留位）。超预算时按优先级丢弃：`output_schema`（必需，永不丢）> `agent_role`（必需）> `domain_checklist` > `available_skills`（降级到懒加载）> `memory_guidance`（降级到 ranking）> `recent_cases`（最旧先丢）。

**后果**：
- 可在大规模（100+ 技能）下保持可预测行为
- 避免 LLM 端 context-overflow 错误
- Telemetry 应当暴露 per cycle 的 "prompt_size / budget"

---

## D-PA-06：扩展 skill metadata schema（声明式触发字段）

**状态**：Accepted（Phase 1，仅 schema；逻辑在 spec 018）

**背景**：EvoSkills 用 frontmatter 中的 `allowed_tools:` 和 `should-trigger:` 做声明式路由。SkillClaw 启动时只用 YAML 目录做路由。两者都把"何时该用这条技能"和"技能本身做什么"解耦 —— 用于检索预过滤（避免仅为查 skill 该不该加载就调用一次 LLM）。

**决策**：spec 017 扩展 `agent_skills/<agent>/<pattern>/SKILL.md` frontmatter（已存在于 spec 014）：
- `regime_tags: [trending, choppy, sideways, breakout, reversal]`（多选）
- `triggers_keywords: [breakout_short, sma_breakdown, …]`（自由形式）
- `level: metadata|body|reference`（为未来渐进式披露准备）

Phase 1 仅 schema。spec 018 实现 regime-tag 预过滤。

**后果**：
- 现有 27 个 SKILL.md 文件需要迁移（增加新 frontmatter 字段，默认 `regime_tags: []`）
- 迁移用一次性脚本即可 —— spec 017 plan 可以包含
- spec 018 检索可获得 fast-path

---

## D-MW-01：记忆单元元数据 = importance + access_count

**状态**：Accepted（Phase 1，仅 schema）

**背景**：MetaClaw 独家在 memory unit 上跟踪 `importance` + `access_count`。没有这些，检索排序就是 ad-hoc 的。cryptotrader-ai 当前 `agent_memory/<agent>/patterns/<name>.md` 只有 PnL-track + maturity 阶段。

**决策**：spec 017 扩展 memory.md frontmatter：
- `importance: 0.0–1.0`（reflection 设定 —— PnL 表现稳定的 pattern 高分）
- `access_count: int`（每次该 pattern 被注入到 prompt 时 +1）
- `last_accessed_at: ISO8601`（spec 018 时间衰减排序的输入）

**后果**：
- 现有 patterns 需要迁移（默认 importance = 0.5、access_count = 0）
- spec 018 排序算法有材料可用
- `importance` 字段成为"人工覆盖"的代理 —— 用户编辑了的 importance，reflection 应当尊重

---

## D-MW-02：因果链会话日志（扩展 spec 014 case 文件）

**状态**：Deferred（在 spec 018 brainstorm 时决定；实施在 spec 018）

**背景**：SkillClaw 独家把会话记录为因果链（user_prompt → tool_calls → intermediate_feedback → final_answer）。我们的 `agent_memory/cases/<commit_hash>.md` 当前仅记录 verdict + risk gate + execution，不是完整因果链。扩展会有助 Phase 2 反思但增加 per-cycle 存储量。

**决策**：具体 schema 留给 spec 018 brainstorm —— 但标记为"高价值杠杆"。Phase 2 须把 SkillClaw 的 session DB 结构与我们的 case 文件结构做完整对比。

**后果**：
- Phase 1：无动作
- spec 018 brainstorm 有明确研究目标

---

## D-MW-03：改进后清空历史

**状态**：Deferred（在 spec 018 决定）

**背景**：EvoSkill 在成功改进后清空 `feedback_history.md`，防止反复 rationalize。cryptotrader-ai 的 reflection job 当前把所有历史 case 都纳入范围，包括已经驱动过 pattern 的那些。

**决策**：留给 spec 018 brainstorm —— 实施需要先就"在我们 PnL 驱动模型下什么算 consumed"达成共识。

**后果**：
- Phase 1：无动作
- spec 018：必须解决（否则 reflection 反复在已修复的 pattern 上循环）

---

## Phase 2 待补 ADR 类别

下列模式类别将在 Phase 2 研究中产生 ADR：

- 进化算法（RL vs pairwise vs dataset-builder vs maturity FSM）
- 技能数据结构（Markdown vs JSON vs DSL）
- 检索算法（关键词+embedding vs 图遍历 vs LLM-routed）
- 评估方法（verifier reward vs PnL vs human-in-loop）
- Agent ↔ Skill 边界（per-agent vs cross-agent）
- 工程实现（文件同步 vs DB vs 内存）

以上仅占位 —— 实际决策待 Phase 2 深读完成。
