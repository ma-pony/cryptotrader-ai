# 综合分析：跨项目模式总结

**状态**：Phase 1 部分完成 —— 仅"提示组装"和"记忆 ↔ 技能"两个章节已完成。
**Phase 2 待补**：进化算法 / 技能数据结构 / 检索机制 / Evaluation / Agent ↔ Skill 边界 / 工程实现。

---

## Phase 1 — 第 1 章：提示组装模式

本章汇总 8 个被研究项目如何构建最终送入 LLM 的输入。源码引用指向各自的 `projects/<name>.md`。

### 模式 P1：命名分段组装（Sectioned Named-Segment）

**采用项目**：Hermes、MetaClaw、EvoSkills、SkillClaw（部分）、microsoft/autogen（工业级实现）

System prompt 由若干**命名的、半独立的段落**拼接而成，而不是单一的长字符串。每段有明确职责（记忆指引、技能列表、检索规则、输出格式等）。

| 项目 | 实现方式 |
|---|---|
| **Hermes** | `prompt_builder.py` 构建 `MEMORY_GUIDANCE` / `SESSION_SEARCH_GUIDANCE` / `SKILLS_GUIDANCE` 等命名段；部分段是 DSPy 参数化（可演进），部分段固定 |
| **MetaClaw** | `api_server.py` 提供三种组装模式（Synergy / Memory-only / Skills-only），每种都显式分段 + 20k token 贪心截断 |
| **EvoSkills** | 13 个 `SKILL.md` 文件每个描述一个段（"研究流程"、"数据分析"等）；声明式 `should-trigger` 元数据决定哪段被注入 |
| **SkillClaw** | 两阶段：第一阶段只注入 YAML name+description 目录；第二阶段命中后追加完整 SKILL.md body |
| **microsoft/autogen** | 工业级**四槽位分离**：`_system_messages`（静态角色）+ `_tools`（schema）+ `Memory.update_context()`（注入钩子）+ `ChatCompletionContext`（历史变体——buffered / head-tail / token-trimmed）；system prompt 本身**不进入** `ChatCompletionContext`，保持 prompt cache 友好 |

**对 cryptotrader-ai 的启示**：spec 017 的 prompt 外置应当采用**命名分段**（如 `agent_role` / `available_skills` / `recent_memory` / `output_schema`），而不是把 ROLE 拼成一大段字符串。每段就可以独立编辑 / 演进 / 受 token 预算约束。autogen 的四槽位模式是最直接可借鉴的工业实现 —— 见 D-PA-01。

### 模式 P2：渐进式披露 / 技能懒加载

**采用项目**：SkillClaw、skill-evolution

技能**不全部前置加载**。Prompt 起初只携带 metadata（name + 1 行描述），仅当 LLM 信号命中时才追加完整 skill body。

| 项目 | 机制 |
|---|---|
| **SkillClaw** | Client Proxy 启动时以全 skill 的 YAML 目录（每条 ~50 字节）注入；当 LLM 发出 `<load_skill>` 信号时为该轮次追加完整 SKILL.md（~1-15KB） |
| **skill-evolution** | 三层结构：L1 元数据始终在 context、L2 SKILL.md 触发时载入、L3 `references/` 文件按需拉取 |

**对 cryptotrader-ai 的启示**：当前 spec 014 架构把所有技能的 `description` 都内联加载。10 个技能 × 1KB 没问题，但到 100+ 时会撑爆预算。spec 017 设计 skill schema 时就要预留懒加载能力，即使 Phase 1 不实际实现。

### 模式 P3：tokenizer 原生模板渲染

**采用项目**：OpenClaw-RL（独家强势）

不自维护 prompt 模板，而是**委托给模型 tokenizer 的 chat template**（如 HuggingFace `tokenizer.apply_chat_template()` + Jinja2）。项目代码只做 role normalization 和内容整形，渲染由 tokenizer 按模型完成。

**对 cryptotrader-ai 的启示**：当前 prompt 是 Python f-string 拼。若改用 tokenizer-native template，prompt 格式就**自动随模型可移植**——从 glm-5 切到 gpt-5 切到 claude-opus-4-7 不再需要重写 prompt。spec 017 brainstorm 时值得评估。

### 模式 P4：注入点 — system prompt vs 最后用户消息

**采用项目**：OpenClaw-RL（用户消息尾部）；MetaClaw / Hermes / SkillClaw（system 注入）

两种截然不同的哲学：

- **System prompt 注入**（Hermes / MetaClaw / SkillClaw）：记忆和技能放在 system prompt 中，跨整个对话持久
- **最后用户消息注入**（OpenClaw-RL）：只在最终用户消息上追加经验，system prompt 保持不变

各有取舍，影响 token 缓存效果（Anthropic prompt cache 在 system prompt 稳定时命中率最高）和模型行为引导力（system prompt 通常被模型更深地内化）。

**对 cryptotrader-ai 的启示**：spec 017 必须明确选一个。考虑到 4-agent + verdict 架构 + spec 004 的 Anthropic prompt cache，**system prompt 注入 + 稳定结构很可能是更好选择**——但 OpenClaw-RL 的证据提示我们：每 cycle 易变的经验数据不妨放在用户消息尾部，避免缓存失效。

### 模式 P5：禁止会话中热替换

**采用项目**：Hermes（显式规则）、SkillClaw（仅在 session 间通过 dashboard sync 刷新）、skill-evolution（用 git 作为切换点）

会话开始时确定的 skill 集合，在会话结束前**不变**。新 skill 在**下一个**会话生效，不在对话进行中切换。这让 prompt cache 更高效，也避免了模型被漂移的 context 搞混。

**对 cryptotrader-ai 的启示**：当前 spec 014 的 reflection job 已经在 trading cycle **之间**运行，已天然遵循该模式。spec 017 应当把这条**写成显式约束**——agent prompt 在 cycle 开始时组装，cycle 内绝不变化。

### 模式 P6：token 预算贪心截断

**采用项目**：MetaClaw（独家具体，20k 上限）

构建出完整理想 prompt；若超预算，按优先级丢弃低优段直到合规。

**对 cryptotrader-ai 的启示**：当前没有 agent prompt 级别的 token 预算约束（仅在 LLM 调用层 retry on overflow）。spec 017 应当考虑显式的预算 enforcer ——在 skill 库扩张到 100+ 时尤其重要。

### 模式 P7：声明式触发元数据

**采用项目**：EvoSkills、SkillClaw（YAML）、skill-evolution（frontmatter + descriptions）

技能文件**声明何时该被考虑**（不只是描述自己做什么）。EvoSkills 用 frontmatter 中的 `allowed_tools:` 和 `should-trigger:`；SkillClaw 的目录用 1 行描述作为 LLM 路由依据。

**对 cryptotrader-ai 的启示**：spec 014 的 SKILL.md frontmatter 有 `name` 和 `description`。应当扩展（或在 spec 017 定义结构）增加声明式触发元数据——例如 `regime_tags: [trending, choppy]`，让 regime 检测层可以在不调用 LLM 的情况下做预过滤。

### 模式 P8：双模式检索（embedding + 关键词）

**采用项目**：MetaClaw

决定注入哪个 skill / memory 时，用**两条检索路径**：
- **模板 / 关键词匹配**：零延迟、高精度路由
- **Embedding 相似度**：语义模糊匹配

**对 cryptotrader-ai 的启示**：与 spec 018（技能检索算法）关系更大，但对 schema 设计已有影响——skill metadata 既要含人可读的触发关键词，又要支持 embedding 索引。

---

## Phase 1 — 第 2 章：记忆 ↔ 技能 的接线（lite 版）

（本章只覆盖记忆**注入到 prompt 的接线**，**不**含记忆演化算法 —— 后者归 Phase 2。）

### 模式 M1：记忆作为 prompt 的一段

8 个项目（有记忆机制的全部）都把记忆作为 P1 描述的命名段之一注入。**没有项目把记忆作为 tool call 结果返回**。

### 模式 M2：importance + access_count 用于检索

**采用项目**：MetaClaw（独家）

记忆单元自带 `importance: float` + `access_count: int`。检索按 importance 排序、按 access_count 衰减以避免反复引用。**启示**：spec 017 的 memory.md 格式应当含类似元数据字段；具体排序算法留给 spec 018。

### 模式 M3：因果链会话日志

**采用项目**：SkillClaw（独家强势）

会话被记录为结构化链：`user_prompt → tool_calls → intermediate_feedback → final_answer`。这成为人审 + 技能演化输入的双重底座。

**对 cryptotrader-ai 的启示**：当前 `agent_memory/cases/<commit_hash>.md` per-cycle 文件已经记录了 verdict + 风控 + 执行。扩展为更结构化的因果链（数据采集 → 4 agent 分析 → debate → verdict → execution → realized PnL）是 Phase 2 设计的自然方向。

### 模式 M4：反思写回（Reflection Write-Back）

**采用项目**：skill-evolution（7 步流程）

失败后 agent 自身按固定流程把学到的写回**技能文件**。**没有独立的"记忆数据库" —— 技能文件就是记忆。**

**对 cryptotrader-ai 的启示**：这是与 spec 014 reflection job + `agent_memory/<agent>/patterns/` 最接近的模式。证明我们的架构在主流。

### 模式 M5：改进后清空历史

**采用项目**：EvoSkill（独家）

成功改进后，驱动改进的失败历史被**清空**。否则下一次迭代的 LLM 会反复 rationalize 已经修过的问题。

**对 cryptotrader-ai 的启示**：当 reflection 产生新的 active pattern 后，触发该 pattern 的 cases 应当被标记 "consumed"（不再进入未来 reflection 窗口）。当前 spec 014 没有这一机制——值得在 spec 018 时考虑。

---

## Phase 1 — 行动建议汇总（≥10 条 actionable）

这些是 spec 017（prompt 外置）和 spec 018 一些核心设计选择的候选输入。

### R1：agent prompt 改用命名分段 schema（spec 017）
- **来源**：Hermes (`agent/prompt_builder.py`)、MetaClaw (`api_server.py`)、EvoSkills（13 个 SKILL.md 段）
- **影响对象**：cryptotrader-ai 中 `src/cryptotrader/agents/{tech,chain,news,macro}.py` 的 ROLE 字符串
- **行动**：把单一 ROLE 字符串替换为 TOML/YAML schema，声明 `agent_role` / `available_skills` / `memory_guidance` / `output_schema` 等键。运行时按段拼接，每段加显式标题。

### R2：技能 schema 支持渐进式披露（spec 017 schema、spec 018 实施）
- **来源**：SkillClaw (`api_proxy/skill_router.py`)、skill-evolution（L1/L2/L3 层级）
- **影响对象**：spec 014 的 `agent_skills/<agent>/SKILL.md` 格式
- **行动**：增加 `level: metadata|body|reference` 字段。Phase 1 prompt 只注入 metadata；后续阶段实施 body 懒加载。

### R3：tokenizer-native 模板（spec 017 评估）
- **来源**：OpenClaw-RL (`/openclaw-rl/utils/template.py`)
- **影响对象**：`src/cryptotrader/agents/base.py` prompt 构建
- **行动**：评估是否使用 LangChain 的 chat-template adapter（已支持 HuggingFace tokenizer）。决策依据：（a）是否计划频繁切换 LLM provider；（b）LangChain template adapter 增加多少摩擦。

### R4：注入点统一为 system prompt（spec 017 决策）
- **来源**：Hermes / MetaClaw / SkillClaw（system）；OpenClaw-RL（last-message）
- **影响对象**：spec 017 整体 prompt 策略
- **行动**：采纳 system prompt 注入策略 + 把每 cycle 易变的经验数据放在用户消息尾部，避免缓存失效。明确写入文档以保证缓存命中率。

### R5：禁止 cycle 内 prompt 变化（spec 017 不变量）
- **来源**：Hermes（明确规则）、SkillClaw（仅在 session 间同步）
- **影响对象**：agent_skills loader 行为
- **行动**：写成硬不变量。cryptotrader-ai 的 reflection job 已经在 cycle 之间，但要把规则显式化以防未来维护者无意中引入热加载。

### R6：显式 token 预算 enforcer（spec 017）
- **来源**：MetaClaw (`build_prompt` 贪心截断、20k 上限)
- **影响对象**：agent prompt 装配过程
- **行动**：分段组装后用 `tiktoken` 或模型特定库计算 token；超预算则按优先级丢弃。可按模型配置（glm-5 32k → 24k 预算；claude-opus-4-7 200k → 64k 预算，给响应留位）。

### R7：扩展 SKILL.md frontmatter 的声明式触发字段（spec 017 schema、spec 018 router）
- **来源**：EvoSkills (`should-trigger` + `allowed_tools`)
- **影响对象**：spec 014 的 `agent_skills/<agent>/<pattern>/SKILL.md` 格式
- **行动**：增加字段：`regime_tags: [trending, choppy, sideways]`（多选）、`triggers_keywords: [...]`。Phase 1 仅 schema；Phase 2（spec 018）实现 regime-tag 预过滤。

### R8：记忆单元带 importance + access_count（spec 017 schema、spec 018 排序）
- **来源**：MetaClaw（`MemoryUnit` dataclass）
- **影响对象**：`agent_memory/<agent>/patterns/<name>.md` 格式
- **行动**：frontmatter 增加 `importance: 0-1`（reflection 设定）和 `access_count: int`（每次注入时 +1）。

### R9：因果链会话日志（spec 018，schema 决策在 017）
- **来源**：SkillClaw（session DB schema）
- **影响对象**：`agent_memory/cases/<commit_hash>.md` 内容结构
- **行动**：把当前的"verdict + risk_gate"扩展为完整因果链（数据 → agents → debate → verdict → execute → realized_pnl）。有助 Phase 2 反思。

### R10：改进后清空历史（spec 018）
- **来源**：EvoSkill（`feedback_history.md` 成功后清空）
- **影响对象**：spec 014 reflection job 输入选择
- **行动**：reflection 产出新的 active pattern 后，把驱动它的 cases 标记 "consumed"，从未来 reflection 窗口中过滤掉。防止反复 rationalize。

### R11：技能成熟度门控信号（spec 018）
- **来源**：skill-evolution（5 个成熟度信号）
- **影响对象**：spec 014 的 PnL maturity FSM（observed → probationary → active → deprecated）
- **行动**：把 skill-evolution 的 5 个信号（样本量 / 胜率 / 时间等）与我们当前的纯 PnL FSM 对比；spec 018 brainstorm 时定夺。

### R12：双源技能检索（embedding + 关键词）（spec 018）
- **来源**：MetaClaw（Template + Embedding 双模式）
- **影响对象**：spec 018 的技能检索算法
- **行动**：留给 spec 018 brainstorm。当前 spec 014 完全没有检索（技能始终全加载）。

---

## Phase 2 待解决问题

1. **进化算法** —— 各项目方案差异巨大（OpenClaw-RL 的 RL、EvoSkill 的 pairwise eval、MetaClaw 的 generation counter、Hermes 的 dataset-builder）。需要 Phase 2 完整对比。
2. ~~**autoresearch 适配性**~~ —— ✅ 已于 2026-05-08 解决：按 D-PROC-01 替换为 `microsoft/autogen`。
3. **EvoSkill（单数）vs EvoSkills（复数）** —— 已确认是完全不同的两个项目。Phase 2 是否两者都保留，还是为预算精简？
4. **autogen 进化空白** —— autogen 对 Phase 2 进化算法研究毫无贡献（纯编排框架）。剩 7 个项目要承担那部分研究压力。
