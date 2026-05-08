# 综合分析：跨项目模式总结

**状态**：✅ **Phase 2 完成** —— 全部 8 个研究角度对全部 8 个项目都已覆盖。
**最后更新**：2026-05-08（Phase 2 合并）

各项目 Phase 2 文档行数：SkillClaw 646 / Hermes 608 / OpenClaw-RL 640 / MetaClaw 651 / EvoSkill 671 / EvoSkills 611 / skill-evolution 696 / autogen 614。共 5137 行项目文档。

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

---

## Phase 2 — 第 3 章：进化算法（Evolution Algorithm）

7 个有进化机制的项目（autogen 无）的核心算法对比：

| 项目 | 算法范式 | 信号来源 | 触发条件 | GPU 需求 |
|---|---|---|---|---|
| **OpenClaw-RL** | GRPO 真 RL（Group Relative Policy Optimization）+ SLIME 分布式训练后端 | PRM Judge 三值打分（+1/-1/0）多数投票 | 持续在线（推理同时训练） | **8 GPU**（4 训练 + 2 推理 + 2 PRM）|
| **Hermes** | **GEPA（遗传-帕累托提示进化）** + DSPy Signature 包装 + Reflective Mutation | LLM 生成自然语言反馈（"为什么这条 prompt 失败"）→ 注入下一次变异 | 离线批量（每次运行 $2-10）| 无（DSPy 全 API 调用，35× 少 rollouts 即可超 GRPO）|
| **SkillClaw** | 监督式 LLM 批处理 5 阶段流水（Summarize → Session Judge → Aggregate → Execute → Skill Verifier）| LLM-as-judge 4 维加权评分（任务完成 55%）| 每 10 分钟轮询，有新会话即触发 | 无 |
| **EvoSkill** | **纯数值 accuracy frontier 替换** —— 子程序在验证集上的准确率与 frontier 最差成员比较，高者替换低者 | **客观 accuracy 指标**（无 LLM-as-judge）| `update_frontier()` 返回 True 即触发 history-clear | 无 |
| **MetaClaw** | **双循环**：快内循环（失败对话 LLM 提炼新技能）+ 慢外循环（用户空闲时 Tinker LoRA 梯度更新）| 失败对话 + LoRA 梯度信号 | 三路：质量门槛（成功率 < 0.4）/ 定时（每 10 轮）/ session_done | LoRA 阶段需 GPU；纯 prompt 阶段无 |
| **EvoSkills** | **IDE/IVE/ESE 三机制**（成功向上抽象 + 失败分类 + 策略蒸馏）| Elo 比较 + 五诊断问题决策矩阵 | 成功后触发 IDE/ESE；失败后触发 IVE | 无 |
| **skill-evolution** | **7 步反思 + 5 信号成熟度 FSM**（draft → tested → stable → clean → mature）| 用户/agent 自身确认 + 静默漏触发信号 | 失败后启动 7 步流程；Step 6 保留人类确认门控 | 无 |

**对 cryptotrader-ai 的核心启示**：
- ❌ **OpenClaw-RL 真 RL 路线排除**（本机无 GPU）
- ✅ **EvoSkill 纯数值 frontier** 最契合本项目特征 —— 我们有客观 PnL 信号，可直接用 PnL 替代 accuracy
- ✅ **Hermes Reflective Mutation** 应嵌入到 reflect 流程：LLM 对失败 case 生成"为什么这条规则失败"的自然语言反馈作为下次规则修改的输入
- ✅ **SkillClaw 5 阶段批处理**作为整体框架：把当前同步 reflection 拆解为离线批量管道，消除交易路径上的 LLM 阻塞
- ✅ **MetaClaw "成功率门槛" 防过拟合**：win_rate 高时跳过规则提炼

---

## Phase 2 — 第 4 章：技能数据结构

| 项目 | 技能基本单元 | 关键 schema 字段 | 大小约束 |
|---|---|---|---|
| **Hermes** | `SKILL.md`（YAML frontmatter + Markdown body）| `name` / `description`（冻结）；body 可演化 | ≤15KB body；提示段 ≤120% 当前长度 |
| **SkillClaw** | `SKILL.md`（YAML frontmatter + body）| `name` / `description` / `keywords` / `version` / `parent`（谱系字段）| 无明确字节上限，但有 `push_min_injections=5` 注入门槛 |
| **EvoSkill** | git 分支 + `program.yaml` + 可执行代码 | `parent_iter` / `score` / `created_at` / `frontier_status` | git tag 标记 frontier；分支 `program/iter-N` |
| **MetaClaw** | 6 类 `MemoryUnit` dataclass（schema-typed）| `type` / `content` / `importance` / `access_count` / `recency` / `keywords` / `embedding` / `confidence` | 无字节上限，按 token 预算贪心截断（20k 上限）|
| **EvoSkills** | 13 个 `SKILL.md`（Markdown-first）| `allowed_tools` 白名单 + `should-trigger` 路由描述（声明式）| 各 SKILL 自然 ≤500 行 |
| **skill-evolution** | 三层结构 L1/L2/L3 | L1 frontmatter（≤300 行）+ L2 `SKILL.md` body + L3 `references/*` 按需 | L1 ≤300 行；脚本/hooks 无字节上限 |
| **OpenClaw-RL** | 工具调用序列（trajectory）| 无 SKILL 抽象，技能体现在策略权重 + 工具描述 | 工具描述 ≤500 字符 |
| **autogen** | `BaseTool` / `FunctionTool` / `Workbench` | JSON Schema（自动生成）+ Python callable | 无明确上限 |

**对 cryptotrader-ai 的核心启示**：
- 当前 spec 014 的 `agent_skills/<agent>/<pattern>/SKILL.md` schema 已与主流（Hermes / SkillClaw / EvoSkills）对齐
- **建议扩展字段**（D-PA-06 + Phase 2 新发现）：
  - `version` + `parent`（EvoSkill 的谱系字段）—— 用于 frontier 管理
  - `importance` / `access_count` / `recency` / `confidence`（MetaClaw 的 MemoryUnit）—— 检索排序
  - `regime_tags` / `triggers_keywords` / `allowed_tools`（EvoSkills 声明式路由 + spec 014 已有）
  - `level: metadata|body|reference`（skill-evolution 三层渐进披露）
  - `maturity: draft|tested|stable|clean|mature`（skill-evolution 5 信号 FSM 替代当前 float）

---

## Phase 2 — 第 5 章：检索机制（Retrieval）

| 项目 | 检索算法 | 阈值 / k 值 | 触发点 |
|---|---|---|---|
| **MetaClaw** | **IDF 加权混合**：(关键词IDF + embedding余弦 + importance + recency_bonus + reinforcement_score) × type_boost × confidence_factor | top-k 可配；recency 连续衰减；多源同时打分 | 每次 prompt 装配前 |
| **SkillClaw** | LLM 自身判断（让模型在 prompt 阶段输出 `<load_skill>` 信号）| `push_min_injections=5` 注入数量门槛 | 第二阶段（catalog → body）|
| **Hermes** | `SESSION_SEARCH_GUIDANCE` 本身是**可演化 prompt 段** —— 检索触发逻辑也被 GEPA 优化 | 由演化决定 | 由 GUIDANCE 文本控制 |
| **EvoSkills** | top-k 余弦相似度（Qwen embedding）+ 关键词路由 fallback | k 默认 5；阈值未明 | 每次科研流程入口 |
| **EvoSkill** | 选择 frontier Top-N 作为 Proposer 输入 | N 由 fitness rank 决定 | 每次提案 |
| **autogen** | `ListMemory` 全文注入 / `ChromaDB` 向量 + `score_threshold` | score_threshold 用户配 | `Memory.update_context()` 注入点 |
| **skill-evolution** | L1 关键词 → L2 触发 → L3 按需（渐进披露）| 无显式阈值，靠声明式 trigger | LLM 主动 load |
| **OpenClaw-RL** | 经验注入到最后一条 user 消息 | session-experience / replay / consolidate 三模式 | 每个 turn |

**对 cryptotrader-ai 的核心启示**：
- 当前 spec 014 **完全没有检索**（所有技能始终全加载）—— spec 018 必须设计检索层
- **推荐方案**：MetaClaw 的 IDF 混合检索 + spec 018 增加 embedding（用本地 Qwen3 或 sentence-transformers）
- **regime_tags 预过滤**作为零延迟首层（EvoSkills 风格的声明式路由）+ embedding 作为语义 fallback

---

## Phase 2 — 第 6 章：评估（Evaluation）

| 项目 | 评估方式 | 客观信号？ | 失败回滚 |
|---|---|---|---|
| **EvoSkill** | **纯数值 accuracy** 在验证集上比较 | ✅ 是（accuracy 是数值）| frontier 替换不胜出则不晋级，自动回滚 |
| **OpenClaw-RL** | PRM Judge 三值打分（+1/-1/0）多数投票 | 半客观（依赖 PRM 模型质量）| GRPO 信用分配 |
| **SkillClaw** | LLM-as-judge 4 维加权（任务完成 55% / 决策质量 / 简洁度 / 引用证据）| ❌ 主观（LLM 评分）| Skill Verifier 四项准入门控（阈值 0.75）|
| **Hermes** | 多目标 fitness：50% 正确率 + 30% 程序遵循 + 20% 简洁（带长度惩罚）| 半客观（依赖 LLM-as-judge 子项）| Pareto 前沿替换 + 不超越则保留 |
| **MetaClaw** | 检索成功率 + 用户反馈（隐式信号）| 半客观 | 无显式回滚（写入磁盘后只能用新 skill 覆盖）|
| **EvoSkills** | Elo 比赛 + 三连败永久归档 | 比赛对比信号客观 | 永久归档（archive.md）|
| **skill-evolution** | 5 信号成熟度通过/不通过判定 | 半客观（部分人类确认）| 用户审批门控 |
| **autogen** | `agbench`（Docker 隔离 + JSONL 任务）| 取决于任务 | 无内置 |

**对 cryptotrader-ai 的核心启示**：
- 我们有**最干净的客观信号 = realized PnL** —— 不需要 LLM-as-judge，可走 EvoSkill 纯数值路线
- **建议评估指标**（未来 spec 018 决策）：win_rate / avg_R:R / Sharpe / max_drawdown / cumulative PnL —— 任选 1-2 作为 frontier 比较函数
- **回滚机制**：参考 EvoSkill 的 frontier 替换 + skill-evolution 的"2 次同问题→停止打补丁"安全阀
- **多目标 Pareto**（Hermes 启发）：win_rate × confidence 双目标 frontier，避免单维度优化损害其他维度

---

## Phase 2 — 第 7 章：Agent ↔ Skill 边界

| 项目 | Agent 抽象 | 边界协议 |
|---|---|---|
| **autogen** | `BaseChatAgent` / `AssistantAgent` / `Team`（3 层）；`SocietyOfMindAgent` 模式 | **增量消息协议**（只传新消息，agent 自维护状态）|
| **OpenClaw-RL** | API Server / Rollout Worker / PRM Judge / Trainer（4 异步循环）| OpenAI 兼容 API 协议；per-token log-probabilities |
| **Hermes** | Hermes Agent（外部仓库）+ optimization pipeline（独立）| 通过环境变量 + `batch_runner.py` + `trajectory.py` 解耦 |
| **SkillClaw** | Client Proxy + Evolve Server | 拦截 `/v1/chat/completions`；客户端通过 `dashboard sync` 拉取 |
| **MetaClaw** | api_server.py 透明代理 | 三模式（Synergy / Memory-only / Skills-only）切换 |
| **EvoSkill** | sentient-agi SDK | Proposer / Generator / Evaluator / Frontier 四角色解耦 |
| **EvoSkills** | 无显式 agent 抽象 | 13 个 SKILL.md 即 prompt body，由科研流程驱动 |
| **skill-evolution** | 为 Claude Code 设计的元框架 | 7 步反思 + 确定性阶梯（LLM/脚本/hooks 职责）|

**对 cryptotrader-ai 的核心启示**：
- **`SocietyOfMindAgent` 模式**完美对应 cryptotrader-ai 的 **debate subgraph 应对外透明**：把 4 agent + debate + verdict 打包成单一 agent 接口暴露给 LangGraph 其余节点
- **autogen 增量消息协议**降低 debate 节点与 LangGraph 全量状态的耦合
- **SkillClaw / Hermes 独立优化进程**模式：spec 018 的 reflect.py 可设计为独立 daemon（不阻塞交易路径）
- **skill-evolution 确定性阶梯**：分清"LLM 该做什么 / 脚本该做什么 / hooks 该做什么"防止 LLM 处理确定性问题

---

## Phase 2 — 第 8 章：工程实现

关键工程模式跨项目对比：

| 模式 | 采用项目 | 实现要点 |
|---|---|---|
| **git 分支谱系** | EvoSkill（核心）、SkillClaw（cross-user history）| `program/iter-N` 分支 + `frontier/*` 标签 + `program.yaml` 谱系字段；零额外存储成本 |
| **离线批量演化进程** | SkillClaw / Hermes / MetaClaw | 与交易/推理路径解耦；不阻塞 hot path |
| **失败签名缓存** | （现有 spec 015 已实现 OKX 51202）| 30 min TTL；下次同签名自动 fallback |
| **token 预算贪心截断** | MetaClaw（20k）| 按段优先级丢弃 |
| **共享存储** | SkillClaw（local + S3）| 多客户端协同进化 |
| **Workbench 工具组织** | autogen | `list_tools()` 动态查询接口 |
| **HeadAndTail 上下文窗口** | autogen 独家 | head 保前缀稳定（缓存命中）+ tail 保最新轮次 |
| **per-token log-probabilities** | OpenClaw-RL | 训练后端必需，纯推理可省 |
| **SLIME 分布式训练后端** | OpenClaw-RL | 8 GPU；本项目不可行 |
| **`agbench` Docker 评估** | autogen | 任务隔离 + JSONL 定义 |

**对 cryptotrader-ai 的核心启示**：
- ✅ **git 谱系**：spec 018 的 ExperienceRule 版本管理可用 git 分支，0 额外存储
- ✅ **离线批量 reflect daemon**：把当前 cycle 内的 reflection 抽出为独立进程
- ✅ **HeadAndTail 上下文**：spec 017 prompt 装配已选 system 稳定 + user tail 易变，与该模式同构
- ✅ **失败签名缓存**：spec 015 已落地（OKX maxMktSz），可推广到其他失败模式

---

## Phase 2 — 行动建议增量（R13 ~ R20+）

延续 Phase 1 的 R1-R12，新增 Phase 2 模式驱动的建议：

### R13：spec 018 进化算法主线 = SkillClaw 5 阶段 + Hermes Reflective Mutation + EvoSkill 数值 frontier
- **来源**：SkillClaw（流水管道）+ Hermes（自反 mutation）+ EvoSkill（客观信号）
- **影响**：spec 018 的核心算法选型
- **行动**：reflection 流程拆为 5 阶段（摘要 / 评分 / 聚合 / 执行 / 准入），每阶段独立可测；变异时让 LLM 生成"为什么失败"反馈作为输入；用 PnL（而非 LLM-as-judge）做 frontier 比较

### R14：5 信号成熟度 FSM 替换 maturity float
- **来源**：skill-evolution
- **影响**：`ExperienceRule.maturity` 字段
- **行动**：`maturity: draft|tested|stable|clean|mature`；状态转换条件参考 skill-evolution（production-tested ≥3 / stable 5 次无修改 / well-structured / clean / self-contained）

### R15：IDF 混合检索 + regime_tags 预过滤
- **来源**：MetaClaw（IDF 混合）+ EvoSkills（声明式路由）
- **影响**：spec 014 当前无检索 → spec 018 加检索层
- **行动**：先 regime_tags 关键词预过滤（零延迟），再 IDF 加权混合（关键词IDF + embedding余弦 + importance + recency + confidence）选 top-k

### R16：双目标 Pareto frontier（win_rate × confidence）
- **来源**：Hermes GEPA Pareto + EvoSkill frontier
- **影响**：当前规则是单维度替换；spec 018 改双维度
- **行动**：保留"在至少一个维度达到最高分"的候选集合；防止单维度优化损害其他维度

### R17：失败分类 prompt 模板（实现失败 vs 根本失败）
- **来源**：EvoSkills IVE
- **影响**：当前亏损归因不区分"短期噪声 vs 策略根本错误"
- **行动**：在 reflect.py 加亏损归因 prompt 模板，5 诊断问题区分两类失败；三连"根本失败"自动归档该规则

### R18：离线 reflect daemon
- **来源**：SkillClaw / Hermes / MetaClaw 都用独立进程
- **影响**：当前 reflect.py 在 cycle 内同步执行，阻塞交易路径
- **行动**：拆为独立 daemon，cycle 只写 case 文件；daemon 每 N 分钟批量 reflect

### R19：HeadAndTailChatCompletionContext 模式
- **来源**：autogen
- **影响**：spec 017 的 prompt 装配
- **行动**：head 段（agent_role / available_skills 等）保前缀稳定（Anthropic cache 命中）+ tail 段（最近 cycle case）保最新；中间删除模式不适合 cache

### R20：SocietyOfMindAgent 边界透明
- **来源**：autogen
- **影响**：debate subgraph 边界
- **行动**：把 4 agent + debate + verdict 打包成单一 agent 接口对外暴露，降低与 LangGraph 全量状态的耦合

### R21：git 谱系 + 离线版本控制
- **来源**：EvoSkill ProgramManager
- **影响**：spec 018 的 ExperienceRule 版本管理
- **行动**：每次规则演化提交到 `experience/rule-<name>/iter-N` 分支；frontier 用 tag；`rule.yaml` 含 parent_iter / score / created_at；零额外存储

### R22：history-clear-on-improve 精确触发条件
- **来源**：EvoSkill `update_frontier()` 返回 True 时清空
- **影响**：spec 018 reflection 输入选择
- **行动**：当且仅当新 pattern 进入 frontier 且超越当前最优时，触发 case "consumed"；保留 frontier 但未超越的 case 不清空

### R23：5 阶段流水替代同步反思
- **来源**：SkillClaw（最完整）
- **影响**：spec 018 的反思架构
- **行动**：Summarize（轨迹压缩）→ Session Judge（多维评分）→ Aggregate（按 regime 分组）→ Execute（LLM 决策 improve/create/skip）→ Skill Verifier（四项准入门控）

---

## Phase 2 总结

研究最重要结论 —— **spec 018 算法路线决策树**：

```
本项目部署形态：本机无 GPU + 客观 PnL 信号 + Anthropic prompt cache + LangGraph 4 agent
  ❌ OpenClaw-RL GRPO 真 RL → 排除（GPU 限制）
  ✅ SkillClaw 5 阶段批处理 → 主框架（R23）
  ✅ Hermes GEPA + Reflective Mutation → 嵌入到 reflect 流程（R13、R16）
  ✅ EvoSkill 纯数值 frontier → 用 PnL 替代 accuracy（R13）
  ✅ MetaClaw IDF 混合检索 → 替代当前无检索（R15）
  ✅ EvoSkills IVE 失败分类 → 亏损归因 prompt（R17）
  ✅ skill-evolution 5 信号 FSM → 替代单维 maturity float（R14）
  ✅ autogen HeadAndTail context + SocietyOfMind → spec 017 装配 + 边界透明（R19、R20）
```

**spec 018 brainstorm 已具备完整证据基础**。SC-R7 满足。
