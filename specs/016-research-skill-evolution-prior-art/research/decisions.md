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

---

## D-EV-01：进化算法主线 = SkillClaw 5 阶段 + Hermes Reflective Mutation + EvoSkill 数值 frontier

**状态**：Accepted（Phase 2，2026-05-08）

**背景**：8 个项目对比显示进化算法可分 3 类：
- RL 真路线（OpenClaw-RL GRPO）—— 需 8 GPU，本项目部署形态不可行
- 监督 LLM-as-judge（SkillClaw 5 阶段、Hermes GEPA、MetaClaw 双循环）—— 可行
- 纯数值 frontier（EvoSkill）—— 客观信号驱动，最契合 PnL 场景

cryptotrader-ai 拥有最干净的客观信号 PnL，且无 GPU。

**决策**：spec 018 主线算法 = **SkillClaw 5 阶段流水**作为整体框架 + **Hermes Reflective Mutation** 作为提示变异方式 + **EvoSkill 客观 frontier** 作为评估准则。具体：
1. **Summarize 阶段**：把每个 trading cycle 的因果链（数据 → 4 agent → debate → verdict → execute → realized PnL）压缩为 8-15 句摘要
2. **Session Judge 阶段**：用客观 PnL 指标（不是 LLM-as-judge）做多维加权评分（win_rate / R:R / drawdown 影响）
3. **Aggregate 阶段**：按 regime_tags 分组
4. **Execute 阶段**：LLM 决策 improve/create/skip，输入含失败 cases + 自反反馈（"为什么这条规则失败"）+ 现有规则枚举
5. **Skill Verifier 阶段**：四项准入门控（empirical rate / 跨 regime 一致性 / 不与 forbidden 冲突 / 通过 pairwise 比较）

**后果**：
- 完全规避 GPU 需求
- LLM 调用集中在离线批处理（不阻塞交易路径）
- 客观信号（PnL）替代主观 LLM-as-judge，减少 hallucination 风险
- 5 阶段每阶段可独立测试 / 调优

---

## D-EV-02：双目标 Pareto frontier（win_rate × confidence）

**状态**：Accepted（Phase 2）

**背景**：Hermes GEPA 不只保留全局最优，而是保留"在至少一个评估实例上达到最高分"的 Pareto 候选集合。EvoSkill 用单维度 accuracy 替换 frontier。但实际交易场景是多目标的 —— 高 win_rate 低 confidence 的规则可能漂移；高 confidence 低 win_rate 的可能 overfit。

**决策**：spec 018 frontier = `win_rate × confidence` 双目标 Pareto。规则进 frontier 当且仅当**在 win_rate 或 confidence 至少一个维度上达到当前最高**。新规则替换 frontier 旧规则当且仅当**两个维度都不劣于**或**一个维度严格优于**。

**后果**：
- 防止单维度优化损害其他维度
- 单维度极优规则保留作为"特化策略"
- 实现复杂度比单维度高一些（O(N²) 比较），但 N 通常 < 100

---

## D-EV-03：5 信号成熟度 FSM 替换 `maturity: float`

**状态**：Accepted（Phase 2）

**背景**：当前 spec 014 的 `ExperienceRule.maturity` 是 0~1 float，缺少**精确的状态转换触发条件**。skill-evolution 提供了 5 个客观信号驱动的 FSM。

**决策**：把 `maturity: float` 改为 `maturity: Literal["draft", "tested", "stable", "clean", "mature"]`，状态转换条件：
- **draft → tested**：production-tested 信号（真实交易成功 ≥3 次）
- **tested → stable**：stable 信号（最近 5 次 cycle 中或 3 天内无 reflect 修改，取先满足者）
- **stable → clean**：well-structured 信号（frontmatter 有效 + body ≤300 行）
- **clean → mature**：clean 信号（无硬编码常量警告）+ self-contained（依赖声明完整）

**后果**：
- 状态可机器验证（无主观浮点判定）
- skill-evolution 的"近期被 reflect 修改则降档"可作为 stable 的撤销条件
- 与现有 PnL FSM 互补：PnL 驱动 + "无修改时长"驱动两路并行

---

## D-EV-04：失败分类 prompt 模板（实现失败 vs 根本失败）

**状态**：Accepted（Phase 2）

**背景**：EvoSkills IVE 用 5 诊断问题区分"实现失败（可重试）vs 根本失败（永久归档）"。当前 cryptotrader-ai 的亏损归因不做这种区分 —— 所有亏损都进入下一次 reflection，可能让"短期市场噪声造成的小亏"误导规则修改。

**决策**：spec 018 的 reflect.py 加**亏损归因 prompt 模板**，含 5 诊断问题（如：是否同 regime 下其他规则也亏？是否进出场价格在合理区间？是否撞了停损？是否符合该规则的 invalidation 条件？规模是否过大）。LLM 输出 `failure_type: implementation|fundamental|noise`。**三连续 fundamental 失败 → 永久归档该规则**。

**后果**：
- 噪声亏损不会触发不必要的规则修改
- 规则可控演化路径（implementation → 修小细节；fundamental → 整体重写或归档）
- 防止过度反应

---

## D-DS-01：技能 schema 完整字段清单（综合 spec 014 + Phase 2 借鉴）

**状态**：Accepted（Phase 2，schema 定型；migration 在 spec 018）

**背景**：综合 D-PA-06（spec 017 已加 `regime_tags` / `triggers_keywords` / `level`）+ D-MW-01（已加 `importance` / `access_count` / `last_accessed_at`）+ Phase 2 新发现，确定最终 schema。

**决策**：`agent_skills/<agent>/<pattern>/SKILL.md` frontmatter 完整字段：
```yaml
name: <agent>::<pattern_name>      # spec 014 已有
description: <one-liner>             # spec 014 已有
version: <semver>                    # 新增（D-EV-01）
parent: <parent_iter>                # 新增（EvoSkill 谱系）
maturity: draft|tested|stable|clean|mature  # 替换 float（D-EV-03）
regime_tags: [trending, choppy, ...] # spec 017（D-PA-06）
triggers_keywords: [...]             # spec 017（D-PA-06）
level: metadata|body|reference       # spec 017（D-PA-06）
importance: 0.0-1.0                  # spec 017（D-MW-01）
access_count: int                    # spec 017（D-MW-01）
last_accessed_at: ISO8601            # spec 017（D-MW-01）
confidence: 0.0-1.0                  # 新增（MetaClaw 启发，由 reflection 设定）
allowed_actions: [long, short, close, hold]  # 新增（EvoSkills 启发）
created_at: ISO8601                  # 新增
forbidden: bool                      # spec 014 已有（forbidden zone）
applied_count: int                   # 新增（实际被 verdict 引用次数）
pnl_track: list of {ts, pnl, regime} # spec 014 已有
```

**后果**：
- 一次性定型避免反复 migration
- 现有 27 个 SKILL.md 文件需要一次性 migration（spec 018 plan 包含）
- Schema 验证可机器化（pydantic / dataclass）

---

## D-RT-01：检索算法 = regime_tags 预过滤 + IDF 加权混合

**状态**：Accepted（Phase 2）

**背景**：当前 spec 014 完全没有检索（所有技能始终全加载）。Phase 2 调研显示 MetaClaw 的 IDF 混合检索最完整，EvoSkills 的声明式路由最低延迟。两者结合更优。

**决策**：spec 018 检索分两层：
1. **第一层（零延迟）**：`regime_tags` 预过滤 —— current_regime 在 skill 的 `regime_tags` 列表中才纳入候选
2. **第二层（IDF 加权）**：候选集打分 = `(关键词IDF + embedding余弦 + importance + recency_bonus) × confidence × maturity_weight`
   - `recency_bonus = exp(-(now - last_accessed_at) / 7days)`
   - `maturity_weight = {draft: 0.3, tested: 0.6, stable: 0.9, clean: 1.0, mature: 1.0}`
   - 取 top-k（默认 5）注入

**后果**：
- 第一层覆盖 80%+ 场景（regime 变化少）
- 第二层处理跨 regime 通用规则
- embedding 用本地模型（spec 018 选 sentence-transformers all-MiniLM 或 Qwen3-0.6B）

---

## D-EVAL-01：评估指标 = 客观 PnL（非 LLM-as-judge）

**状态**：Accepted（Phase 2）

**背景**：8 项目里 EvoSkill 是唯一用纯数值评估的，其他都用 LLM-as-judge 或半客观。但本项目有最干净的客观信号 = realized PnL。

**决策**：spec 018 的规则评分函数 = **PnL 直接驱动**，不用 LLM-as-judge：
- `win_rate = wins / total`
- `avg_R_R = mean(target - entry) / mean(stop - entry)`
- `consistency = 1 - std(returns) / mean(returns)`（变异系数倒数）
- `coverage = applied_count / cycles_in_regime`（命中率）

frontier 比较函数：fitness = win_rate × R_R_factor × consistency × coverage_normalizer

**后果**：
- 评估完全机器化，无 LLM 调用
- 防 LLM-as-judge 漂移
- 可与回测系统直接联通（spec 015 已有的 backtest 模块复用）

---

## D-AS-01：debate subgraph 透明边界（SocietyOfMindAgent 模式）

**状态**：Accepted（Phase 2）

**背景**：autogen 的 `SocietyOfMindAgent` 把内嵌 Team 对外呈现为单一 agent 接口。cryptotrader-ai 当前 4 agent + debate + verdict 在 LangGraph 中是 12 个节点，每个都看到全量状态，导致：
- debate 节点修改的 state 影响下游不易追踪
- 全量状态 diff 难做（内部消息进出多）

**决策**：spec 018 把 4 agent + debate + verdict 抽象为**单一 ContextAggregator 节点**对外暴露：
- 输入：snapshot + portfolio + history
- 输出：verdict（含 thesis / invalidation / target / applied:）
- 内部 debate 状态 **不暴露**到外部 LangGraph state；external state 只看 verdict

**后果**：
- LangGraph 上下游节点对内部 debate 不感知（边界更清晰）
- debate 内部可独立演进（spec 018 / 019）不影响其他节点
- 测试更易（mock 单一接口）
- 实现：用 LangGraph 的 subgraph 抽象 + `interrupt()` 隔离

---

## D-ENG-01：离线 reflect daemon

**状态**：Accepted（Phase 2）

**背景**：当前 `learning/reflect.py` 在 cycle 内同步执行（`loop.create_task` 异步触发但仍在 trading loop event loop 内）。SkillClaw / Hermes / MetaClaw 都用独立进程做离线演化。

**决策**：spec 018 把 reflect 拆为独立 daemon：
- Trading cycle 只写 `agent_memory/cases/<commit>.md`（已有）
- reflect daemon 是**独立进程**，每 N 分钟（可配 default 30）批量读 cases，跑 5 阶段流水，写回 `agent_memory/<agent>/patterns/`
- 通过文件锁 + `last_processed_commit_hash` 标记保证不重复处理

**后果**：
- 交易路径完全脱离 LLM 调用
- reflect 可消耗任意时长（不受 cycle interval 约束）
- 部署稍复杂（多一个进程；但 launchd / systemd 都可管）

---

## D-ENG-02：git 谱系作为 ExperienceRule 版本控制

**状态**：Accepted（Phase 2）

**背景**：EvoSkill 用 `program/iter-N` 分支 + `frontier/*` 标签 + `program.yaml` 谱系字段做版本控制，零额外存储成本。本项目当前对 ExperienceRule 没版本管理（每次 reflect 直接覆盖磁盘文件）。

**决策**：spec 018 给每条 ExperienceRule 维护独立 git 分支：
- `experience/rule-<name>/iter-N` 分支保存每代版本
- `experience/rule-<name>/frontier-N` 标签标记 frontier 成员
- `<rule>.md` frontmatter 含 `parent: experience/rule-<name>/iter-N-1` 谱系字段
- 替换 / 回滚 / 审计都通过 git 操作

**后果**：
- 完全审计可追溯
- 零额外存储（git 增量）
- 回滚 = `git reset --hard <previous-iter-tag>`
- 实现：reflect daemon 写完后 `git commit` + `git tag`，全自动

---

## D-PROC-02：spec 016 研究完成确认

**状态**：✅ Accepted（2026-05-08）

**背景**：Phase 2 全部 8 项目深读完成，5,137 行项目文档，新增 ADR D-EV-01..04 / D-DS-01 / D-RT-01 / D-EVAL-01 / D-AS-01 / D-ENG-01..02 共 11 条。SC-R7 显式条件全部满足：
- ✅ SC-R1 ~ SC-R5 满足（8 项目文档 + frontmatter license + 矩阵 + synthesis ≥10 条建议）
- ✅ synthesis.md 8 角度都有覆盖（Phase 1 + Phase 2 6 章节）
- ⏳ 等用户审核确认

**决策**：spec 016 研究阶段已完成，Phase 1 + Phase 2 全交付。下一步：
1. user 审核 Phase 2 合并产物
2. 通过后 commit
3. 启动 spec 017 brainstorm 或 spec 018 brainstorm（按用户优先级）

**后果**：
- spec 018 brainstorm 解锁
- spec 017 brainstorm 可独立启动（Phase 1 已解锁）
- 8 项目本地 git clone 可在确认后归档/删除（产物文档已含必要源码引用）
