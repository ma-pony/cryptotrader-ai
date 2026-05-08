---
name: SkillClaw
url: https://github.com/AMAP-ML/SkillClaw
license: Apache-2.0
tier: 1
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# SkillClaw — AMAP-ML

## 架构概览

SkillClaw 是由阿里巴巴高德地图 DreamX 团队（AMAP-ML）于 2026 年 4 月 10 日开源的**集体技能演化框架**，配套论文 [arXiv 2604.08377](https://arxiv.org/abs/2604.08377)「SkillClaw: Let Skills Evolve Collectively with Agentic Evolver」。核心问题是：LLM Agent（如 OpenClaw/Codex/Hermes）依赖可复用的 SKILL.md 技能文件执行复杂任务，但这些技能在部署后几乎保持静态；不同用户反复踩到同样的坑，却没有任何机制把这些跨用户的失败经验回写到技能本身。

SkillClaw 的解法分两层：**Client Proxy**（本地 API 代理）负责拦截所有 Agent 请求、记录会话轨迹、管理本地技能库；**Evolve Server**（可选）负责从共享存储中读取会话数据、运行自治 Evolver 生成技能更新并写回。两个组件通过同一存储层（本地文件系统 / S3 / 阿里云 OSS）和同一技能格式（SKILL.md）协同，从而把一个用户调试出来的经验自动传播给所有使用同一共享组的 Agent 实例。

技术栈为 Python（FastAPI + uvicorn 对外暴露 `/v1/chat/completions` 和 `/v1/messages` 两个兼容端点），依赖 `click`、`pyyaml`、`httpx`、`tiktoken`。安装后通过 `skillclaw setup` / `skillclaw start --daemon` 启动本地代理，对上游 Agent 零侵入。2026 年 4 月新增 `skillclaw dashboard sync/serve` 用于可视化检查技能库、版本历史和会话追踪。

SkillClaw 的技能演化在六轮迭代后可带来约 88% 的性能提升（WildClawBench 测试），且兼容 Hermes、Claude Code、OpenClaw、Codex、QwenPaw 等所有 OpenAI 兼容 API 的 Agent 运行时。

---

## Prompt Assembly（Phase 1）

### 技能文件格式（SKILL.md）

SkillClaw 沿用整个 \*Claw 生态的标准 SKILL.md 格式：

```
---
name: <技能名>
description: <何时使用本技能（触发描述）>
version: x.y.z
triggers:
  - <关键词或语义描述>
---

# 正文：技能的完整操作指令
（Markdown 格式，包含步骤、工具调用参数模板、注意事项等）
```

前置 YAML 块是**元数据层**（Level 1），正文是**指令层**（Level 2）。这一分层设计是整个 prompt 注入机制的核心。

### 两阶段注入（Lazy-Load 模式）

Agent 启动时，Proxy 读取所有已安装技能的 YAML 前置块，将 `name` + `description` 组成一个紧凑的技能目录注入 **system prompt** 的固定位置——此时不包含指令正文，仅让模型知道「有哪些技能、何时调用」。代币消耗极低。

当模型根据 description/triggers 判断当前任务匹配某技能时，Proxy 把对应 SKILL.md 的**完整正文**注入到当前上下文窗口——作为追加的 system_prompt 段或特殊 user 消息（取决于上游 Agent 的格式）。这一步等价于一次「按需扩展上下文」，而非在所有请求中都携带全量指令。

### Proxy 的拦截与 prompt 改写流程

Client Proxy 暴露与 OpenAI API 完全兼容的端点（`/v1/chat/completions` / `/v1/messages`），Agent 将其作为 base_url。每次请求到达时，Proxy：

1. **解析请求**：提取 `messages` 数组中的 system prompt 和 user 消息；
2. **技能匹配**：与本地技能库中所有技能的 description/triggers 做语义匹配；
3. **注入技能内容**：若命中，将匹配技能的完整 SKILL.md 正文插入 messages（通常拼接在 system 消息末尾，或作为独立的 system 角色消息）；
4. **透传上游**：将改写后的完整 messages 转发给真实 LLM API，结果原样返回给 Agent；
5. **记录会话**：把原始请求 + 工具调用 + 中间反馈 + 最终回答全链路存入会话存储。

### 注入质量门控（push_min_injections）

SkillClaw 设有注入次数阈值（默认 `push_min_injections=5`）：新技能必须累积至少 5 次实际注入记录才能被纳入共享演化池。这防止了低质量、从未真正触发过的技能污染集体经验库。PRM（Process Reward Model）评分作为可选的质量评估层，在 `skillclaw setup` 阶段配置。

---

## Memory ↔ Skill（Phase 1 lite）

### 会话轨迹的存储格式

每次 Agent 会话结束后，Proxy 将整个交互记录为**结构化因果链**：

```
用户 prompt → Agent 动作（含工具调用参数）→ 中间反馈（工具返回值 / 错误消息 / 用户回应）→ 最终答案
```

保留完整中间过程是 SkillClaw 与一般 chat history 记录的关键区别：技能层面的失败（错误参数格式、遗漏验证步骤、工具调用顺序错误）只能从动作-反馈链中诊断，最终答案不呈现这些信息。

存储层支持本地文件系统、S3 或阿里云 OSS，格式为结构化 JSON 文件，每个会话一个条目，包含会话 ID、时间戳、关联技能名、轨迹数据。

### 技能库本地存储

技能文件以目录形式存储在本地（Hermes 集成默认路径 `~/.hermes/skills/`，其他 Agent 在 `skillclaw setup` 时配置）。每个技能目录包含：

- `SKILL.md`（必需）：元数据 + 指令
- `scripts/`（可选）：辅助脚本
- `references/`（可选）：按需加载的额外文档
- `assets/`（可选）：模板、配置文件

版本历史通过存储层快照维护；`skillclaw restore` 可回滚到上一个干净快照（防止 Evolver 破坏技能状态）。

### Task Ledger（跨会话目标记忆）

SkillClaw 内置**任务账本（Task Ledger）**，在会话之间持久化追踪目标、决策和结果。Task Ledger 不是 RAG 知识库，而是面向单一 Agent 实例的结构化状态日志：每次会话开始时读入相关历史目标与决策，会话结束时写回新的决策和结果。这使得 Agent 在多次会话后能感知「我之前在哪个目标上遇到了什么问题」，而不依赖 LLM 上下文窗口保留记忆。

### 共享存储与技能同步

多用户场景下，各 Client Proxy 指向同一共享存储。Evolve Server 周期性扫描存储中的会话组（按技能名分组），运行 Agentic Evolver 后将更新后的 SKILL.md 写回共享存储，各客户端的下次 `skillclaw dashboard sync` 或自动同步周期将拉取最新版本。内存中没有全局共享状态——所有协调通过文件系统（或对象存储）的读写完成。

---

## Phase 2 Placeholders

以下各节延迟至 Phase 2 研究，此处有意留空：

- **演化算法**（Evolution Algorithm）
- **技能数据结构细节**（Skill Data Structure）
- **检索机制**（Retrieval Mechanism）
- **评估体系**（Evaluation / WildClawBench）
- **Agent ↔ Skill 边界**（Agent ↔ Skill Boundary）
- **工程实现细节**（Engineering Implementation Details：proxy 内部模块、Evolver Harness 代码、PRM 评分实现）

---

## 借鉴建议（Phase 1 — prompt + memory wiring）

以下建议仅聚焦 prompt 组装与记忆接线，不涉及演化逻辑（Phase 2 才处理）。

### 1. 两阶段技能注入：元数据目录常驻 system prompt，正文按需注入

**来源**：SkillClaw/OpenClaw SKILL.md 格式规范（YAML 前置块 + Markdown 正文的分离设计）

**思路**：crypto-trader-ai 的四个 Agent（`agents/tech.py`、`agents/chain.py`、`agents/news.py`、`agents/macro.py`）目前在 `create_llm()` 调用时直接构造完整的 system prompt 字符串，随着技能/角色描述增多会持续膨胀上下文。可借鉴两阶段设计：将每个 Agent 的「能力目录」（名称 + 触发描述）压成一段固定的轻量 system 前缀（<200 tokens），只在检测到对应任务类型时才将完整的分析框架/指令段追加到 messages。

**映射**：`agents/base.py` 的 `create_llm()` 或 `nodes/agents.py` 中的提示构建逻辑；技能目录字符串可从外部 YAML 文件读取（对应 spec 017 的提示外化方向）。

### 2. 结构化因果链会话记录（prompt → action → feedback → answer）

**来源**：SkillClaw arXiv 2604.08377，§3 「Session Recording」

**思路**：cryptotrader-ai 当前的 `journal/store.py` 记录最终的 `AgentAnalysis` 结果，但丢失了中间工具调用路径。SkillClaw 的会话格式（user_prompt / tool_calls / intermediate_feedback / final_answer 四段结构化 JSON）是后续经验反思的基础——只有保留这一链路，才能诊断「哪一步工具调用参数格式错了」。可在 `nodes/agents.py` 的 Agent 调用包装层加一个 `SessionRecorder`，将 LangGraph 的消息序列序列化为此格式写入 SQLite（复用 `db.py` 的异步会话工厂）。

**映射**：`nodes/agents.py` 中的 agent invoke 封装 → `learning/reflect.py` 的输入侧；`journal/store.py` 扩展一张 `agent_sessions` 表。

### 3. 技能版本快照 + 自动回滚

**来源**：SkillClaw `skillclaw restore` 机制（dashboard sync 附带的快照管理）

**思路**：cryptotrader-ai 的 `learning/reflect.py` 在每次反思后会覆写 `ExperienceMemory`，若 Evolver 产出劣质规则（如过拟合小样本），当前没有快照回滚机制。可在 `learning/reflect.py` 的 `_merge_memories()` 之前将旧版本序列化存储（一个 `experience_snapshots` 表，保留最近 N 个版本），并提供 `arena experience restore <snapshot_id>` CLI 命令。

**映射**：`learning/reflect.py` → `journal/store.py`（新增 snapshot 写入）；`cli/main.py` 增加 `restore` 子命令。

### 4. 注入次数门控替代纯时间衰减

**来源**：SkillClaw `push_min_injections=5` 质量门控设计

**思路**：cryptotrader-ai 的经验规则成熟度（`maturity`）目前仅依赖 `trade_count` 阈值（`learning/reflect.py` 中 `_assign_maturity()`）。SkillClaw 的「注入次数」门控更贴近实际使用：一条规则被实际检索并注入到 Agent 决策的次数，比规则的产生轮数更能代表其可信度。可在 `ExperienceRule` 上新增 `injection_count` 字段，`verbal_reinforcement()` 每次注入一条规则时递增计数，`_assign_maturity()` 同时考量 `trade_count` 和 `injection_count`。

**映射**：`models.py`（`ExperienceRule` 新增字段）→ `learning/verbal.py`（`search_by_regime()` 返回后递增计数）→ `learning/reflect.py`（`_assign_maturity()` 加入注入次数权重）。

### 5. 技能文件外化到目录（替代嵌入式字符串）

**来源**：SkillClaw/OpenClaw SKILL.md 目录约定（`~/.hermes/skills/<skill-name>/SKILL.md`）

**思路**：当前每个 Agent 的角色描述和分析框架以 f-string 硬编码在 Python 文件中（`agents/tech.py` 等），版本控制颗粒度粗，且修改需要改代码。借鉴 SKILL.md 的目录布局，可将每个 Agent 的 prompt 模板外化为 `agents/skills/<agent_name>/SKILL.md`，在 `create_llm()` 或 Agent 构造时动态读取，支持不重新部署修改 prompt。这与 spec 017「prompt externalization」直接对应。

**映射**：`agents/base.py` 新增 `load_skill_prompt(agent_name)` 工具函数；`agents/{tech,chain,news,macro}.py` 的 system prompt 字符串替换为文件读取调用。

---

## 注意事项 / 待深入问题

**Phase 1 已识别，Phase 2 需要深入的点：**

1. **proxy 模块文件结构尚未确认**：直接 fetch GitHub 受网络限制，`skillclaw/` 子目录下的具体模块（proxy.py / api.py / skill_manager.py / formatter.py / session_recorder.py 等）尚未通过源码核实。Phase 2 应克隆仓库后逐文件分析。

2. **PRM 评分的具体集成方式**：setup 阶段可配置 PRM，但与 Evolver 的协作逻辑（评分时机、分数如何影响 Refine/Create/Skip 决策）属于演化算法范畴，留 Phase 2。

3. **Task Ledger 的持久化格式**：仅从文档层面确认「跨会话追踪目标/决策/结果」，JSON schema 及与 SKILL.md 的关联关系尚不清晰，Phase 2 需查源码。

4. **License 核实**：README 搜索结果未明确列出 LICENSE 文件内容；多个第三方文章将其描述为开源，但 Apache-2.0 来源于 AMAP-ML 组织的其他仓库惯例（如 RL3DEdit / Thinking-with-Map），Phase 2 应直接读取 LICENSE 文件。

5. **语义技能匹配的实现方式**：注入时是关键词匹配、BM25 还是 embedding 相似度？影响 cryptotrader-ai 借鉴时的检索架构选择。

**已确认可直接用于 spec 017 的设计决策：**
- SKILL.md 的两阶段加载是成熟实践，可直接映射到 prompt 外化需求。
- 会话记录的因果链格式是 `learning/reflect.py` 输入侧的改进方向。
- 注入计数作为规则成熟度指标比纯交易次数更合理。
