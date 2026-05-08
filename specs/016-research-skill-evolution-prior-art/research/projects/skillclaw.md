---
name: SkillClaw
url: https://github.com/AMAP-ML/SkillClaw
license: MIT
tier: 1
last_accessed: 2026-05-07
phase_1_complete: true
phase_2_complete: true
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

## 演化算法（Phase 2）

### 三阶段流水线总览

Evolve Server 的核心是一条**批处理式 LLM 演化流水线**，由 `evolve_server/pipeline/` 下的四个模块串行执行：

```
[Session Store] → Drain All → Summarize → Session Judge → Aggregate → Execute → Verify → Publish
```

每 10 分钟（`interval_seconds=600`，可配置）触发一次；若队列中有任何会话，流水线即启动——**没有最低会话数门槛**，逻辑为 `if sessions: run_pipeline()`。

---

#### Stage 1：Summarize（摘要化）

`pipeline/summarizer.py` 的 `summarize_sessions_parallel()` 并行处理所有已排队会话。每个会话经过两步压缩：

1. **轨迹提取**：从 `turns[]` 中提取工具调用序列、每步 PRM 分数、中间错误，生成 `_trajectory` 字段（结构化文本，保留行号 + 工具名 + 返回值摘要）。
2. **LLM 摘要**：以 `_trajectory` 为输入，调用 LLM 生成 8–15 句的 `_summary`，聚焦目标、关键转折点（turning points）、工具使用模式、最终结果质量。

每个会话处理后产出附加字段：`_trajectory`、`_summary`、`_skills_referenced`（从工具调用中提取的技能路径列表）、`_avg_prm`（平均 PRM 分）、`_has_tool_errors`（布尔值）。

---

#### Stage 2：Session Judge（质量过滤）

`pipeline/session_judge.py` 在 Summarize 之后、Aggregate 之前运行，过滤低质量会话。若会话已有以下任意可信分数则跳过评判：
- `_judge_scores.overall_score`（已有判断结果）
- `benchmark.overall_score`（基准测试分数）
- `aggregate.mean_score`（聚合分数）

未通过快速检查的会话提交给 LLM judge，按四个维度评分（0.0–1.0）：

| 维度 | 权重 | 说明 |
|------|------|------|
| 任务完成度 | 55% | 目标是否实现 |
| 响应质量 | 30% | 正确性与完整性 |
| 效率 | 5% | 避免无效重试 |
| 工具使用 | 10% | 工具选择合理性 |

评分原则：以轨迹为 ground truth，实际产出物（文件写入、代码运行结果）作为任务完成的强证据；框架启动噪音不扣分。

---

#### Stage 3：Aggregate（按技能分组）

`pipeline/aggregation.py` 将会话按 `_skills_referenced` 字段分组为「技能桶」。一个会话可能同时出现在多个技能桶中（多技能引用）。无技能引用的会话归入 `NO_SKILL_KEY` 特殊桶，等待可能的新技能创建。

---

#### Stage 4：Execute（技能更新决策）

`pipeline/execution.py` 对每个技能桶独立运行，决定四种操作之一：

| 操作 | 触发条件 | 语义 |
|------|----------|------|
| `improve_skill` | 会话暴露出缺失指导或错误信息 | 针对性修改现有技能正文 |
| `optimize_description` | 触发描述导致错误任务匹配 | 仅重写 description，正文不变 |
| `create_skill` | `NO_SKILL_KEY` 桶中有重复模式 | 创建全新技能目录 |
| `skip` | 证据不足，或失败源于 Agent/环境问题 | 保持不变 |

LLM 决策输入包含：当前 SKILL.md 全文、所有历史版本（`history/v*.md`）、会话轨迹、PRM 均值、成功/失败计数、稳定性指标。LLM 返回结构化 JSON：`{"action": ..., "rationale": ..., "skill": {...}}`。

**关键约束**：LLM 被指示「保守修改，以当前技能为 source of truth，目标做针对性编辑而非全面重写；每次修改必须引用具体 session ID、PRM 分数或错误模式作为证据」。

---

#### Stage 5：Skill Verifier（发布前验收门）

`pipeline/skill_verifier.py` 以低温度（0.1）对每个候选技能运行四项准入检查，**全部通过才放行**（默认阈值 0.75）：

1. **证据接地性**：修改内容是否有会话数据支撑
2. **保留现有价值**：是否丢弃了有用的环境特定细节
3. **具体可复用性**：内容是否可操作，而非泛化建议
4. **立即可发布性**：是否连贯完整

未通过验收的技能被阻止上传；`workflow` 引擎在合并冲突时回退到 LLM 自动合并（`execute_merge()`），合并失败则保留传入版本。

---

### 两种演化引擎

SkillClaw 提供两种可互换引擎，通过 `--engine` 参数选择：

**Workflow 引擎**（默认，`evolve_server/engines/workflow.py`）：固定的三阶段 LLM 流水线，每批最多处理 20 个技能（`evolve_batch_size=20`），每个技能桶的会话数上限 8 条（verifier evidence）。适合生产部署。

**Agent 引擎**（`evolve_server/engines/agent.py`）：在隔离 Workspace 中启动一个 OpenClaw 实例，直接读写技能文件。Agent 能自由浏览 `history/` 目录、对比历史版本、决定是否跳过。相当于一个会写代码的「技能维护工程师」，超时上限 600 秒。适合复杂技能的深度重构。

两种引擎共享同一 Session Store → Summarizer → Aggregator 前置管道，以及 Skill Verifier + Publisher 后置管道，仅执行阶段不同。

---

### 演化策略与 LLM 参数

默认策略 `dynamic_edit_conservative`（temperature=0.4），禁止完全重写（`reject_rewrite=True`）。发布模式：
- `direct`（直接发布，跳过客户端二次验证）
- `validated`（需至少 1 次客户端 replay 验证通过 + 分数 ≥ 0.75 + 拒绝次数 ≤ 1）

默认 LLM 模型：workflow 用 `gpt-4o`，agent 用 `gpt-5.4`；同时支持 Anthropic、Google Gemini、Ollama。

---

## 技能数据结构（Phase 2）

### SKILL.md 完整 Schema

经源码核实的 YAML frontmatter 全字段（`skillclaw/skill_manager.py`）：

```yaml
---
# 必需字段
name: <slug，格式 ^[a-z][a-z0-9-]{1,63}$>
description: <触发描述，包含适用场景与排除场景>

# 可选字段
version: x.y.z                          # 由 Evolve Server 维护
category: general                        # general | coding | research | data_analysis |
                                         # security | communication | automation |
                                         # agentic | productivity | common_mistakes
metadata:
  openclaw: { ... }                      # OpenClaw 运行时元数据
  skillclaw: { ... }                     # SkillClaw 框架元数据
disable-model-invocation: false          # true 则从技能目录中隐藏但保留加载
---
```

frontmatter 之外的全部 Markdown 正文为「指令层」，可包含步骤、工具调用参数模板、注意事项、环境特定配置。round-trip 写入时，未知字段通过 `_extra_frontmatter` 原样保留。

### 技能目录布局

```
{group_id}/skills/<skill-name>/
    SKILL.md           # 主文件（元数据 + 指令）
    history/
        v1.md          # 版本 N 对应的 SKILL.md 快照
        v1_evidence.md # 版本 N 对应的决策证据（session IDs、PRM 分数、修改计划）
        v2.md
        v2_evidence.md
        ...            # history 上限 20 条版本记录
    scripts/           # 可选辅助脚本
    references/        # 可选按需加载文档
    assets/            # 可选模板/配置文件
```

`history/` 是演化引擎的核心审计轨迹，Agent 引擎在每次 `improve_skill` 操作前**必须读取所有历史版本**（硬性要求），防止重蹈已修复的老问题。

### 会话记录精确格式

`skillclaw/api_server.py` 每一轮（turn）生成以下结构化记录：

```json
{
  "session_id": "...",
  "turn": 3,
  "timestamp": "2026-05-07 14:32:01",
  "messages": [...],
  "instruction_text": "<最后一条 user 消息>",
  "prompt_text": "<tokenized prompt>",
  "response_text": "<tokenized response>",
  "tool_calls": [
    {
      "id": "...",
      "function": { "name": "bash", "arguments": "..." },
      "result": "...",
      "error": null
    }
  ],
  "next_state": "<下一轮 user 消息或 null>",
  "_prm_score": 0.85,
  "_token_ids_prompt": [...],
  "_logprobs_prompt": [...],
  "_token_ids_response": [...],
  "_logprobs_response": [...],
  "read_skills": ["git-workflow"],
  "modified_skills": [],
  "injected_skills": ["git-workflow"]
}
```

会话级元数据（session close 时写入）：
```json
{
  "session_id": "...",
  "start_time": "...",
  "end_time": "...",
  "turns": [...],
  "_trajectory": "<结构化执行路径，含步骤号、工具调用、PRM 分>",
  "_summary": "<8-15 句 LLM 分析文本>",
  "_skills_referenced": ["git-workflow"],
  "_avg_prm": 0.72,
  "_has_tool_errors": true
}
```

### 技能注册表（Registry）元数据

Evolve Server 维护 `{prefix}evolve_skill_registry.json`，每个技能条目：

```json
{
  "skill-name": {
    "skill_id": "a3f9b2c1d4e5",
    "version": 4,
    "content_sha": "sha256:...",
    "history": [
      {
        "version": 3,
        "content_sha": "sha256:...",
        "timestamp": "...",
        "action": "improve_skill"
      }
    ]
  }
}
```

`skill_id` 为 `SHA-256(skill_name)` 的前 12 位十六进制字符，确定性生成，跨客户端不变。

### Task Ledger（任务账本）

Task Ledger 是 SkillClaw 内置的**跨会话目标持久化结构**，源码中作为 `skillclaw/runtime_state.py` 的一部分维护。核心语义是：在会话开始时读入「已知目标与历史决策」，会话结束时写回「新决策与结果」。其字段从文档层面确认包含目标（goal）、决策（decision）、结果（outcome）三段，与技能名关联。Task Ledger 与 SKILL.md 不同——SKILL.md 是共享知识，Task Ledger 是单个 Agent 实例的状态日志，不会被 Evolver 处理或跨用户共享。

---

## 检索机制（Phase 2）

### 两阶段注入中的技能匹配细节

技能匹配发生在 `skillclaw/skill_manager.py` 的 `_inject_skills()` 调用链中，支持两种模式：

**Template 模式（默认，零延迟）**：
- 不做语义计算，直接按**有效性分数（effectiveness score）降序排列**所有技能
- `effectiveness = positive_count / inject_count`（新技能默认 0.5）
- 返回 top-k 技能（k 由上下文窗口 token 预算决定）
- 适合技能库小（< 20 个）或实时性要求高的场景

**Embedding 模式（语义检索）**：
- 使用 SentenceTransformer（默认 `Qwen/Qwen3-Embedding-0.6B`）对当前对话上下文与技能 description 做相似度计算
- 最终排分公式：`score = similarity × (0.3 + 0.7 × effectiveness)`（相似度权重 70%，质量权重 30%）
- 对候分 > 0.9 的技能对做去重，防止相似技能同时注入撑爆上下文
- 从排序后的候选池依次填充注入槽，直到 token 预算耗尽

两阶段架构（目录层 + 正文层）在两种模式下通用：目录层（name + description）总是注入 system prompt，只有命中技能的完整正文才追加注入。

### push_min_injections 的精确语义（源码核实版）

Phase 1 文档中对 `push_min_injections=5` 的描述需要修正。源码中的准确逻辑（`skill_hub.py`）是：

- `min_injections` 是**推送（push）质量门控**而非触发门控
- 规则一：`inject_count == 0`（从未注入过）的技能**绕过门控**，直接进入候选池（作为待曝光新技能）
- 规则二：`inject_count >= min_injections`（已过实习期）的技能才检查 `effectiveness >= min_effectiveness`；不满足则阻止上传

因此，`push_min_injections=5` 的语义是：**前 5 次注入是「实习期」——不管效果好坏都允许发布，用于收集初始反馈数据；第 6 次注入后开始要求效果达标才能继续共享**。这与 Phase 1 的「注入 5 次才纳入演化池」描述是两回事。

### 注入计数与反馈记录

```
record_injection(skill_names)  → 递增 inject_count，更新 last_injected_at
record_feedback(skill_names, score)  → 递增 positive_count / negative_count / neutral_count
```

每 10 次计数变化后持久化一次到 `skill_stats.json`（减少 I/O）。`effectiveness` 实时从 `positive_count / inject_count` 计算，首次注入前默认 0.5（中性）。

---

## 评估体系（Phase 2）

### WildClawBench 基准

WildClawBench（[github.com/InternLM/WildClawBench](https://github.com/InternLM/WildClawBench)）是 SkillClaw 论文（arXiv 2604.08377）的核心评估平台，包含 **60 个复杂任务**，覆盖 6 个域：

| 域 | 任务数 | 代表性任务类型 |
|----|--------|--------------|
| Productivity Flow | 10 | 信息综合、文档处理、日程调度 |
| Code Intelligence | 12 | 代码库理解、视觉谜题解题、程序生成 |
| Social Interaction | 6 | 多轮对话、API 编排 |
| Search & Retrieval | 11 | 多约束信息核对 |
| Creative Synthesis | 11 | 多模态生成（视频/音频处理） |
| Safety Alignment | 10 | 对抗健壮性、凭证意识 |

评分范围 0.00–1.00，在隔离 Docker 容器中运行，使用真实工具（浏览器、bash、文件系统、邮件、日历），执行后注入 ground truth 和评分脚本。

### 演化效果数据

论文报告的关键指标（6 轮演化后）：

- **Creative Synthesis**：相对提升 **88.41%**（最大提升域）
- **Social Interaction**：从 54.01% 升至 60.34%（第 2 轮即达稳定，说明关键瓶颈被快速解决）
- **Safety Alignment**：0 到最终轮保持稳定提升，说明安全对齐类技能有持续改进空间
- 4 个任务类别均呈现单调递增趋势（六轮无退化）

作为参照：WildClawBench 排行榜上**所有前沿基础模型得分均低于 0.55**（Claude Opus 4.6：51.6%，GPT-5.4：50.3%），SkillClaw 的技能演化在此基础上带来显著增益。

### 评估方式：离线 Replay 验证

SkillClaw 的主评估模式是**离线 replay 验证**（`validation_worker.py`），而非 A/B 测试或 online 对比：

1. 客户端（空闲时）领取 Evolve Server 下发的验证任务（一个候选技能 + 若干历史测试会话）
2. 本地复现会话（使用新技能 vs 基线技能），用 PRM Scorer 评分
3. 接受条件：`candidate_mean >= 0.75` **AND** `candidate_mean >= baseline_mean`（既要绝对分达标，又要超越基线）
4. 结果回写 `ValidationStore`，汇总后决定是否正式发布

每日验证任务数量受 `validation_max_jobs_per_day` 限制，仅在客户端空闲（`validation_idle_after_seconds`）时运行——避免干扰正常使用。

### PRM Scorer 评分机制

PRM（Process Reward Model）是 SkillClaw 的**逐步骤评估器**（`skillclaw/prm_scorer.py`），对每个 Agent 响应步骤打分：

- 并行发起 N 次（默认 N=3）LLM judge 调用，每次返回 +1（明确完成）/ -1（偏离目标）/ 0（不确定）
- 多数投票决定最终分：平局或全 0 → 默认 0.0
- 分数回写到当前 turn 的 `_prm_score` 字段，同时通过 `record_feedback()` 更新技能有效性统计

PRM 的「逐步评估」使演化引擎能区分「最终答案正确但过程低效」与「过程高效但最终失败」两种完全不同的技能改进方向。

---

## Agent ↔ Skill 边界（Phase 2）

### Client Proxy 拦截实现细节

`skillclaw/api_server.py` 暴露两个完全兼容端点：`/v1/chat/completions`（OpenAI）和 `/v1/messages`（Anthropic）。拦截流程（每次请求）：

1. **会话边界检测**：从请求头提取 `X-Session-Id`、`X-Turn-Type`（`main`/`side`）、`X-Session-Done`。无 header 的 Agent（如 Codex CLI）使用消息数量变化 + 空闲超时（默认 180 秒）做启发式边界判断。

2. **系统提示压缩**（仅 OpenClaw）：长系统提示通过一次 LLM 调用压缩后缓存，减少每次请求的 token 开销。

3. **技能注入**（仅 `main` turn）：调用 `_inject_skills()`，查询 SkillManager，将命中技能的 SKILL.md 完整正文追加到 system 消息末尾。`side` turn（内部工具调用）不注入，避免污染。

4. **上下文截断**：按 `max_context_tokens` 截断 messages 列表，优先保留 system 消息和最近若干轮，防止超出窗口。

5. **工具调用格式归一化**：将多种格式（OpenAI 结构化 `tool_calls`、Kimi XML 标记、Qwen `<tool_call>` 包装、文本内嵌工具调用）归一化为 OpenAI 标准结构，并去重（按 ID 或函数签名）。

6. **logprob 采集**：向上游请求开启 logprobs，记录 token 级别概率，为 PRM 评分提供额外信号。

7. **会话关闭**：刷新 pending 记录，等待异步 PRM 评分任务，上传到共享存储（若 sharing 已启用）。

### Dashboard Sync 协议

`skillclaw/skill_hub.py` 实现基于**清单文件（manifest）的同步协议**，manifest 存储在 `{group_id}/manifest.jsonl`（每行一个技能记录）：

```
pull first → push (with quality gate) → atomic local replacement
```

**增量拉取**（`mirror=False`）：仅下载远端 manifest 中与本地 SHA256 不同的技能，本地独有技能保留。

**镜像拉取**（`mirror=True`，默认）：备份本地目录 → 暂存下载 → 原子替换。本地存在但远端 manifest 没有的技能被删除。自动保留最近 3 个本地备份快照（`skillclaw restore` 依赖此机制）。

同步由用户显式触发（`skillclaw dashboard sync`）或由 `validation_worker` 在发布验证完成后自动调用，**没有后台定时拉取守护进程**。

### 技能所有权：共享池模式

SkillClaw 采用**完全共享池（shared pool）**模式：所有技能属于 `group_id` 共享组，不区分所有者。Registry 中没有 `user_id` 字段，版本控制基于内容 SHA256 而非用户身份。这意味着：

- 任何用户的会话都可以触发任意技能的演化
- 技能更新对组内所有成员同时生效
- 冲突解决依赖 Evolve Server 的 LLM 合并（`execute_merge()`），而非版本锁或所有者仲裁

**与 Hermes 的关键差异**：Hermes 的技能是单用户本地文件（`~/.hermes/skills/`），无共享演化机制。SkillClaw 是 Hermes 的超集——单用户模式退化为 Hermes 行为，多用户共享模式是 SkillClaw 的核心增值。

---

## 工程实现细节（Phase 2）

### 完整模块地图

```
skillclaw/                          # Client Proxy 侧
    api_server.py                   # FastAPI 端点，核心拦截逻辑
    skill_manager.py                # 技能检索、注入计数、effectiveness 计算
    skill_hub.py                    # manifest 同步、push_min_injections 门控
    prm_scorer.py                   # 多数投票式 PRM 评分
    validation_worker.py            # 客户端 replay 验证（后台空闲时运行）
    validation_store.py             # 验证结果持久化
    dashboard_server.py             # 可视化 dashboard（技能库 + 版本历史 + 会话追踪）
    dashboard_ingest.py             # dashboard 数据摄入
    dashboard_store.py              # dashboard 数据持久化
    object_store.py                 # 统一存储抽象（local / S3 / OSS）
    data_formatter.py               # 多格式工具调用归一化
    runtime_state.py                # Task Ledger + 运行时状态
    claw_adapter.py                 # *Claw 生态适配
    bedrock_client.py               # AWS Bedrock 集成
    config.py / config_store.py     # 配置管理
    setup_wizard.py                 # `skillclaw setup` 向导

evolve_server/                      # Evolver 侧
    pipeline/
        summarizer.py               # Stage 1：会话 → trajectory + summary
        session_judge.py            # Stage 2：四维质量评分过滤
        aggregation.py              # Stage 3：按技能分组
        execution.py                # Stage 4：LLM 决策（improve/create/skip）
        skill_verifier.py           # Stage 5：四项准入门控
    engines/
        workflow.py                 # 固定三阶段流水线编排器
        agent.py                    # OpenClaw 工作区 Agent 引擎
        agent_workspace.py          # 工作区目录管理
        agents_md.py                # AGENTS.md 动态生成
        openclaw_runner.py          # OpenClaw 进程启动/监控
        EVOLVE_AGENTS.md            # Agent 引擎指令文档（源头）
    core/
        skill_registry.py           # 技能注册表（SHA-256 ID + 版本历史）
        config.py                   # 演化参数（interval/batch/threshold）
        llm_client.py               # 统一 LLM 客户端（multi-provider）
        constants.py                # 全局常量
```

### 本地 + 云存储同步机制

`object_store.py` 提供三后端统一 CRUD 接口（`get_object` / `put_object` / `delete_object` / `iter_objects`）：

- **LocalObjectStore**：基于 `pathlib.Path`，适合单机或 NFS 挂载的多客户端场景
- **S3ObjectStore**：通过 `boto3` 对接 AWS S3（也兼容 MinIO 等 S3-compatible 服务）
- **OSSObjectStore**：通过 `oss2` 对接阿里云 OSS

存储层无内置锁机制，并发写冲突依赖「最后写入者胜」（last-write-wins）语义，演化引擎通过 SHA256 内容比对检测过时覆写后触发 LLM 合并。

### 多客户端共享 Evolve Server 部署

```
Client A (User 1) ──┐
Client B (User 2) ──┤──→ Shared Storage (S3/OSS) ←──→ Evolve Server
Client C (User 3) ──┘                                   (单实例或多实例)
```

所有 Client Proxy 通过相同的 `group_id` 指向同一存储 prefix，Evolve Server 可水平扩展（多实例共享同一 Storage），但 Registry 写入无显式分布式锁——多实例并发演化同一技能时依赖 SHA256 冲突检测 + 合并。

`--port` 模式允许 Evolve Server 暴露 HTTP 端点供 Client 主动触发单次演化（`--once`），而非只等待定时轮询——适合 CI/CD 流水线集成。

### arXiv 2604.08377 关键实验数据（从公开摘要和搜索结果确认）

- **基准**：WildClawBench，60 任务，6 个域，隔离 Docker 环境，真实工具
- **演化轮数**：6 轮（论文测试范围）
- **最大增益域**：Creative Synthesis，相对提升 **88.41%**
- **收敛速度最快域**：Social Interaction（第 2 轮即稳定）
- **基准水位**：前沿基础模型（Claude Opus 4.6、GPT-5.4）在同一 benchmark 上得分 < 0.55，SkillClaw 技能演化在此基础上叠加显著增益
- **无 RL**：系统明确为「监督式技能演化」（supervised skill evolution），无策略梯度或价值函数

---

## 与 Hermes / OpenClaw-RL 的关键差异（Phase 2 补充）

| 维度 | SkillClaw | Hermes（本地技能管理） | OpenClaw-RL（假设强化学习变体） |
|------|-----------|----------------------|-----------------------------|
| 技能所有权 | 共享池（group_id） | 单用户本地 | 不适用 |
| 演化机制 | LLM 批处理（监督式）| 无自动演化 | RL 策略优化（若存在） |
| 跨用户经验 | 核心特性 | 不支持 | 不支持 |
| 演化触发 | 定时轮询（有会话即触发）| 手动 | 在线 reward 信号 |
| 质量门控 | PRM + Verifier + replay 三层 | 无 | reward 函数 |
| 部署复杂度 | 需 Evolve Server | 零服务端 | 需训练基础设施 |
| 版本回滚 | `skillclaw restore`（最近 3 快照）| 无 | 无 |

---

## 借鉴建议（Phase 1 + Phase 2）

Phase 1 建议聚焦 prompt 组装与记忆接线；Phase 2 建议延伸至演化算法与评估体系。

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

### 6. 三层质量门控替代单一成熟度阈值（Phase 2 新增）

**来源**：SkillClaw 的 PRM Scorer + Skill Verifier + replay 验证三层叠加设计

**思路**：cryptotrader-ai 的 `learning/reflect.py` 目前只用 `trade_count + injection_count` 单维度判断规则成熟度。SkillClaw 的三层设计更健壮：①在规则产生时用 LLM 验证器做「四项准入检查」（证据接地性、保留现有价值、具体可复用性、立即可发布性）；②在规则被使用时用 PRM 逐步骤评分，区分「过程优质但结果差」与「结果对但过程冗余」；③在规则提升后用 replay 对比验证（新规则 >= 基线 AND >= 0.75 绝对分）才正式晋级。对应 spec 018 可将这三层映射为：`reflect_verifier`（产生时）→ `verbal_prm`（注入时）→ `replay_gate`（晋级时）。

**映射**：`learning/reflect.py`（`_merge_memories()` 之前加产生验证）→ `learning/verbal.py`（注入后 PRM 评分）→ 新增 `learning/replay.py`（离线 replay 对比）。

### 7. 会话摘要前置 + 演化批处理解耦（Phase 2 新增）

**来源**：SkillClaw Summarize → Judge → Aggregate → Execute 的流水线解耦

**思路**：cryptotrader-ai 的 `learning/reflect.py` 在每次交易周期结束时同步执行 LLM 反思，直接用原始 `AgentAnalysis` 作为输入。SkillClaw 的关键设计是将「摘要生成」与「演化决策」解耦：Summarizer 只做轨迹压缩（轻量，可并行），Session Judge 只做质量过滤（设有权重、可跳过），Executor 只做策略决策（昂贵，但输入已经是压缩后的摘要）。cryptotrader-ai 可借鉴此分层：`nodes/data.py` 的 `verbal_reinforcement()` 先做摘要化，`learning/reflect.py` 后台批量处理摘要，避免交易路径上的同步 LLM 阻塞。

**映射**：`nodes/data.py`（新增 session_summarizer 步骤）→ `learning/reflect.py`（从摘要而非原始分析中运行 _merge_memories()）。

### 8. Agent 引擎模式：给复杂规则演化用 LLM 工作区（Phase 2 新增）

**来源**：SkillClaw `evolve_server/engines/agent.py` 的 Agent 引擎设计

**思路**：对于高度复杂的经验规则演化（如多因子相关性规则、跨时间窗口模式），固定的 LLM 流水线可能不够灵活。SkillClaw 的 Agent 引擎将一个 LLM Agent 放入「工作区」，让它自主读取历史版本、对比数据、决定改什么——这与 spec 018 的「自治经验演化」方向高度吻合。可在 `learning/reflect.py` 的月度深度反思模式（`full_evolve=True`）中实现类似的 Agent 引擎分支：用 LangGraph 子图代替固定的 `_merge_memories()` 调用。

**映射**：`learning/reflect.py`（新增 `agent_reflect()` 分支）→ `nodes/debate.py`（复用 debate 子图框架）。

---

## 注意事项（Phase 2 更新）

**Phase 2 已核实的修正点：**

1. **License 已核实为 MIT**（非 Apache-2.0），已更新 frontmatter。AMAP-ML 的其他仓库用 Apache-2.0，SkillClaw 本身为 MIT。

2. **`push_min_injections` 语义已修正**：Phase 1 描述「注入 5 次才纳入演化池」不准确。源码逻辑是「0 注入绕过门控；≥5 次注入后才检查 effectiveness 是否达标」——「实习期」语义，不是入场门槛。

3. **无 RL 组件**：Agent 引擎与 Workflow 引擎均为监督式 LLM 演化，无策略梯度或价值函数，与「OpenClaw-RL」假设不同。

4. **Task Ledger 细节**：源码在 `runtime_state.py` 中，但 Phase 2 未直接读取该文件完整内容，goal/decision/outcome 三段结构为文档层面确认，JSON schema 精确字段仍有待进一步核实（如需用于实现，应读取源码）。

5. **语义检索已确认**：Template 模式（effectiveness 排序，默认）和 Embedding 模式（Qwen3-Embedding-0.6B，可选）均已在 `skill_manager.py` 中核实。

**已确认可直接用于 spec 017/018 的设计决策：**
- SKILL.md 两阶段加载（Phase 1 确认，Phase 2 强化）
- Summarize → Judge → Execute 流水线解耦（Phase 2 新增）
- PRM 逐步骤评分 + 三层质量门控（Phase 2 新增）
- `history/` 版本审计轨迹 + evidence 文件（Phase 2 新增）
- 共享池演化 + manifest 同步协议（Phase 2 新增）
