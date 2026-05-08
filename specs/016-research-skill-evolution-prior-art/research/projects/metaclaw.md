---
name: MetaClaw
url: https://github.com/aiming-lab/MetaClaw
license: MIT
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# MetaClaw — aiming-lab

## 架构概述

MetaClaw 是一个透明代理框架，拦截个人 AI Agent（OpenClaw、CoPaw、IronClaw 等）与
LLM API 之间的通信，在每次对话轮次中注入技能（Skill）与记忆（Memory），并在会话结束
后自动提炼新技能。系统提供三种工作模式：

| 模式 | 说明 |
|------|------|
| Skills Only | 仅技能注入，无需 GPU |
| RL | 技能注入 + GRPO 连续微调（Tinker/MinT/Weaver 后端） |
| Auto（默认） | RL + 调度器，在空闲/休眠窗口延迟权重更新 |

核心模块：
- `metaclaw/api_server.py` — 代理拦截与 Prompt 组装
- `metaclaw/skill_manager.py` — 技能检索、格式化、注入
- `metaclaw/memory/manager.py` — 记忆检索与 Prompt 渲染
- `metaclaw/memory/store.py` — SQLite + FTS5 持久化
- `metaclaw/memory/consolidator.py` — 去重、冲突消解、重要度衰减

---

## Prompt 组装（Phase 1）

### 组装流程（`api_server.py`）

请求到达代理后，系统按以下顺序组装最终 Prompt：

1. **系统提示压缩缓存**：首次收到原始 system message 时，用 LLM 对其压缩并
   写入 `system_prompt_cache.json`；后续请求直接复用，避免重复处理。

2. **轮次类型判断**：仅 `"main"` 类型轮次触发增强注入；`"side"` 轮次跳过，
   只记录日志，不产生训练样本。

3. **注入模式三分支**：
   - **Synergy（技能 + 记忆）**：调用 `_inject_augmentation()`，将检索到的
     记忆单元与技能指导混合注入 system 消息。
   - **Memory Only**：调用 `_inject_memory()`，将上下文记忆追加到 system 消息。
   - **Skills Only**：检索 top-k 匹配技能，格式化后前置到 system 消息；若原
     始请求无 system 消息则新建 system role。

4. **消息规范化**：
   - role 映射：`"developer"` → `"system"`，`"toolResult"` → `"tool"`
   - 多段 content 列表展平为纯文本
   - 解析 Kimi 标记（`<|tool_call_begin|>...<|tool_call_end|>`）和 Qwen
     包装（`<tool_call>...</tool_call>`）
   - 清理推理块（`<think>...</think>`），内容单独保留

5. **Token 预算管理**：
   - 保留 `max_tokens`（默认 2048）给响应
   - 可用预算 = `max_context_tokens`（默认 20,000）− 保留量
   - 贪心截断：迭代丢弃最旧的非 system 消息直至符合预算
   - 保证至少保留一条 user 消息
   - System 消息始终保留，不参与截断

### 技能检索（`skill_manager.py`）

技能以 Markdown 文件存储于 `~/.metaclaw/skills/`，YAML frontmatter 包含
`name`、`description`、`category` 字段。

**检索模式**（由 `retrieval_mode` 配置）：

- **Template 模式**（零延迟关键词匹配）：
  - 对输入文本分词（过滤停用词，≥3 字符 token），与 `_CONV_TASK_TYPES`
    字典（coding、research、data_analysis 等）做关键词重叠检测
  - 返回全部通用技能 + 匹配任务类别技能 + 常见错误技能
  - 可配置 `task_specific_top_k` 限制数量

- **Embedding 模式**（语义相似度）：
  - 使用 SentenceTransformer（默认 `Qwen/Qwen3-Embedding-0.6B`）计算余弦相似度
  - 初始化后缓存技能嵌入
  - 通用技能与任务专属技能分别按 `top_k` 检索

- **Relevance 过滤**（`retrieve_relevant()`）：
  - 重叠系数关键词匹配，`min_relevance=0.07` 过滤
  - 按相关度降序排序

**格式化输出**（`format_for_conversation()`）：
```
## Active Skills

### {name}
_{description}_

{content}
```

**技能进化信号**：`add_skills()` 成功后 `generation` 计数器自增，通知 RL
训练器丢弃进化前的历史样本（对应 MAML support/query 集分离逻辑）。

---

## 记忆 ↔ 技能（Phase 1 精简版）

### 记忆类型（`memory/manager.py` + `memory/models.py`）

系统维护六类记忆，每个 `MemoryUnit` 包含：content、summary、topics、
entities、importance（0-1）、confidence、tags、时间戳、access_count。

| 类型 | 用途 |
|------|------|
| Episodic | 会话级事件与交互历史 |
| Semantic | 事实性知识与通用信息 |
| Preference | 用户偏好与约定 |
| Project State | 当前项目配置与状态 |
| Procedural Observation | 工作流与操作模式 |
| Working Summary | 近期会话综合摘要 |

### 检索机制

`retrieve_for_prompt()` 在 token 预算约束下检索匹配任务描述的记忆：
- 支持 keyword、embedding、hybrid 三种检索模式
- LRU 缓存（最多 16 条）加速重复查询
- 结果按相关度分数 + 访问频率综合排序
- `access_count ≥ 3` 的记忆自动提升 importance

### Prompt 渲染格式（`render_for_prompt()`）

```
## Relevant Long-Term Memory
### [Memory Type]
- [content] [freshness_tag]
```

- 按类型分组，层级化展示
- Pinned 记忆（importance ≥ 0.99）优先展示，确保可见性

### 记忆固化（`memory/consolidator.py`）

- 检测内容冲突（高 topic 重叠 + 内容不同）
- 内容去重防止冗余存储
- 频繁访问记忆的重要度增强；陈旧记忆的重要度衰减
- 摄入后若 `auto_consolidate=True` 自动触发

### 记忆与技能的协同关系

在 Synergy 模式下，`_inject_augmentation()` 将记忆上下文与技能指导融合注入
同一 system 消息，形成统一的增强上下文。两者共享同一 token 预算，
`max_context_tokens=20000` 为总量上限，系统优先保留 system 消息。

记忆提取在会话结束后进行（可配置间隔，默认每 5 轮），新技能摘要也在会话后
异步写入，两者均不阻塞在线推理路径。

---

## Phase 2 占位符

（空列表 — 进化算法 / 技能数据结构深析 / 高级检索策略 / 评估 Benchmark /
Agent↔Skill 边界设计 / 工程实现细节）

---

## 借鉴建议（仅 Phase 1）

1. **双模式检索开关**：Template 模式（零延迟关键词）与 Embedding 模式（语义）
   可按场景切换，适合 cryptotrader-ai 在低延迟行情触发与高质量经验检索间的
   权衡——行情节点用 Template，复盘经验用 Embedding。

2. **系统提示分层组装**：MetaClaw 将技能区块（`## Active Skills`）与记忆区块
   （`## Relevant Long-Term Memory`）分开注入同一 system 消息的思路，可直接
   应用于 `verbal_reinforcement()` 节点的 prompt 构造，保持结构清晰可读。

3. **Token 预算贪心截断**：按"保留 system + 丢弃最旧 user"的贪心截断策略简单
   可靠，优于复杂的动态摘要；cryptotrader-ai 的行情快照拼接可采用同样逻辑。

4. **MemoryUnit importance 字段**：0-1 浮点重要度 + access_count 自动提升机制，
   可参考用于 `ExperienceRule` 的 maturity/成熟度管理，替代当前的显式分级。

5. **系统提示压缩缓存**：首次压缩 + 后续复用的缓存策略，可用于 cryptotrader-ai
   中长期不变的 agent persona 提示，减少每次调用的 token 开销。

6. **generation 计数器作进化信号**：技能库版本化（generation++）后通知下游
   丢弃旧样本，是防止经验记忆混入过期规则的轻量机制，值得在 `reflect.py` 的
   `_verify_rules` 流程中参考。

---

## 注意事项 / 开放问题

- **记忆提取触发时机**：每 5 轮提取一次记忆是固定策略；对于 cryptotrader-ai
  的交易 session，应考虑按"决策边界"（下单/拒单）而非固定轮次触发。

- **Synergy 模式的技能/记忆 token 竞争**：两者共享同一预算，文档未说明优先级
  分配比例，需读 `_inject_augmentation()` 源码确认（Phase 2 任务）。

- **Embedding 模型依赖**：默认使用 `Qwen/Qwen3-Embedding-0.6B`，若 cryptotrader-ai
  部署在 CPU-only 环境，需评估此依赖的可行性或退化为 Template 模式。

- **记忆固化的冲突检测细节**：`consolidator.py` 的 topic 重叠算法尚未精读，
  与 cryptotrader-ai `_merge_memories` 的加权平均策略是否等价待 Phase 2 确认。

- **多 Agent 集成接口**：MetaClaw 通过 OpenAI 兼容代理接入多 Agent，其
  `claw_adapter.py` 的适配逻辑可能对 cryptotrader-ai 的 LangGraph 集成有参考
  价值，但接口细节未读。
