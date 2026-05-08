---
name: MetaClaw
url: https://github.com/aiming-lab/MetaClaw
license: MIT
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: true
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

## Phase 2：进化算法（核心机制）

### 双循环进化架构

MetaClaw 将学习分为两个时间尺度不同的循环（`scheduler.py` 注释中有明确描述）：

| 循环 | 频率 | 内容 |
|------|------|------|
| **快内循环**（Fast inner loop） | 每 N 轮对话后，或 `session_done` 时 | 技能进化：从失败对话中提炼新技能 |
| **慢外循环**（Slow outer loop） | 仅在用户空闲/睡眠/开会期间 | RL 梯度更新：Tinker LoRA 权重更新 |

这一设计使得技能库随使用实时进化，而昂贵的模型权重更新被推迟到不影响交互体验的窗口。

### generation 计数器的精确驱动逻辑

`skill_manager.py` 中的 `generation` 是一个整数计数器，只有在 `add_skills()` 实际成功添加了至少一个技能时才自增：

```python
# skill_manager.py — add_skills() 核心逻辑（伪码）
def add_skills(self, skills: list[dict], category: str) -> int:
    added_total = 0
    for skill in skills:
        if self.add_skill(skill):   # 单技能插入（内部去重检查）
            added_total += 1
    if added_total > 0:
        self.generation += 1       # 仅有实际新增才计数
    return added_total
```

当 `generation` 增大后，`trainer.py` 的 `_maybe_evolve_skills()` 方法立即执行缓冲区刷新：

```python
# trainer.py — _maybe_evolve_skills()（第 357-369 行）
new_generation = self.skill_manager.generation
if new_generation > old_generation:
    self._current_skill_generation = new_generation
    discarded_pending = len(self._pending_batch)
    self._pending_batch.clear()
    discarded_queue = self.rollout_worker.clear_output_queue()
    logger.info(
        "[Trainer] skill_generation %d→%d: discarded %d pending + %d queued samples",
        old_generation, new_generation, discarded_pending, discarded_queue,
    )
```

每个 `ConversationSample` 在生成时被打上当前 `generation` 时间戳；训练循环在收集样本时过滤掉旧版本样本（`trainer.py` 第 497-500 行）：

```python
fresh = [
    s for s in group
    if s.skill_generation >= self._current_skill_generation
]
```

### 演化触发条件

技能进化有三个独立触发路径：

1. **质量门槛触发**（RL 模式）：每个训练批次完成后调用 `_maybe_evolve_skills()`，检查批次成功率。触发阈值由 `config.skill_update_threshold` 控制（默认 0.4）：
   ```python
   # skill_evolver.py — should_evolve()（第 112-126 行）
   def should_evolve(self, batch, threshold=0.4) -> bool:
       successes = sum(1 for s in batch if getattr(s, "reward", 0) > 0)
       rate = successes / len(batch)
       return rate < threshold
   ```

2. **定时批量触发**（Skills-only / RL 模式）：每积累 `skill_evolution_every_n_turns` 轮（默认 10）对话后触发，不依赖 reward 质量：
   ```python
   # api_server.py 第 1430-1439 行
   if len(buf) >= evolution_every_n:
       evolution_turns = list(buf)
       self._session_turns[session_id] = []
       self._safe_create_task(self._evolve_skills_for_session(evolution_turns))
   ```

3. **会话结束触发**（session_done 信号）：收到 `X-Session-Done: true` 头时，若还有未处理的对话缓冲区，立即触发最后一次进化。

### 进化 LLM Prompt 设计

`skill_evolver.py` 的 `_build_analysis_prompt()` 构造三段式 prompt（第 237-308 行）：

- **段 1 — 失败案例**：最多 6 个失败对话，每个摘取末尾 600 字节上下文 + 前 500 字节响应
- **段 2 — 现有技能名称**：防止 LLM 生成重复技能
- **段 3 — 生成指令**：要求输出 JSON 数组，每个技能必须包含 `name`（小写 slug）、`description`（一句话，说明何时触发及效果）、`content`（6-15 行 Markdown，含 Anti-pattern 小节）、`category`

响应解析后，通过 `_finalise_names()` 确保 slug 合法；无法命名的技能自动分配 `dyn-NNN` 格式编号（N 从现有最大 `dyn-*` 下标 + 1 开始）。

### 跨 session 去重与冲突解决

技能层面的去重在 `SkillManager.add_skill()` 内部通过名称唯一性检查实现（同名技能不写入）。记忆层面的跨 session 整合由 `MemoryConsolidator.consolidate()` 完成，其五步流水线为（`consolidator.py` 第 32-73 行）：

1. **过期工作摘要清理**：仅保留最新的 `WORKING_SUMMARY`，其余标记为 `SUPERSEDED`
2. **精确内容去重**：以 `(memory_type, content.strip())` 为键，同键后续记录标记为已超越
3. **近似内容合并**：token 集合 Jaccard 相似度 ≥ 0.80 的同类记忆合并（保留 importance 较高者）
4. **实体跨强化**：共享实体的记忆获得 +0.05 的 `reinforcement_score` 提升（上限 +0.3）
5. **时效衰减**：30 天后开始按线性/指数公式衰减 importance，下限 0.15

---

## Phase 2：技能数据结构精确 Schema

### 技能文件格式（Markdown + YAML frontmatter）

技能以 `.md` 文件存储于 `~/.metaclaw/skills/`，YAML frontmatter 包含三个必需字段：

```yaml
---
name: handle-partial-input          # 小写 hyphenated slug，正则 ^[a-z][a-z0-9-]{1,}$
description: "Use when the user provides incomplete data. ..."   # 单句，检索触发描述
category: coding                    # 七类之一：coding/research/data_analysis/security/
                                    #          communication/automation/agentic
                                    # 或 general / common_mistakes
---
# 技能 content 正文（6-15 行 Markdown）
## Handle Partial Input

1. Detect that input is incomplete...
...
**Anti-pattern:** Assuming input is complete without checking.
```

内存中技能以嵌套 dict 组织：

```python
{
  "general_skills":      [{"name": ..., "description": ..., "category": ..., "content": ...}],
  "task_specific_skills": {"coding": [...], "research": [...], ...},
  "common_mistakes":     [{"name": ..., ...}]
}
```

### MemoryUnit 完整字段表

来源：`memory/models.py`（24 个字段）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `memory_id` | str | — | UUID 主键 |
| `scope_id` | str | — | 多租户隔离键 |
| `memory_type` | MemoryType | — | 六类枚举 |
| `content` | str | — | 原始内容 |
| `summary` | str | `""` | LLM 生成摘要 |
| `source_session_id` | str | `""` | 来源会话 |
| `source_turn_start` | int | `0` | 来源轮次起点 |
| `source_turn_end` | int | `0` | 来源轮次终点 |
| `entities` | list[str] | `[]` | 实体列表 |
| `topics` | list[str] | `[]` | 主题标签 |
| `importance` | float | `0.5` | 重要度（0-1） |
| `confidence` | float | `0.7` | 置信度（0-1） |
| `access_count` | int | `0` | 访问次数（≥3 自动提升 importance） |
| `reinforcement_score` | float | `0.0` | 跨实体强化得分 |
| `status` | MemoryStatus | `ACTIVE` | ACTIVE / SUPERSEDED / ARCHIVED |
| `supersedes` | list[str] | `[]` | 被本条取代的 memory_id 列表 |
| `superseded_by` | str | `""` | 取代本条的 memory_id |
| `embedding` | list[float] | `[]` | 向量表示（存为 JSON 数组） |
| `created_at` | str | UTC ISO | 创建时间 |
| `updated_at` | str | UTC ISO | 更新时间 |
| `last_accessed_at` | str | `""` | 最后访问时间 |
| `expires_at` | str | `""` | TTL 到期时间（可选） |
| `tags` | list[str] | `[]` | 自定义标签（用于 context_tags 加权） |

检索结果包装为 `MemorySearchHit`，额外携带 `score`（float）、`matched_terms`（list[str]）、`reason`（str 说明原因）。

### render_for_prompt() 精确渲染规则

`memory/manager.py` 的渲染逻辑：

1. 无结果时返回空字符串
2. 写入标题 `## Relevant Long-Term Memory`
3. **Pinned 优先**：`importance ≥ 0.99` 的记忆提前展示
4. **按类型分组**：每组一个 `### [MemoryType]` 子标题
5. **内容优先摘要**：优先使用 `content`，避免 summary 与内容重复
6. **新鲜度标签**：对每条记忆追加 `_freshness_tag()`（如 `[recent]`、`[1d ago]`）

```
## Relevant Long-Term Memory
### working_summary
- [内容文本] [recent]
### project_state
- [内容文本] [2d ago]
```

---

## Phase 2：检索机制深析

### 四种检索路径

`memory/retriever.py` 中 `MemoryRetriever.retrieve()` 支持四种路径：

| 模式 | 触发条件 | 算法 |
|------|----------|------|
| `keyword` | 默认 | FTS5 全文检索 + 查询扩展 |
| `embedding` | 显式配置 | 纯向量余弦相似度 |
| `hybrid` | 显式配置 或 `auto` + embedder 可用 + 查询 ≥ 4 词 | IDF 关键词 + embedding 融合 |
| `auto` | 配置为 auto | 短查询→keyword，中长查询+embedder→hybrid |

### Hybrid 模式精确评分公式

（`retriever.py` 第 93-174 行，从源码直接提取）：

```
score = (
    keyword_weight   × Σ log₂(N / df(term))  [命中 content]
  + metadata_weight  × Σ log₂(N / df(term))  [命中 topics/entities]
  + cosine_similarity(query_embedding, unit.embedding)
  + importance_weight × unit.importance
  + recency_weight   × max(0, 1 - age_hours/recent_bonus_hours)
  + unit.reinforcement_score
) × type_boost[memory_type] × (0.8 + 0.2 × unit.confidence)
```

默认权重（`memory/policy.py`）：
- `keyword_weight = 1.0`，`metadata_weight = 0.45`
- `importance_weight = 0.5`，`recency_weight = 0.3`
- `recent_bonus_hours = 72`（3 天内视为"新鲜"）
- type_boost：`working_summary=1.2`，`project_state=1.1`，`episodic=0.8`

未命中任何查询词的记忆单元直接跳过（零召回无代价）。

context_tags 的标签加权在 `_apply_tag_boost()` 中：每匹配一个标签 +15%，上限 +50%。

### Embedding 模型的使用方式

`memory/embeddings.py` 提供两种 embedder 实现：

- **HashingEmbedder**（默认，无依赖）：SHA-256 哈希到 64 维空间，L2 归一化。用于 keyword + hybrid 降级场景。
- **SentenceTransformerEmbedder**：通过 `SentenceTransformer(model_name)` 加载，调用 `encode(text, normalize_embeddings=True)` 得到归一化向量。默认模型为 `all-MiniLM-L6-v2`（384 维），可通过配置替换为 `Qwen/Qwen3-Embedding-0.6B`。

Embedding 何时计算：记忆单元摄入（ingest）时，如果 embedder 可用则计算 embedding 并写入 `embedding_json` 列；若 embedder 不可用则 embedding 为空列表，hybrid 模式中 `embedding_score` 项贡献 0。

Embedding 如何存储：`store.py` 的 `embedding_json` 列存储 JSON 数组，读取时反序列化为 `list[float]`。向量不单独建立 ANN 索引，检索时遍历所有 active 记忆（上限 500 条）逐一计算余弦相似度（纯 Python `sum(a*b for a,b in zip(left, right))`，要求向量已预归一化）。

### 关键词路由的命中规则与查询扩展

FTS5 检索返回结果不足（< max(2, limit//2)）时，自动触发查询扩展：将原始 token 通过内置同义词表扩展（如 `"db"→"database"`，`"k8s"→"kubernetes"`，共 20 余组），重新检索并将扩展命中结果以 0.85 折扣合并，避免纯扩展匹配污染高质量直接匹配。

---

## Phase 2：评估体系

### MetaClaw-Bench 结构

Benchmark 位于 `benchmark/`，包含：

- **metaclaw-bench**：30 天交互历史场景（完整版）
- **metaclaw-bench-small**：12 天场景（快速验证用）
- 每个场景以 `day{N}/` 目录组织，包含 `workspaces/`（真实项目文件）和 `eval/`（评估问题）

评估任务分为两类（`scoring/scoring_cmd.py`）：

| 类型 | 评分逻辑 |
|------|----------|
| `multi_choice` | 从模型响应提取 `\bbox{...}` 或 `\boxed{...}` 答案集合，与 ground-truth 计算 exact_match / IoU / F1 / score（score = 1 − (FP+FN)/q_num） |
| `file_check` | 读取推断结果中的 `inline_score.passed` 布尔值，直接判 1.0 / 0.0 |

### A/B 对比实验脚本

`benchmark/scripts/` 提供六种标准化对比配置：

| 脚本 | 配置说明 |
|------|----------|
| `baseline_run.py` | 直连 LLM API，无任何增强 |
| `proxy_passthrough_run.py` | 过代理但不注入，测量代理本身开销 |
| `skills_only_run.py` | 仅技能注入，无记忆 |
| `memory_run.py` | 仅记忆注入，无技能 |
| `rl_only_run.py` | 仅 RL，无技能/记忆 |
| `madmax_memory_run.py` | RL + 技能 + 记忆（全功能） |

还有 `skills_memory_run.py` 和 `rl_only_memory_run.py` 作为中间配置。

### PRM（过程奖励模型）评分

`prm_scorer.py` 实现了一个 LLM-as-judge 方案：

- **M 次平行投票**（`prm_m`，默认 3）：对同一 response 并发发出 M 次评判请求
- **判决 prompt**：要求 judge LLM 以 `Score: 1`（有帮助）/ `Score: -1`（无帮助）/ `Score: 0`（不确定）结尾
- **多数投票**：`_majority_vote()` 取票数最多的分值；若 1/-1 平票则返回 0.0
- PRM 判断针对的是"当前 response 是否回应了当前 instruction"，而非"是否导致了良好结局"，避免对话历史状态依赖

进化触发时 reward ≤ 0 的样本被收集为 `failed_samples` 传入 `SkillEvolver.evolve()`。

### OPD（On-Policy Distillation）辅助评估

`config.use_opd=True` 时，每次推理后同步向教师模型查询其对学生响应的 token 级 log-probabilities（`api_server.py` 第 1115-1153 行），将 KL 散度惩罚项 `kl_penalty_coef` 加入训练损失，实现对齐约束。

---

## Phase 2：Agent ↔ Skill 边界

### 透明代理协议

MetaClaw 作为 OpenAI 兼容反向代理运行于 `localhost:{proxy_port}`（默认 30000）。下游 Agent 无需修改——所有信息通过 HTTP header 传递：

| Header | 含义 |
|--------|------|
| `X-Session-Id` | 会话标识，用于追踪 turn 计数和记忆 scope |
| `X-Turn-Type` | `"main"`（训练轮次）或 `"side"`（工具子任务，不产生训练数据） |
| `X-Session-Done` | `"true"` 触发会话结束处理（记忆摄入 + 技能进化） |
| `X-Memory-Scope` | 显式记忆空间 ID（可选，默认从 session_id 派生） |
| `X-User-Id` / `X-Workspace-Id` | 多租户支持 |

若 Agent 未发送 `X-Session-Id`（TUI 模式），自动生成 `tui-{model}` 作为 session_id，并将 `turn_type` 默认为 `"main"`。

### 三种工作模式的精确切换条件

切换逻辑在 `api_server.py` 的 `_handle_request()` 方法中（第 1254-1266 行）：

```python
if turn_type == "main":
    if self.memory_manager and self.skill_manager and self.config.synergy_enabled:
        messages = await self._inject_augmentation(messages, scope_id=...)   # Synergy
    elif self.memory_manager:
        messages = await self._inject_memory(messages, scope_id=...)          # Memory Only
    elif self.skill_manager:
        messages = self._inject_skills(messages)                               # Skills Only
```

| 模式 | 前提条件 |
|------|----------|
| **Synergy** | `memory_manager` 已初始化 + `skill_manager` 已初始化 + `config.synergy_enabled=True` |
| **Memory Only** | 仅 `memory_manager` 已初始化 |
| **Skills Only** | 仅 `skill_manager` 已初始化（config.mode="skills_only"，不使用 Tinker，直连 LLM） |

在 Skills Only 模式下，`api_server.py` 的 `_forward_to_llm()` 而非 `_forward_to_tinker()` 被调用，即请求直接转发到用户配置的 LLM provider，不经过 Tinker LoRA sampling client，无 RL 梯度更新。

### claw_adapter.py 的适配协议

`claw_adapter.py` 为五种 Agent 提供一键接入方法：

| Agent | 配置文件 | 重载方式 |
|-------|----------|----------|
| openclaw | CLI 命令行 | `openclaw config set models.providers.metaclaw` + `gateway restart` |
| copaw | `~/.copaw/config.json` | JSON patch + `copaw daemon restart`（支持 ConfigWatcher 热重载） |
| ironclaw | `~/.ironclaw/.env` | 行级 `.env` patch + `ironclaw service restart` |
| picoclaw | `~/.picoclaw/config.json` | model_list 插入 + `picoclaw gateway restart` |
| zeroclaw | `~/.zeroclaw/config.toml` | TOML 行级 patch + `zeroclaw service restart` |

所有适配器的本质是将 Agent 的 LLM endpoint 重定向为 `http://127.0.0.1:{proxy_port}/v1`，`api_key` 设为 `config.proxy_api_key`（默认 `"metaclaw"`）。

---

## Phase 2：工程实现细节

### 20k token 贪心截断的精确丢弃顺序

截断逻辑位于 `api_server.py` 的 `_truncate_messages()` 方法（基于 `_handle_request()` 第 1273-1276 行的调用方式）：

- **可用预算** = `config.max_context_tokens`（默认 20000）− `body["max_tokens"]`（默认 2048）
- **截断规则**：
  1. System 消息**始终保留**，不参与截断
  2. 从最旧的非 system 消息开始丢弃
  3. 保证至少保留**最后一条 user 消息**
  4. token 计数通过 `tokenizer.apply_chat_template()` + `tokenizer.encode()` 精确计算（无分词器时退化为空格分词估算）

### 存储介质与数据库 Schema

**技能库**：Markdown 文件于磁盘 `~/.metaclaw/skills/`，启动时全量加载到内存 dict；`add_skill()` 同时写磁盘和内存。

**记忆库**：SQLite 数据库（默认路径来自配置，sidecar 模式下独立文件）。`store.py` 维护 7 张表：

| 表名 | 用途 |
|------|------|
| `memories` | 主记忆存储（含 `embedding_json`、`entities_json`、`topics_json`） |
| `memories_fts` | FTS5 虚表（`unicode61` 分词，索引 content + summary + entities_text + topics_text） |
| `memory_events` | 变更审计日志 |
| `memory_links` | 记忆间关系图（source_id → target_id + link_type） |
| `memory_watches` | 订阅/通知机制 |
| `memory_annotations` | 用户备注 |
| `scope_access` | 多租户权限控制 |
| `schema_version` | DB 迁移版本（当前 v6） |

**RL 训练样本**：`conversations.jsonl` + `prm_scores.jsonl` 两个 JSONL 文件，训练开始时清空（`purge_record_files()`）。

### SlowUpdateScheduler：三触发窗口检测

调度器（`scheduler.py`）通过状态机 `IDLE_WAIT → WINDOW_OPEN → UPDATING → PAUSING → IDLE_WAIT` 管理训练窗口：

| 窗口类型 | 检测方式 | 说明 |
|----------|----------|------|
| **睡眠时段** | 时钟与配置的 `sleep_start`/`sleep_end` 比对（支持跨午夜） | 如 23:00–07:00 |
| **键盘空闲** | `IdleDetector.idle_seconds()` ≥ `scheduler_idle_threshold_minutes * 60` | 默认 5 分钟 |
| **日历忙碌** | `GoogleCalendarClient.is_busy_now()` | 可选，用户在会议中时算"不活跃" |

每 60 秒检查一次。用户重新活跃时设置 `pause_event`，训练器在下次批次收集循环中检测到后保存当前部分批次（已过滤旧 generation 的样本）并等待下一个窗口。

### 部署架构

MetaClaw 在本机以多线程混合模式运行：

```
主进程（asyncio event loop）
├── MetaClawTrainer.run()          ← 训练主循环
├── SlowUpdateScheduler.run()      ← 窗口检测协程
└── AsyncRolloutWorker             ← 管理代理服务器线程
    └── 独立线程（uvicorn）
        └── MetaClawAPIServer      ← FastAPI 代理
            ├── /v1/chat/completions  ← 拦截 Agent 请求
            └── /v1/memory/*          ← 记忆 REST API
```

Tinker LoRA 训练客户端（`tinker.LoraTrainingClient`）是云端服务，通过网络调用；RL 的 `forward_backward_async` + `optim_step_async` + `save_weights_and_get_sampling_client_async` 三步串行（权重保存有 600s 超时保护），每 5 个 step 持久化一次检查点（`step_{N:04d}`）。

sidecar 模式（`openclaw-metaclaw-memory/`）将 MemoryManager 独立为单独进程，通过 HTTP 通信，适合内存与技能进化分开部署的场景。

---

## 借鉴建议

### Phase 1 条目（原有）

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

### Phase 2 条目（新增）

7. **双循环分离设计**：MetaClaw 将"快内循环（技能进化，每 N 轮）"与"慢外循环
   （权重更新，仅空闲时）"严格分离的设计，对 cryptotrader-ai 有直接参考价值——
   经验摘要（`reflect.py`）可设计为每次决策后异步触发（快），而经验规则的矛盾
   解决与重要度整合（`consolidate()`）可推迟到夜间低频批量执行（慢）。

8. **should_evolve 质量门槛**：`success_rate < threshold` 才触发进化的思路可直
   接移植到 `reflect.py`：若近期决策的胜率高于阈值（如 0.7），跳过规则提炼以
   避免过拟合；仅在连续失败时才触发经验更新。

9. **IDF 加权混合检索**：`retriever.py` 的混合评分公式（IDF 关键词 + 向量相似
   度 + importance + recency + reinforcement_score）的复合得分设计，比 cryptotrader-ai
   目前的"相关度 + 访问频率"两维度检索更精细，尤其是 recency_bonus 的连续衰减
   函数可替换当前的离散新鲜度标记。

10. **status 三态生命周期**：`ACTIVE / SUPERSEDED / ARCHIVED` 三态代替简单删除，
    保留被取代记忆的审计链（`supersedes` / `superseded_by` 字段）。这对
    cryptotrader-ai 的经验规则版本追踪有参考价值，可知道哪条旧规则被哪条新规则
    替换，便于回溯调试。

11. **token_budget 感知的 MemoryPolicy**：MemoryPolicy 将 `max_injected_units`（默认 6）
    和 `max_injected_tokens`（默认 800）解耦，并提供 `balanced/recall/precision/recent`
    四个命名 profile，是比硬编码上下文预算更灵活的设计，可应用于 cryptotrader-ai
    的 `verbal_reinforcement()` 中按市场波动率动态调整注入精度。

12. **多 Agent 适配注册表模式**：`claw_adapter.py` 的 `_ADAPTERS` dict 注册表模式
    使得新增 Agent 支持只需添加一个 `_configure_XXX()` 函数，无需修改 dispatcher
    核心逻辑——这种模式可用于 cryptotrader-ai 的多 LLM 提供商切换逻辑。

---

## 注意事项 / 开放问题

- **记忆提取触发时机**：实际实现中记忆提取在 `session_done` 时批量触发（非固定轮次），
  但 RL 模式下的记忆摄入与技能进化是分开的两个异步任务。对于 cryptotrader-ai 的
  交易 session，应考虑按"决策边界"（下单/拒单）而非 session_done 触发。

- **Synergy 模式的技能/记忆 token 竞争**（已解答）：两者均在同一 system 消息中
  注入，共享 `max_context_tokens - max_tokens` 的可用预算，之后统一走贪心截断。
  `_inject_augmentation()` 先注入记忆，再追加技能内容，无显式优先级分配——实际上
  两者各自独立按自己的 token 预算（memory: `max_injected_tokens=800`）截断，
  截断后合并写入 system 消息。

- **Embedding 模型依赖**：代码中默认使用 `all-MiniLM-L6-v2`（384 维），`Qwen3-Embedding-0.6B`
  是可配置替换项而非强制依赖。CPU-only 环境可直接使用内置的 `HashingEmbedder`
  （64 维 SHA-256 哈希），无任何 Python 包外部依赖，相比语义 embedding 质量下降
  但可接受。

- **记忆固化的冲突检测**（已解答）：`consolidator.py` 不使用 topic 重叠检测冲突，
  而使用 Jaccard token 相似度（阈值 0.80）检测近似重复，无"内容冲突"概念——
  两条观点相反但 token 重叠低的记忆会同时保留。与 cryptotrader-ai 的 `_merge_memories`
  加权平均策略不等价；MetaClaw 的策略更保守，倾向于保留而非合并。

- **多 Agent 适配接口**（已解答）：`claw_adapter.py` 详见 Phase 2 "Agent ↔ Skill 边界"
  章节，核心是 endpoint 重定向 + config patch + service restart 三步，与 LangGraph
  集成时可仿照 `_configure_none()` 跳过自动配置，手动在 LangGraph 节点中调用
  proxy endpoint。

- **RL 训练的 Tinker 闭源依赖**：`trainer.py` 的外循环强依赖 `tinker` 云端 SDK
  （`LoraTrainingClient`），这部分在 cryptotrader-ai 中不可直接复用。技能进化
  （`skill_evolver.py`）和记忆固化（`consolidator.py`）均无此依赖，可独立移植。
