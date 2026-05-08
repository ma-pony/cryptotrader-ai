---
name: OpenClaw-RL
url: https://github.com/Gen-Verse/OpenClaw-RL
license: Apache-2.0
tier: 1
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# OpenClaw-RL — Gen-Verse

## 架构概览

OpenClaw-RL 是一个完全异步的强化学习框架，核心论文发表于 arXiv 2603.10165（ICML 2026 相关工作）。
其核心洞察是：**每次 agent 交互产生的「下一状态」（用户回复、工具输出、环境状态变化）都是天然的训练信号**，无需人工标注。

系统由四个解耦的异步循环构成：

1. **API Server（策略服务）** — 作为 OpenAI 兼容的 chat proxy，转发请求给策略模型并收集 per-token log-probabilities
2. **Rollout Worker** — 将 API server 产生的 trajectory 数据异步推送给训练后端
3. **PRM Judge（奖励评估）** — 基于下一状态对每个 turn 独立打分（+1 / -1 / 0），多数投票聚合
4. **Trainer** — 后台持续更新策略（基于 SLIME/GRPO 训练后端），与推理零协调开销

关键特点：模型在推理时**同时**接受训练，不阻塞服务。支持三种学习范式：二元 RL（GRPO）、在线经验蒸馏（OPD / OEL）、混合方法。

---

## Prompt 组装（Phase 1）

### 1. 消息规范化管道

在将对话历史送入 tokenizer 之前，系统通过三个预处理函数规范化消息格式：

```python
# openclaw-rl/openclaw_api_server.py（简化，约 15 行关键逻辑）
def _normalize_messages_for_template(messages):
    # 1. 将 "developer" role 转换为 "system"
    # 2. 将多模态内容列表（list[dict]）展平为纯文本字符串
    # 3. 将 tool_call arguments 从 JSON 字符串转回 dict（供 Jinja2 模板使用）
    ...

def _flatten_message_content(content):
    # 处理 str / list[{"type":"text","text":...}] 两种格式
    ...

def _normalize_tool_call(tool_call):
    # 确保 arguments 字段是 dict 而非 JSON 字符串
    ...
```

### 2. Chat Template 应用

系统使用 tokenizer 的内置 Jinja2 chat template 组装最终 prompt，而不是维护自己的模板字符串：

```python
# _handle_request() 核心逻辑（openclaw_api_server.py）
norm_msgs = _normalize_messages_for_template(messages)
norm_resp = _normalize_messages_for_template([response_msg])[0]

# 仅包含历史 → 生成 prompt 文本（含 add_generation_prompt=True）
prompt_text = tokenizer.apply_chat_template(
    norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True
)
# 包含本次 response → 提取 response token 范围
full_text = tokenizer.apply_chat_template(
    norm_msgs + [norm_resp], tools=tools, tokenize=False, add_generation_prompt=False
)
# response_text = full_text[len(prompt_text):]
```

**关键设计**：prompt 边界通过字符串 prefix 截取确定，不依赖特殊 token 标记。

### 3. 经验注入（OEL 模块）

在 `openclaw-rl/oel/openclaw_oel_api_server.py` 中，当会话具有已积累的经验（experience）时，系统在发送给策略模型之前动态注入：

```python
EXPERIENCE_SOLVE_PROMPT_TEMPLATE = """\
Given previous learned experience:
# Experience
{experience}

Apply the relevant experience to respond to the following:
{prompt}"""

def _inject_experience_to_messages(self, messages, experience):
    # 定位最后一条 user 消息，将内容替换为模板渲染后的增强版本
    enhanced_content = EXPERIENCE_SOLVE_PROMPT_TEMPLATE.format(
        experience=experience, prompt=content
    )
    # 返回修改后的 messages 列表（不修改原始对话历史）
```

**注入位置**：最后一条 user 消息的内容被替换，system prompt 和其他消息不变。

### 4. PRM 评估 Prompt

奖励模型的评估 prompt 由 `_build_prm_eval_prompt()` 动态构建：

```python
# 结构（伪代码）
system = "You are a process reward model (PRM)..."
user = (
    "## Assistant output\n" + response_text +
    "\n\n## Next state (" + next_state_role + ")\n" + next_state_text
)
# 输出：boxed score，格式 \boxed{1} / \boxed{-1} / \boxed{0}
```

---

## Memory ↔ Skill（Phase 1 lite）

### 会话级内存数据结构

`OpenClawAPIServer` 使用以下 in-memory 字典管理会话状态（`openclaw_oel_api_server.py`）：

```python
self._turn_counts: dict[str, int]              # session_id → 当前 turn 编号
self._pending_records: dict[str, dict]          # 等待下一状态的 turn 缓冲
self._pending_turn_data: dict[str, dict[int, dict]]  # session_id → turn_num → {prompt_ids, response_ids, logprobs}
self._session_effective: dict[str, int]         # 可训练样本计数
self._session_conversations: dict[str, list]    # session_id → [{"user": str, "assistant": str}]
self._session_experience: dict[str, str]        # per-session 经验字符串（OEL 模式）
self._oel_tasks: dict[str, dict[int, asyncio.Task]]  # 异步经验提取任务
```

所有状态存储于进程内存，**无持久化层**（重启丢失）。

### 经验提取机制（Phase 1 lite — 仅 wiring）

每个 turn 完成后，系统触发异步经验提取任务（`loop.create_task()`），基于对话历史生成结构化经验文本。提取使用两个可配置的 prompt 模板：

- **V1**：通用经验精炼，生成 `- EXPERIENCE ITEM:` 格式的 markdown 列表
- **V2**：强调可操作的 do/don't 规则，包含去重逻辑，聚焦用户反馈信号

生成的经验字符串存入 `_session_experience[session_id]`，供下一 turn 的 `_inject_experience_to_messages()` 调用。

### 三种经验模式（wiring 层面）

| 模式 | 触发时机 | 跨会话性 |
|------|----------|----------|
| **session-experience** | 每 turn 提取，立即注入下一 turn | 无跨会话 |
| **replay** | 会话结束后对全历史提取，再重放教师评估 | 无跨会话 |
| **consolidate（多经验池）** | 从预加载的经验文件池中随机采样 | 支持离线跨会话 |

### 与训练的接线关系

经验只影响**推理侧**（注入 prompt）和**教师评估侧**（OPD 的 teacher context 构建），不直接写入模型权重。权重更新通过 GRPO / OPD 损失由 SLIME 后端处理。

**与 RL 训练管道的耦合点**：`_pending_turn_data` 中的 trajectory 数据（含 logprobs）通过 rollout worker 传给训练器，经验提取和奖励计算在同一异步循环中并行进行，但彼此解耦。

---

## Phase 2 占位符

（空 — 演化算法 / 技能数据结构 / 检索 / 评估 / Agent↔技能边界 / 工程实现）

---

## 借鉴建议（Phase 1）

### 建议 1：Tokenizer-native Chat Template 作为唯一 prompt 组装出口

OpenClaw-RL 不维护自己的模板字符串库，而是将所有 prompt 渲染委托给 tokenizer 的 Jinja2 chat template（`tokenizer.apply_chat_template()`）。这样做的优点：模板与模型版本自动绑定，工具调用格式自动处理，不存在「模板和模型不匹配」的问题。

**对 cryptotrader-ai 的借鉴**：在 `agents/base.py` 的 `create_llm()` 工厂中，可考虑将系统提示的最终组装交给 LangChain 的 `ChatPromptTemplate`，而非在每个 agent 中手动拼接字符串，保证格式一致性。

### 建议 2：动态经验注入点 — 仅修改最后一条 user 消息

OpenClaw-RL 的 `_inject_experience_to_messages()` 只替换最后一条 user 消息的内容，system prompt 和历史 turns 保持不变。这比在 system prompt 中堆砌经验更干净：避免 system prompt 膨胀，且经验与指令的语义边界清晰。

**对 cryptotrader-ai 的借鉴**：在 `nodes/agents.py` 的 GSSC 经验注入逻辑中，可将经验注入点从 system prompt 末尾迁移到 user message 前缀，与 OpenClaw-RL 的 `EXPERIENCE_SOLVE_PROMPT_TEMPLATE` 模式对齐：`Given previous learned experience:\n{experience}\n\nApply to: {user_query}`。

### 建议 3：Turn 级 loss masking — 最后 turn 排除策略

OpenClaw-RL 明确将「没有收到下一状态的 turn」（即会话最后一 turn）排除出训练，但保留「至少一条」的兜底保证。这是一种简单但有效的训练信号质量控制机制。

**对 cryptotrader-ai 的借鉴**：在 PnL 反馈驱动的经验演化中（specs/016），类似地，只有已平仓的交易决策才应产生经验更新（因为只有它们有完整的奖励信号），开仓中的持仓不应触发 reflect。

### 建议 4：异步经验提取 + 推理不阻塞

OpenClaw-RL 使用 `loop.create_task()` 在后台异步执行经验提取（LLM 调用），不阻塞当前 turn 的响应。

**对 cryptotrader-ai 的借鉴**：`nodes/data.py` 的 `verbal_reinforcement()` 中已有类似模式（background reflection），可将此作为规范化的设计原则，确保所有经验更新路径都是非阻塞的。

---

## 注意事项 / 开放问题

1. **无持久化**：OpenClaw-RL 的 session memory 是纯内存结构，重启即清空。这对 cryptotrader-ai 不适用（需要跨重启的经验持久化）。
2. **经验格式非结构化**：经验存储为自由文本字符串，无 schema 约束。与 cryptotrader-ai 的 `ExperienceRule` 结构化数据模型不兼容，但可作为 LLM 注入的序列化格式参考。
3. **OPD teacher 依赖外部 LLM**：教师评估需要调用额外模型（OpenAI API），在低延迟场景下有额外开销。
4. **consolidate 模式细节待查**：多经验池的采样策略（随机 vs 相似度检索）在 Phase 1 文档中未充分披露，留待 Phase 2 研究。
5. **论文地址**：arXiv 2603.10165，可能对应 ICML 2026 投稿，建议后续跟踪接受状态。
