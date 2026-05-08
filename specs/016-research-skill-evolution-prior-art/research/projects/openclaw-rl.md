---
name: OpenClaw-RL
url: https://github.com/Gen-Verse/OpenClaw-RL
license: Apache-2.0
tier: 1
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: true
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

## Phase 2 — 进化算法（GRPO 完整算法）

### 2.1 GRPO 算法全貌

OpenClaw-RL 的核心训练算法为 **Group Relative Policy Optimization（GRPO）**，但在 OpenClaw 个人 agent 场景中做了显著简化。

**标准 GRPO 的优势估计：**

```
对于一组 G 条响应 {r_1, ..., r_G}，第 i 条的标准化优势为：
A_i = (score_i - mean(scores)) / std(scores)
```

**OpenClaw-RL 的实际配置（`run_qwen3_4b_openclaw_rl.sh` 第 98-107 行）：**

```
--advantage-estimator grpo       # GRPO 优势估计器
--disable-rewards-normalization  # 禁用奖励归一化
--use-kl-loss                    # 启用 KL 正则
--kl-loss-coef 0.0               # 但 KL 系数为 0（实际不生效）
--eps-clip 0.2                   # PPO 下界截断
--eps-clip-high 0.28             # PPO 上界截断（非对称）
--entropy-coef 0.00              # 禁用熵正则
```

**关键简化**：在 OpenClaw 个人 agent 场景中，每个 session 通常只有 1 条响应（`n-samples-per-prompt 1`），因此无法做组间相对优势估计。系统将每个 turn 的 PRM 分数 **直接广播** 为所有响应 token 的优势值：`A_t = score`（score ∈ {+1, -1, 0}）。

**PPO 截断目标函数（`combine_loss.py` 第 131-137 行）：**

```python
ppo_kl = old_log_probs - new_log_probs     # 近似 KL
combined_advantages = w_opd * teacher_advantages + w_rl * grpo_advantages
pg_loss, pg_clipfrac = compute_policy_loss(
    ppo_kl, combined_advantages,
    args.eps_clip,      # 0.2
    args.eps_clip_high, # 0.28
)
```

**非对称截断**（ε_low=0.2, ε_high=0.28）意味着对"激进提升"比对"激进惩罚"更宽容 — 鼓励模型从正奖励中主动学习。

---

### 2.2 三种学习范式的精确区分

| 维度 | 二元 RL（Binary RL） | 在线经验蒸馏（OPD/OEL） | 混合方法（Combine） |
|------|-------------------|--------------------|-------------------|
| 训练信号 | PRM 打分 (+1/-1/0) → scalar reward | KL(student \|\| teacher) | w_rl × reward + w_opd × (teacher - student) |
| 教师模型 | 无（自我奖励） | 同一模型 + 经验增强 context | 同一模型 + 经验增强（OPD 侧）+ PRM 分数（RL 侧） |
| 奖励 hacking 风险 | 中等（PRM 可被攻击） | 低（KL 惩罚异常偏移） | 低（双信号交叉校验） |
| 推理开销 | 1x（student only） | 2x（student + teacher inference） | 2x |
| 适合场景 | 明确对/错的任务 | 偏好/风格学习 | 两者兼有 |

**OPD（On-Policy Distillation）的核心机制**：将"下一状态"中的 hint 注入 teacher 的 context（hindsight relabeling），使 teacher 的 logprob 分布更优 → 用 KL(student || teacher) 作为梯度信号，引导 student 向更优分布靠拢。教师即学生自身，区别仅在于输入 prompt 是否携带后见之明的 hint。

**OEL（Online Experiential Learning）模式划分**（`openclaw_oel_api_server.py` 第 285-287 行）：

```python
MODE_ONLINE = "online"       # OPCD 风格：实时积累经验，实时训练
MODE_EXTRACT = "extract"     # OEL 阶段 1：只收集轨迹+经验，不训练
MODE_DEPLOY = "deploy"       # OEL 阶段 2：只收集轨迹（不提取经验）
MODE_CONSOLIDATE = "consolidate"  # OEL 阶段 3：加载离线轨迹+经验池，蒸馏训练
```

**混合方法触发矩阵**（`openclaw_combine_api_server.py`，逻辑来自 WebFetch）：

| hint 提取成功 | PRM 打分有效 (±1) | 输出 |
|:---:|:---:|:---|
| 是 | 是 | Combined sample（OPD + RL 双信号） |
| 是 | 否 | OPD-only（reward=0.0，仅 KL 信号） |
| 否 | 是 | RL-only（仅标量奖励） |
| 否 | 否 | 丢弃（不提交训练） |

---

### 2.3 Turn-level Loss Masking 精确实现

Loss masking 是 OpenClaw-RL 训练信号质量控制的核心机制。完整逻辑在 `openclaw_api_server.py` 第 620-662 行：

**排除规则（`exclude = True` → `loss_mask = [0, 0, ..., 0]`）：**
1. 该 turn 没有收到下一状态（`has_next_state == False`），即 session 最后一个 turn
2. PRM 打分结果为 0（中性/不确定）

**兜底保证（at-least-one guarantee）**：如果某个 session 到目前为止已产生零个有效训练样本，则即使当前 turn 满足排除条件（score=0），也强制 `exclude = False`，确保每个 session 至少向训练器贡献一个 token（第 639-641 行）：

```python
if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
    exclude = False  # 强制提升：确保每 session 至少一个有效样本
```

**side turn 的处理**：当请求的 `turn_type != "main"` 时，请求直接转发给 SGLang，**完全不产生训练数据**（第 578-579 行）。这是 OpenClaw 多任务场景（工具调用、背景检索等辅助任务）的核心隔离机制。

---

## Phase 2 — 技能数据结构

### 3.1 "技能"在 OpenClaw-RL 中的表示方式

OpenClaw-RL **没有显式的"技能"对象**。与其他框架不同，它将技能隐式地编码在以下三个层面：

**层面 1：工具描述（OpenAI Function Calling 格式）**

技能以标准 OpenAI `tools` 参数传入 API：

```python
# openclaw_api_server.py _handle_request() 第 480-490 行
tools = body.get("tools")  # list[dict]，标准 OpenAI function schema
# 工具描述通过 tokenizer.apply_chat_template(tools=tools, ...) 注入 prompt
```

工具描述没有专门的注册表，直接随每次请求传入。Jinja2 chat template 负责将工具 schema 渲染为模型可理解的 prompt 文本。

**层面 2：经验文本（技能的隐式表示）**

在 OEL 模式下，经验字符串即为"技能知识"的载体，格式为 markdown 列表：

```
# Experience
- EXPERIENCE ITEM: When calling search_tool, always include the date range parameter
  to avoid outdated results.
- EXPERIENCE ITEM: For data analysis tasks, prefer pandas over custom loops.
```

这是一种**非结构化技能知识**——系统不区分"技能"和"偏好"，将两者都编码为可操作的 do/don't 规则。

**层面 3：轨迹数据（技能的行为证据）**

每次交互的 `turn_data` 字典包含了技能执行的完整证据链：

```python
turn_data = {
    "prompt_ids": [...],        # 技能触发的上下文
    "response_ids": [...],      # 技能执行的 token 序列
    "response_logprobs": [...], # 当前策略对该技能执行序列的置信度
    "prompt_text": "...",
    "response_text": "...",
    "messages": [...],          # 完整对话历史（含 tool_calls）
    "tools": [...],             # 本次调用的工具 schema
}
```

**对 spec-018 的启示**：OpenClaw-RL 的技能表示完全依赖 LLM 的 in-context 能力。在 cryptotrader-ai 中，如果引入 OpenClaw-RL 风格的 RL，"技能"就是已确认的交易模式，用 `ExperienceRule` 结构化表示并注入为经验文本。

---

## Phase 2 — 检索机制

### 4.1 经验检索在 RL 框架中的角色

OpenClaw-RL 的"检索"不是向量相似度搜索，而是**基于模式的规则性注入**。三种模式的精确切换条件：

**session-experience 模式（`OPENCLAW_OEL_SESSION_EXPERIENCE=1`）**

- 检索粒度：**per-turn，within-session**
- 触发条件：每个 main turn 完成后，异步启动经验提取任务（`_extract_session_experience(session_id)`）
- 检索逻辑：每次 teacher inference 时调用 `get_experience_for_turn(session_id)`，返回该 session 当前积累的经验文本
- 跨会话性：**无**，session 结束时清空（第 742-745 行）
- 适合场景：需要在单次对话内快速适应用户偏好

**replay 模式（`OPENCLAW_OEL_SESSION_EXPERIENCE=2`）**

- 检索粒度：**post-hoc，whole-session**
- 触发条件：session 结束（`session_done=True`）时，先提取全局经验，再对该 session 所有 turns **重放** teacher inference（`_replay_session()`）
- 检索逻辑：用 session 完整对话历史提取经验 → 对每个历史 turn 用该经验重新计算 teacher logprob → 补全训练样本
- 优点：Turn 1 也能受益于全 session 经验（普通 session-experience 模式下 Turn 1 没有历史可提取）

**consolidate 模式（多经验池）**

- 检索粒度：**random sampling from pool**
- 触发条件：`OPENCLAW_OEL_MULTI_EXPERIENCE=1` 且预加载了 `experience_list.txt`（每行一个经验文件路径）
- 检索逻辑：`random.choice(self._experience_pool)`（第 534 行），纯随机，无相似度计算
- 跨会话性：完全离线，经验文件由之前的 extract/deploy 阶段产生
- 硬限制：经验文本长度上限 `OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH`（默认 2048 tokens）

**三种模式的精确切换条件对照（`openclaw_oel_api_server.py` 第 372-391 行）：**

```python
self._session_experience_mode = (os.getenv("OPENCLAW_OEL_SESSION_EXPERIENCE", "0") == "1")
self._replay_mode = (os.getenv("OPENCLAW_OEL_SESSION_EXPERIENCE", "0") == "2")
self._multi_experience = (os.getenv("OPENCLAW_OEL_MULTI_EXPERIENCE", "0") == "1")
```

---

## Phase 2 — 评估（论文实验设计与防奖励 Hacking）

### 5.1 arXiv 2603.10165 核心实验设计

论文比较了三种方法在个人 agent 场景和通用 agent 场景的效果：

**个人 agent 实验**（PersonalBench 类评估）：
- 场景：数学辅导、写作助手、代码助手，注入个人偏好（输出格式、计算表达、语言风格）
- 基线：原始模型（Qwen3-4B-Thinking-2507）
- 评估方式：PRM Judge 打分（online），独立评估模型复测（offline）

**通用 agent 实验**（Terminal/GUI/SWE/Tool-call）：
- terminal-rl、swe-rl：PRM 奖励来自 shell 执行结果（exit code + diff 质量）
- toolcall-rl：PRM 奖励来自工具调用结果的语义正确性（使用 `retool` 数据集）
- gui-rl：PRM 奖励来自桌面状态截图的视觉比对（`qwen3vl_reward_agent.py`）

**奖励信号的精确来源 — "下一状态"的定义**：

```
next_state = 下一条到达 API server 的消息
role="user"  → 用户的文字回复（正面 or 负面反馈）
role="tool"  → 工具执行返回值（成功的 tool output → +1；错误信息 → -1）
```

PRM 判断依据（`_build_prm_judge_prompt()`，第 94-136 行）：
- `\boxed{1}`（好）：用户继续推进、工具返回非错误结果
- `\boxed{-1}`（坏）：用户要求 redo/retry/correction，或环境返回错误
- `\boxed{0}`（中性）：下一状态语义模糊，无法判断成功/失败

### 5.2 防止奖励 Hacking 的机制

OpenClaw-RL 通过四层防御减少奖励 hacking：

**防御 1：多数投票聚合（m=3 次独立评估）**

```python
def _majority_vote(scores):
    counter = Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0  # 平局 → 中性
    return float(top[0])
```

三次独立投票，平局返回 0（中性，被 loss masking 排除），防止随机噪声进入训练。

**防御 2：非对称截断 ε_low=0.2 / ε_high=0.28**

允许策略从正奖励中充分学习，但对负奖励的惩罚幅度有下界。防止在错误的 -1 奖励下过度退化。

**防御 3：score=0 的样本被 loss masking 排除**

中性 turn（第 633-641 行中 `exclude = not has_next_state or score == 0.0`）不参与训练，减少噪声梯度。

**防御 4：OPD/OEL 的 KL 约束**

对于 OPD 方向的训练信号，loss 是 KL(student || teacher)，天然约束 student 不会离 teacher（即自身带经验版本）太远。混合模式下 OPD 侧与 RL 侧互相校验，当两侧信号矛盾时相互抵消，自然降低攻击面。

---

## Phase 2 — Agent ↔ 技能边界（四组件接口）

### 6.1 四个异步循环的精确接口

**接口图：**

```
[OpenClaw 客户端]
       │ POST /v1/chat/completions
       │ Headers: X-Session-Id, X-Turn-Type, X-Session-Done
       ▼
[API Server（FastAPI + uvicorn）]
       │ 转发请求 → SGLang（策略模型推理）
       │ 同步返回响应（logprobs 已捕获）
       │ 异步触发 PRM 打分任务
       │ asyncio.Queue（output_queue）
       ▼
[AsyncRolloutWorker]
       │ _drain_output_queue() 等待 rollout_batch_size=16 个 group
       │ sorted by sample.index
       │ 返回 RolloutFnTrainOutput(samples=[...])
       ▼
[SLIME Trainer（Megatron-LM + Ray）]
       │ 接收 batch → 计算 GRPO/Combine 损失 → 反向传播
       │ 更新策略权重（Actor：4 GPUs）
       │ 同步权重到 SGLang Rollout（2 GPUs）
       ▼
（下一批 rollout 开始）
```

**API Server → Rollout Worker 接口（`output_queue`）：**

```python
# openclaw_rollout.py 第 76-83 行
def get_completed_groups(self) -> list[tuple]:
    completed = []
    while True:
        try:
            completed.append(self.output_queue.get_nowait())
        except queue.Empty:
            break
    return completed
```

队列元素格式：`(group_index: int, [sample: Sample])`
每个 `Sample` 包含 `tokens`, `rollout_log_probs`, `loss_mask`, `reward`, `response_length`。

**权重同步的"零协调"机制：**

API Server 通过 `submission_enabled` threading.Event 实现**暂停/恢复**协议：

```python
# openclaw_rollout.py 第 65-74 行
def pause_submission(self):
    if self._submission_enabled.is_set():
        self._submission_enabled.clear()
        self._server.purge_record_files()  # 清空已收集的记录文件

def resume_submission(self):
    if not self._submission_enabled.is_set():
        self._submission_enabled.set()
```

训练器在每个 rollout 批次前调用 `resume_submission()` 开始收集，批次凑满后调用 `pause_submission()` 暂停，然后执行训练 + 权重同步，完成后再 resume。这期间 API Server 仍在处理推理请求（只是不向训练队列提交），**推理不中断**。

**重要**：SGLang 的 rollout 副本（2 GPUs）持有"旧版本"权重在服务推理，Megatron actor（4 GPUs）持有"当前版本"权重在训练。两者之间有**单向异步权重同步**：训练完成后，Trainer 将权重 broadcast 给 SGLang rollout。这就是"零协调"的本质 — 并非完全无协调，而是**训练期间推理不等待**，仅在权重同步时有短暂暂停（由 `submission_enabled` 门控）。

---

## Phase 2 — 工程实现

### 7.1 SLIME 训练后端集成

SLIME（arXiv 同组工作）是 OpenClaw-RL 的训练基础设施，基于 **Megatron-LM + SGLang + Ray** 三层架构。

**组件分工（`run_qwen3_4b_openclaw_rl.sh` 第 18-28 行）：**

```bash
NUM_GPUS=8
ACTOR_GPUS=4   # Megatron 训练（策略更新）
ROLLOUT_GPUS=2 # SGLang 推理（采样生成）
PRM_GPUS=2     # SGLang PRM（奖励打分）
```

**SLIME 接入点（自定义函数路径，第 143-146 行）：**

```bash
--rollout-function-path openclaw_rollout.generate_rollout_openclaw
--custom-generate-function-path openclaw_api_server.generate
--custom-rm-path openclaw_api_server.reward_func
```

三个入口完全解耦：`generate_rollout_openclaw` 等待 API 驱动的真实用户交互数据，不主动生成 rollout；`generate` 用于 eval 模式的批量评估；`reward_func` 直接透传 `sample.reward["score"]`（奖励由 PRM 打分已写入样本）。

**GPU 需求确认**：

- 最小配置：**8× 消费级 GPU**（如 A100/H100），官方默认 8 卡
- 实际分布：4 卡训练 + 2 卡推理 + 2 卡 PRM
- 无 GPU 替代方案：项目同时支持 **云端 Tinker 平台**（API 化，不需要本地 GPU），README 明确声明"Zero API or Zero GPU"（两种路径选其一）
- 对于 cryptotrader-ai（本机部署）：**直接使用 SLIME 训练后端不可行**，需要另寻轻量替代

### 7.2 Per-token Log-probabilities 的捕获与传递

**捕获层（API Server，第 554-558 行）：**

SGLang 返回 `logprobs.content[i].logprob`（每个 token 一个 float）：

```python
response_logprobs = _extract_logprobs_from_chat_response(choice)
# 对齐：若 logprobs 长度 != response_ids 长度，用 0.0 填充
if len(response_logprobs) > len(response_ids):
    response_logprobs = response_logprobs[: len(response_ids)]
elif len(response_logprobs) < len(response_ids):
    response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))
```

**存储层（Sample 对象）：**

```python
sample.rollout_log_probs = turn_data["response_logprobs"]  # list[float]，长度 = response token 数
```

**训练层（Megatron 批次）：**

Trainer 将 `rollout_log_probs` 作为"旧策略"基准（`old_log_probs`），用于 PPO 重要性采样比率 `exp(new_log_probs - old_log_probs)`。启用 `--use-rollout-logprobs` 时直接使用，否则用 Megatron 重新计算参考 logprobs（后者更精确但有训练/推理 tokenization 不一致的风险）。

**Top-K logprobs（OEL 教师蒸馏，`oel_distillation_loss.py`）：**

OEL 模式额外捕获教师模型的 top-K token 分布（默认 K=50），存储为 `[response_len, K]` 张量，用于 KL(student || teacher) 的 top-K 近似计算（见第 134-305 行）。尾部概率通过 `log(1 - sum(exp(topk_logprobs)))` 估计，防止顶 K 之外的词汇贡献被忽略。

### 7.3 推理-训练并行的具体机制

**双副本模型权重布局：**

```
SGLang Rollout（2 GPUs，TP=2）── 持有当前版本权重（cold copy）
Megatron Actor（4 GPUs，TP=4）── 持有训练中的权重（hot copy）
Megatron Ref（4 GPUs 或 CPU offload）── 持有初始版本权重（frozen ref）
```

**无阻塞服务的实现**（`openclaw_api_server.py` 第 284-286 行）：

```python
if not owner.submission_enabled.is_set():
    raise HTTPException(status_code=503, detail="submission paused for weight update")
```

权重同步期间（submission paused），API Server 返回 503 而非无限等待，OpenClaw 客户端负责重试。这是唯一的"协调点"，持续时间极短（仅权重 broadcast 耗时）。

---

## 借鉴建议（Phase 2 增补）

### 建议 5：cryptotrader-ai realized PnL → GRPO Reward Signal 的适配方案

OpenClaw-RL 的 PRM 奖励结构（{+1, -1, 0}）和 cryptotrader-ai 的 realized PnL 在语义上高度对齐：

| OpenClaw-RL PRM 信号 | cryptotrader-ai 等价 |
|-------------------|--------------------|
| `\boxed{1}`（next_state 显示正进展） | 已平仓交易 PnL > 阈值（如 +1%） |
| `\boxed{-1}`（next_state 显示需要 redo） | 已平仓交易 PnL < 阈值（如 -0.5%） |
| `\boxed{0}`（中性/持仓中） | 持仓中的 unrealized PnL（不确定） |

**直接对接的可行性**：从信号格式看，**完全可行**。PnL 信号对应 OpenClaw-RL 中的 `sample.reward = {"score": pnl_score}`，loss masking 机制可直接复用（只对已平仓交易产生梯度）。

**需要的适配层（4 个）：**

1. **奖励归一化器**：PnL 是连续值（如 -5% 到 +5%），需要映射到 {+1, -1, 0} 或保留连续值（OpenClaw-RL 禁用了归一化，但支持连续奖励 — `reward_func` 直接透传 score）。建议保留连续值（`score = tanh(pnl / threshold)`），比三值量化信息更丰富。

2. **时间延迟处理器**：PnL 信号天然是延迟的（持仓期 = hours to days），而 OpenClaw-RL 假设下一状态在秒级到达。需要一个 `pending_trade_buffer`，存储"开仓决策 → 等待平仓" 的 turn_data，在平仓时才触发打分（对应 `_fire_prm_scoring()` 的延迟调用）。

3. **多回合聚合器**：一次完整交易可能跨越多个 LangGraph 节点（data → agents → verdict → execute）。需要决定哪个节点的"响应"作为训练 turn（建议：`verdict` 节点的最终决策 token 序列，因为该节点产生可归因的 buy/sell/hold 输出）。

4. **基准线构造**：GRPO 需要组内相对比较。单模型单样本时，可以通过 **跨周期比较**（同一市场条件下不同时间段的多个决策）构造 group，用相对于基准（如简单均值策略的 PnL）的超额收益作为优势估计。

**最大障碍**：SLIME 训练后端需要 GPU，cryptotrader-ai 部署在用户本机。解决方向：将 GRPO 替换为**纯推理侧的经验演化**（OEL consolidate 模式的无 GPU 近似），即用 PnL 信号筛选高质量 trajectory，提取经验文本，注入下一次决策的 prompt — 这正是 cryptotrader-ai 已有的 `learning/reflect.py` 思路，但信号来源从 LLM 评分改为 realized PnL。

### 建议 6：OEL consolidate 模式 → 交易周期性决策的最佳范式

三种范式中，**consolidate 模式最适合 cryptotrader-ai 的"周期性交易决策"场景**，原因：

- **session-experience**：适合单对话内快速调整，但交易决策是独立事件（每次决策是新对话），无 session 内积累
- **replay 模式**：适合"回看整个对话后重新标注"，但交易的结果（平仓 PnL）可能在决策结束后数小时才知晓，不符合实时 replay 假设
- **consolidate 模式**：交易周期 = OEL 的 deploy/extract 阶段（收集轨迹），PnL 到达 = consolidate 阶段（用离线信号蒸馏经验），与 cryptotrader-ai 的"周期性 reflect → 更新 ExperienceRule"流程完全对应

**实施路径**：
1. deploy 阶段：每次交易决策（`nodes/verdict.py` 输出）保存为轨迹 JSONL
2. consolidate 触发条件：有新的 realized PnL（平仓事件）
3. 经验提取：用 PnL 信号和 EXPERIENCE_UPDATE_PROMPT_V2 风格的 prompt 提取 `ExperienceRule`
4. 注入：下一次 `verbal_reinforcement()` 时注入，复用现有 GSSC 管道

### 建议 7：SLIME 的 GPU 需求替代 — 本机无 GPU 部署策略

SLIME 需要 GPU，但 cryptotrader-ai 部署在用户本机，**需要 GPU-free 替代**：

| OpenClaw-RL 组件 | cryptotrader-ai 等价（无 GPU）|
|-----------------|---------------------------|
| Megatron Actor（梯度更新）| **跳过**（不做在线 fine-tune，只做经验文本演化）|
| SGLang Rollout（策略推理）| OpenAI API / Anthropic API（已有）|
| PRM Judge（奖励打分）| Realized PnL（硬信号，无需 LLM 评判）|
| 经验蒸馏（OEL KL loss）| `learning/reflect.py` 经验文本更新（纯 LLM in-context）|

结论：**cryptotrader-ai 应采用 OEL consolidate 模式的纯 prompt-side 近似**，而非 SLIME 完整训练管道。这意味着放弃权重更新，转而专注于经验文本的高质量演化（已有 `ExperienceRule` + `_verify_rules()` 体系）。当未来 cryptotrader-ai 有专用 GPU 时，可以接入 SLIME 训练后端，为当前的 prompt-side 体系增加梯度学习层。

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
4. **consolidate 模式采用纯随机采样**：Phase 2 研究确认，多经验池采样策略为 `random.choice(pool)`（第 534 行），无相似度检索，对 cryptotrader-ai 而言可升级为基于市场 regime 的相似性检索（复用 `learning/regime.py` + `journal/search.py` 的 `search_by_regime()`）。
5. **论文地址**：arXiv 2603.10165，可能对应 ICML 2026 投稿，建议后续跟踪接受状态。
