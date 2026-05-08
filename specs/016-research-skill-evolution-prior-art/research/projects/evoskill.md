---
name: EvoSkill
url: https://github.com/sentient-agi/EvoSkill
license: Apache-2.0
tier: 2
last_accessed: 2026-05-07
phase_1_complete: true
phase_2_complete: true
---

# EvoSkill — sentient-agi

## Architecture Overview

EvoSkill 是一个自动化框架，用于从编码代理的失败尝试中发现并创建可复用的 agent 技能（skills），以提升跨平台编码代理的性能。支持的 agent 包括 Claude Code、OpenCode、OpenHands、Goose 和 Codex CLI。

核心五阶段迭代循环：

```
Base Agent → Proposer → Generator → Evaluator → Frontier Update
     ↑                                                   |
     └───────────────────────────────────────────────────┘
```

1. **Base Agent**：当前最优程序尝试基准测试问题集
2. **Proposer**：分析失败样本，提出技能或提示词的修改方案
3. **Generator**：根据提案写入新技能文件或重写系统提示词
4. **Evaluator**：在验证集上对新变体打分（pairwise 比较）
5. **Frontier**：维护 Top-N 最优程序版本，跨迭代保留

主要语言：Python 73.8%，Jupyter Notebook 25.9%。版本 v1.1.0（2026-05 发布），697 stars。

---

## Prompt Assembly（Phase 1）

### 提案提示词构建（`build_proposer_query()`）

`src/loop/runner.py` 中的 `build_proposer_query()` 将以下上下文注入 Proposer 的 LLM 调用：

| 注入内容 | 来源 |
|---|---|
| 失败样本（agent 输出 + 真实答案） | 训练集运行结果，score < 0.8 的样本 |
| 反馈历史（历次提案及结果） | `.evoskill/feedback_history.md` |
| 当前活跃技能列表 | `.claude/skills/*/SKILL.md` 文件名枚举 |
| 任务约束 | 初始化时传入的领域限制 |
| 演化模式路由 | `mode: "skill_only"` / `"prompt_only"` |
| 截断级别（0/1/2） | 若 Proposer 失败则逐级压缩上下文 |

**关键设计**：失败上下文 + 历史反馈 + 现有技能三者共同构成提案上下文，Proposer 做出 `prompt vs skill` 的二元路由决策。

### 当前最优程序（"program"）的表示

存储于 `.evoskill/config.toml` 中的 `program.yaml`，包含：
- 系统提示词（system prompt）
- 工具规范（tool specifications）

程序版本通过 git 分支管理：`program/iter-skill-N`，Top-N 变体打 `frontier/*` 标签。

### 提案输出 Schema

```python
# src/schemas/proposer.py — ProposerResponse
optimize_prompt_or_skill: Literal["prompt", "skill"]
proposed_skill_or_prompt: str   # 实际内容（技能描述或提示词文本）
justification: str               # 修改理由

# src/schemas/skill_proposer.py — SkillProposerResponse
action: Literal["create", "edit"]  # 默认 "create"
target_skill: str | None           # edit 时必填：目标技能名
proposed_skill: str                # 技能的高层描述
justification: str                 # 为何此技能能填补能力缺口
related_iterations: list[str]      # 引用的历史迭代 ID
```

### Feedback Descent（`src/feedback_descent.py`）

核心优化抽象，解耦 Proposer 与 Evaluator：

```python
# Proposer Protocol
generate_initial(problem) -> artifact
propose(current_best, feedback_history) -> candidate

# Evaluator Protocol
evaluate(current_best, candidate) -> EvaluationResult(preferred: bool, rationale: str)
```

终止条件：连续 3 次未改进，或达到最大迭代数（默认 10）。每次迭代无论是否改进都记录反馈；改进成功时清空反馈历史（避免历史污染新起点）。

---

## Memory ↔ Skill（Phase 1 lite）

### 持久化记忆层

| 文件 | 内容 | 作用 |
|---|---|---|
| `.evoskill/feedback_history.md` | 每次提案 + 理由 + 结果 + 得分 + 当时活跃技能 | 跨迭代告知 Proposer 历史决策轨迹 |
| `.evoskill/loop_checkpoint.json` | 采样状态（各类别轮询偏移） | 支持 `--continue` 精确断点续跑 |
| `program/iter-skill-N` git 分支 | 完整程序快照（提示词 + 技能） | 可回滚、可比较任意历史版本 |
| `frontier/*` git 标签 | Top-N 最优程序 | 父程序选择策略的候选池 |

### 技能文件结构

技能存储于 `.claude/skills/<skill-name>/SKILL.md`，为可复用的指令包，包含 frontmatter 元数据。写入后通过 `skill_frontmatter_normalization` 适配各 agent 平台（OpenCode / Goose / Codex CLI 各有差异）。

### 关键记忆设计决策

- **改进后清空反馈历史**：防止旧失败模式污染新起点的提案上下文
- **active skills 注入提案**：Proposer 可感知当前已有技能，避免重复创建
- **`related_iterations` 字段**：显式关联历史迭代，支持跨轮归因

---

---

## Phase 2 — 进化算法精确流水

### 2.1 四角色精确分工与接口

EvoSkill 的进化循环由五个 agent 协同完成（`src/agent_profiles/` 各自对应一个系统提示 + 输出 schema）：

| 角色 | 输入 | 输出 Schema | 职责边界 |
|---|---|---|---|
| **Base Agent** | 训练问题 | `AgentResponse{final_answer, reasoning}` | 执行基准任务，产生答案供打分 |
| **Skill Proposer** | 失败样本 + 反馈历史 + 现有技能列表 | `SkillProposerResponse` | 分析失败模式，决定 create / edit 技能 |
| **Prompt Proposer** | 失败样本 + 反馈历史 + 当前提示词 | `PromptProposerResponse` | 分析失败模式，提出提示词修改方向 |
| **Skill Generator** | Proposer 的描述 + justification | `ToolGeneratorResponse` | 实际写出 `.claude/skills/<name>/SKILL.md` |
| **Prompt Generator** | Proposer 的修改方向 + 原始提示词 | `PromptGeneratorResponse{optimized_prompt}` | 重写 `src/agent_profiles/base_agent/prompt.txt` |

`skill_only` 模式激活 Skill Proposer + Skill Generator 链路；`prompt_only` 模式激活 Prompt Proposer + Prompt Generator 链路。两条链路完全并行隔离，互不干扰。

**Proposer → Generator 的接口契约**（`src/loop/helpers.py`）：

- `build_skill_query_from_skill_proposer(proposer_trace)` — 将 `proposed_skill + justification` 格式化为 Generator 的输入提示词
- `build_prompt_query_from_prompt_proposer(proposer_trace, original_prompt)` — 将提示词修改方向 + 原始提示词传给 Prompt Generator
- `build_proposer_query(...)` — 将失败样本 + 反馈历史 + 现有技能列表组装为 Proposer 的完整上下文

### 2.2 失败样本驱动的算法精确定义

**"失败"的定义**：`score < 0.8`（硬阈值，`runner.py` 第 183 行）：

```python
avg_score = self.scorer(question, agent_answer.strip().lower(), answer.strip().lower())
if avg_score < 0.8:
    failures.append((trace, agent_answer, answer, category))
```

该阈值针对 multi-tolerance 加权平均分：精确匹配权重最高（1.0），10% 容忍权重最低（0.33）。分数落在 0.8 以下意味着答案"明显错误"，而非"接近但有偏差"。

**采样策略**：轮询（round-robin）跨类别采样，避免每轮只看同一类失败：

```python
# src/loop/runner.py — 核心采样逻辑
for j in range(n_cats_this_iter):          # 默认每轮采 3 个类别
    cat_idx = (self._category_offset + j) % n_cats
    cat = categories[cat_idx]
    for _ in range(self.config.samples_per_category):   # 默认每类 2 个样本
        sample_idx = self._per_cat_offset[cat] % len(pool)
        question, answer = pool[sample_idx]
        test_samples.append((question, answer, cat))
        self._per_cat_offset[cat] += 1
self._category_offset += n_cats_this_iter
```

每轮最多产生 `categories_per_batch × samples_per_category = 3 × 2 = 6` 个训练样本，其中失败的那部分（`score < 0.8`）全部送入 Proposer。Proposer 的任务是识别**跨样本的共性模式**，而非针对单一失败案例"打补丁"（这是提示词的明确指令）。

**多级截断回退机制**（`_mutate_with_fallback`）：

```
Level 0（full）:    每个 trace 头 60k + 尾 60k 字符，全部失败样本，全量反馈历史
Level 1（moderate）: 头 20k + 尾 10k，最多 3 个失败，最近 20 行反馈
Level 2（aggressive）: 头 5k + 尾 2k，最多 2 个失败，最近 5 行反馈
最终回退:            取 trace 最短的单个失败，Level 2 截断
```

仅当上下文超出模型限制或 Proposer 返回解析失败时触发逐级压缩。

### 2.3 pairwise 比较打分的精确实现

EvoSkill **不使用** LLM-as-judge 做 pairwise 比较——而是用**客观准确率分数**（accuracy score）在验证集上直接比较新旧版本：

```python
# runner.py — 评估子程序
child_score = await self._evaluate(self.val_data)  # 验证集准确率

added = self.manager.update_frontier(child_name, child_score, max_size=self.config.frontier_size)
if added:
    no_improvement_count = 0
else:
    self.manager.discard(child_name)
    no_improvement_count += 1
```

`FeedbackDescent`（`src/feedback_descent.py`）中的 `Evaluator Protocol` 定义了 pairwise 接口：

```python
class Evaluator(Protocol[T]):
    async def evaluate(self, current_best: T, candidate: T) -> EvaluationResult:
        # EvaluationResult: preferred_for_candidate: bool, rationale: str
```

在具体实现中，`preferred_for_candidate = child_score > parent_score`，`rationale` 记录得分差。这是**纯数值比较**，而非语义比较。

`ProgramManager.update_frontier()` 的录入规则（`src/registry/manager.py`）：

```python
def update_frontier(self, name: str, score: float, max_size: int = 5) -> bool:
    # 1. 先把 score 写进 program.yaml 的 metadata 字段，git commit 保存
    # 2. 查当前 frontier 成员列表（通过 frontier/* 标签）
    # 3. frontier 未满 → 直接录入
    # 4. frontier 已满 → 仅当 score > worst_score 时，踢出最差，录入新成员
    # 5. 返回 True = 录入成功，False = 不够格
```

### 2.4 history 清空的精确触发条件

**触发时机**：当且仅当 `child_score > worst_score_in_frontier`（即 candidate 成功进入 frontier）。

`FeedbackDescent` 的核心算法（`src/feedback_descent.py`）：

```
Initialize: best ← generate_initial(problem)
For each iteration:
  → candidate ← propose(best, feedback_history)
  → result ← evaluate(best, candidate)
  → feedback_history.append(result)        ← 无论输赢都记录
  → if candidate wins:
        best ← candidate
        feedback_history = []              ← 清空！进入新基线
        no_improvement_count = 0
  → else:
        no_improvement_count += 1
        if no_improvement_count >= no_improvement_limit: break
```

**清空的设计理由**（`docs/architecture.md` 原文）："When something works, you forget all the old failures. Why? Because you have a new baseline now — what failed against the old version might not be relevant anymore."

**只追加，不清空的情形**：candidate 进入 frontier 但不是"新最优"（`outcome = "kept"` 而非 `"improved"`），反馈历史**不清空**——因为 `runner.py` 的 `append_feedback` 在评估后统一调用，清空逻辑仅在 `FeedbackDescent` 的 `best ← candidate` 赋值时触发。

**磁盘级清空**（`runner.py`）：在非 `continue_mode` 的新运行启动时，若 `reset_feedback=True`（默认 True），直接删除 `.evoskill/feedback_history.md`：

```python
if not self.config.continue_mode:
    if self.config.reset_feedback and self._feedback_path.exists():
        self._feedback_path.unlink()
```

这是**跨运行**的隔断，与迭代内的清空是两个独立机制。

---

## Phase 2 — 技能数据结构

### 3.1 技能文件 Schema

技能的完整目录结构（源自 `.claude/skills/skill-creator/SKILL.md`）：

```
.claude/skills/<skill-name>/
├── SKILL.md                  (必须)
│   ├── YAML frontmatter      (必须: name + description)
│   └── Markdown body         (触发后加载的执行指令)
├── scripts/                  (可选：确定性脚本，可直接执行无需读入上下文)
├── references/               (可选：按需加载的参考文档)
└── assets/                   (可选：输出产物，不加载入上下文)
```

**frontmatter 精确格式**（仅两个必填字段，不得扩展）：

```yaml
---
name: skill-name
description: >
  完整触发描述，包含 "what" 和 "when to use"。
  description 是触发机制——body 加载前 agent 只看这里决定是否激活。
---
```

**三级渐进加载（Progressive Disclosure）**：

1. `name + description`（~100 词）：始终在上下文中，供 agent 决定是否触发
2. `SKILL.md body`（< 5k 词）：技能触发后加载
3. `references/` 和 `scripts/`（无上限）：Claude 按需读取，scripts 可直接执行绕过上下文

这是控制 token 消耗的核心设计：大型参考文档（数据库 schema、API 文档）放 `references/`，避免污染主上下文。

**跨 agent 适配（`skill_frontmatter_normalization`）**：

写入技能后，若当前 SDK 为 OpenCode / OpenHands / Goose / Codex，自动调用 `normalize_project_skill_frontmatter()` 为每个平台生成兼容格式。Claude Code 使用 `SKILL.md` 原生格式；其他 agent 可能需要不同字段名或不同加载机制。

### 3.2 feedback_history.md 的精确格式

由 `append_feedback()`（`src/loop/helpers.py`）写入，每条记录结构：

```markdown
## iter-skill-3
**Proposal**: 提出的技能描述或提示词修改（前 50 字符截断用于 git commit message）
**Justification**: 为何此修改能解决失败
**Outcome**: IMPROVED (score: 0.5132 (+0.0932))
**Active Skills**: percentage-calculator, table-reader
**Failure Category**: methodology
**Root Cause**: 简短根因描述
```

| 字段 | 可选性 | 说明 |
|---|---|---|
| `## iter-skill-N` | 必填 | 迭代标识符（含 `skill` / `prompt` 类型前缀） |
| `**Proposal**` | 必填 | 技能描述或提示词变更 |
| `**Justification**` | 必填 | 改动理由 |
| `**Outcome**` | 可选 | `IMPROVED` / `NO_IMPROVEMENT` / `DISCARDED` + 得分 + 增量 |
| `**Active Skills**` | 可选 | 该轮评估时活跃的技能列表（诊断用） |
| `**Failure Category**` | 可选 | 失败类型分类（`methodology` / `formatting` / 等） |
| `**Root Cause**` | 可选 | 根因简述 |

格式设计为"对 LLM 友好"的叙事式 Markdown，而非结构化 JSON——目的是让 Proposer 在 few-shot 阅读时自然理解历史决策轨迹。

### 3.3 `related_iterations` 字段的回溯算法

`SkillProposerResponse.related_iterations: list[str]` 是 Proposer 在输出中**显式声明**的字段，不是自动计算的。

```python
# src/schemas/skill_proposer.py
class SkillProposerResponse(BaseModel):
    related_iterations: list[str] = Field(default_factory=list,
        description="relevant past iterations referenced in the proposal")
```

Proposer 的系统提示明确要求："Reference any related DISCARDED iterations and explain how your proposal differs"（`build_proposer_query` 生成的提示末尾）。因此 `related_iterations` 是 **LLM 自主回溯**的结果：Proposer 读取 `feedback_history.md` 中的历史记录，识别出本次提案借鉴或刻意区分的历史迭代，将其 ID 列入该字段。

没有 embedding 检索，没有向量相似度——完全依赖 Proposer 的自然语言理解能力，对历史做因果推断。这使得 `related_iterations` 可解释（Proposer 在 `justification` 中同步说明关联原因），但也依赖 LLM 不遗漏重要历史。

---

## Phase 2 — 检索与 Proposer 上下文构建

### 4.1 现有技能列表的选择与排序

Proposer 接收的"现有技能列表"**无嵌入检索，直接枚举**（`build_proposer_query` 函数，`src/loop/helpers.py`）：

```python
skills_dir = Path(project_root) / ".claude" / "skills"
existing_skills = []
if skills_dir.exists():
    for skill_dir in skills_dir.iterdir():
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            existing_skills.append(skill_dir.name)
skills_list = "\n".join([f"- {s}" for s in existing_skills]) or "None"
```

以文件系统迭代顺序列出技能名（`str` 排序，无语义过滤）。Proposer 看到的是技能目录名，而非 `SKILL.md` 的完整内容——避免技能数量增多时的上下文膨胀。Proposer 判断"是否已有能覆盖此场景的技能"完全依赖 `description` 字段的命名和语义（这正是 `description` 质量至关重要的原因）。

Proposer 提示词的明确指令：

```
1. Check if any EXISTING skill should have handled these failures
2. If yes → propose EDITING that skill (action="edit", target_skill="skill-name")
3. If no → propose a NEW skill (action="create")
```

这是一个**两阶段决策**：先判断 `edit vs create`，再描述内容。`edit` 路径要求精确命名目标技能（通过 `target_skill` 字段），Generator 收到 edit 指令后会先读取现有 `SKILL.md` 再做增量修改。

### 4.2 是否使用 Embedding

**不使用 Embedding**。EvoSkill 的整个检索层是基于 LLM 的直接理解：

- 技能列表：文件名枚举（无语义向量）
- 历史反馈：完整文本截断注入（无向量检索）
- 失败样本匹配：不检索历史相似样本，而是实时采样新的训练样本

这是一个有意为之的简单化选择：技能数量预期在 10-30 个范围内（一次进化实验不会产生数百个技能），LLM 直接阅读枚举列表完全可行，引入 embedding 反而增加了系统复杂度和失败节点。

---

## Phase 2 — 评估机制

### 5.1 四种打分器与判定规则

| 打分器类型 | 触发配置 | 判定逻辑 |
|---|---|---|
| `multi_tolerance`（默认） | `[scorer] type = "multi_tolerance"` | 5 个容忍度加权平均：`Σ score(tol) × weight(tol)` |
| `exact` | `type = "exact"` | 大小写不敏感字符串匹配，有括号缩略词剥离 |
| `llm` | `type = "llm"` | LLM-as-judge（SealQA 专用，GPT-5-mini via OpenRouter） |
| `script` | `type = "script"` | Shell 命令打分（LiveCodeBench：Docker 执行代码，比对 stdout） |

**multi_tolerance 加权权重计算**（精确公式，`runner.py`）：

```python
TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]
# weight = 1 / (1 + 20 * tol)
# 0.0%  → 1.00（精确匹配，最高权重）
# 1.0%  → 0.83
# 2.5%  → 0.67
# 5.0%  → 0.50
# 10.0% → 0.33
```

最终得分是加权平均值，范围 [0.0, 1.0]。低于 0.8 判定为失败。

**SealQA LLM-as-judge 细节**（`src/evaluation/sealqa_scorer.py`）：

1. 填充含 CORRECT / INCORRECT / NOT_ATTEMPTED 示例的评分模板
2. 调用 GPT-5-mini（OpenRouter），获取 "A" / "B" / "C"
3. "A" = 1.0，其余 = 0.0

用于语义复杂答案（"San Francisco" 应等价于 "San Francisco, California"），纯字符串匹配会产生假阴性。

**LiveCodeBench 代码执行打分**（`src/evaluation/livecodebench/`）：

1. 正则提取 Python 代码块
2. 从 ground_truth 解析 JSON 测试用例
3. `llm_sandbox.SandboxSession` 在 Docker 中执行，每个测试用例 5 秒超时
4. Pass@1：所有测试用例全部通过 → 1.0，否则 0.0

### 5.2 评估数据集的构建（失败样本转化）

数据集分割（默认比例，可在 `.evoskill/config.toml` 中覆盖）：

```toml
[dataset]
train_ratio = 0.18    # 18%：采样失败，送给 Proposer
val_ratio   = 0.12    # 12%：评估新程序是否进步
# 剩余 70%：保留为测试集（不参与演化循环）
```

**"用户失败样本如何转化为训练数据"的机制**：

数据来源是用户提供的 CSV（`question_column + ground_truth_column`）。EvoSkill 不从生产日志中自动收集失败——用户需要准备一个包含正确答案的评估数据集。训练集的 18% 在运行时被实时分配为"Proposer 输入"：每轮从训练池中按类别轮询抽取若干样本，让 Base Agent 实际运行，得分 < 0.8 的样本成为本轮 Proposer 的输入。

这意味着：**同一个训练样本可能在多轮迭代中被反复使用**（`_per_cat_offset[cat]` 循环取余），直到该类别被完全覆盖，然后重头循环。

### 5.3 评估失败时的回滚机制

当新程序 `child_score <= worst_frontier_score` 时（即 `update_frontier` 返回 False）：

```python
self.manager.discard(child_name)   # 删除 program/iter-skill-N 分支
no_improvement_count += 1
```

`discard()` 的完整逻辑（`src/registry/manager.py`）：

1. 若当前 HEAD 在该分支，先 `git checkout` 到父程序或其他安全分支
2. `git branch -D program/iter-skill-N`（强制删除）
3. 若存在对应 `frontier/iter-skill-N` 标签，同步删除

回滚后，下一次迭代从 frontier 中重新 `select_parent()`，切回最优分支继续演化。程序快照通过 `.claude/program.yaml` 的 `parent` 字段维护完整谱系，即使中间版本被删除，frontier 中的版本仍可通过 `get_lineage()` 追溯祖先链。

**缓存失效与评估一致性**：`RunCache` 以 `git tree hash` 为键，仅当技能文件实际变化时才失效——`.claude/skills/` 修改会改变 tree hash，从而强制重新评估；仅更新 `program.yaml` 中的 score metadata 则不影响 hash，避免不必要的重跑。

---

## Phase 2 — Agent ↔ Skill 边界与 SDK 集成

### 6.1 Sentient-AGI SDK 集成方式

EvoSkill 通过 `src/harness/` 抽象层集成 Claude Code SDK（主路径），并支持 OpenCode / OpenHands / Goose / Codex CLI。

**Claude Code SDK 集成**（`src/harness/_claude_executor.py`）：

```python
client = ClaudeSDKClient(options)   # spawn Claude Code 进程
client.query(query)                  # 发送问题
messages = client.receive_response() # 流式接收所有消息
# messages[-1] 是 ResultMessage，包含 structured_output（JSON schema 强制输出）
```

Agent 配置通过 `ClaudeAgentOptions` 传入：

```python
ClaudeAgentOptions(
    system_prompt = {"type": "preset", "preset": "claude_code", "append": system},
    output_format = {"type": "json_schema", "schema": AgentResponse.model_json_schema()},
    allowed_tools = ["Read", "Write", "Bash", "Skill", ...],
    cwd = project_root,             # agent 工作目录，即包含 .claude/skills/ 的仓库根
    setting_sources = ["user", "project"],
    permission_mode = "acceptEdits",
)
```

**关键**：`cwd` 设置为 git 仓库根，使 Claude Code 自动发现 `.claude/skills/` 目录并在推理时加载相关技能。

**韧性设计**：

- 每次 SDK 调用最多重试 3 次，间隔指数退避（30s / 60s / 120s）
- 单次调用 20 分钟超时（`asyncio.timeout(1200)`）
- 验证集并发上限由 `concurrency`（默认 4）控制，通过 `asyncio.Semaphore` 实现

### 6.2 Skill 是 per-agent 还是共享池

**per-program（不是 per-agent）**：技能绑定到 git 分支（`program/iter-skill-N`），每个 program 分支有自己的 `.claude/skills/` 快照。

```
program/base         → .claude/skills/ 为空（或仅含初始技能）
program/iter-skill-1 → .claude/skills/percentage-calculator/
program/iter-skill-2 → .claude/skills/percentage-calculator/ + table-reader/
```

当 `ProgramManager.switch_to("iter-skill-2")` 执行 `git checkout` 时，工作区的 `.claude/skills/` 随之切换为该版本的快照。不同程序分支的技能**完全独立**，没有共享池。

但技能**可以跨任务迁移**：SealQA 上进化出的技能可以零迁移（zero-shot）用于 BrowseComp，论文实测提升 +5.3%。迁移方式是手动复制 `.claude/skills/` 目录，而非自动向量检索。

### 6.3 Proposer 与 Generator 的接口契约（精确签名）

```python
# Proposer → 决策层（不写文件）
# skill_only 路径
SkillProposerResponse:
    action: Literal["create", "edit"]   # "edit" 时 target_skill 必填
    target_skill: str | None            # validator: edit 时不能为 None
    proposed_skill: str                 # 高层描述（Generator 的输入）
    justification: str
    related_iterations: list[str]       # 显式引用历史迭代

# prompt_only 路径
PromptProposerResponse:
    proposed_prompt_change: str         # 修改方向描述
    justification: str

# Generator → 执行层（实际写文件）
# skill_only：Generator 调用 Bash/Write 工具写 .claude/skills/<name>/SKILL.md
# prompt_only：Generator 调用 Write 工具重写 src/agent_profiles/base_agent/prompt.txt
```

Proposer 不写任何文件——它只产生"高层提案"（`proposed_skill` 是描述，不是内容）。Generator 收到提案后自主决定文件内容。两者之间没有结构化中间格式，Generator 的输入就是 Proposer 输出的自然语言描述。

---

## Phase 2 — 工程实现细节

### 7.1 git 分支策略的精确实现

**分支命名规则**（`src/registry/manager.py`）：

```
program/base                    ← 基线程序
program/iter-skill-{N}          ← skill_only 模式第 N 次迭代
program/iter-prompt-{N}         ← prompt_only 模式第 N 次迭代
```

**frontier 标签规则**：

```
frontier/base                   ← 若 base 进入 Top-K
frontier/iter-skill-7           ← 第 7 轮迭代进入 frontier
```

**版本生命周期**（精确时序）：

```
1. manager.switch_to(parent)         → git checkout program/iter-skill-2
2. parent_config = get_current()     → 读 .claude/program.yaml
3. child_config = parent_config.mutate("iter-skill-5")
   # ProgramConfig.mutate() 设置: parent="program/iter-skill-2", generation=3
4. manager.create_program("iter-skill-5", child_config, parent="iter-skill-2")
   # → git checkout program/iter-skill-2
   # → git checkout -b program/iter-skill-5
   # → 写入 .claude/program.yaml（含 parent 字段）
   # → git add .claude/program.yaml
   # → git commit "Create program: iter-skill-5"
5. [Generator 写入 .claude/skills/<name>/SKILL.md]
6. manager.commit("iter-skill-5: <proposal[:50]>")
   # → git add .claude/
   # → git commit（仅当有变更时执行）
7. child_score = await _evaluate(val_data)
8. manager.update_frontier("iter-skill-5", child_score, max_size=3)
   # → git checkout program/iter-skill-5
   # → config.with_score(child_score) → 更新 metadata.score
   # → git add .claude/program.yaml → git commit "Update score: 0.5132"
   # → 若得分足够：git tag frontier/iter-skill-5
   # → 若超出 frontier_size：删除最差 frontier 标签
   # → 若不够格：返回 False
9. [若 False] manager.discard("iter-skill-5")
   # → git branch -D program/iter-skill-5
```

**EvoSkill 写保护**：所有分支创建均以 `program/` 为前缀，所有标签均以 `frontier/` 为前缀。用户的工作分支（如 `main`）从不被触碰。切换分支前若有未提交变更，自动 stash（含 untracked 文件，`-u` 标志），切换后自动 pop stash。

**谱系追踪**（`get_lineage(name)`）：

```python
# 从 program.yaml 的 parent 字段逐级回溯
["iter-skill-5", "iter-skill-2", "base"]
```

即使中间版本被 `discard`，只要 `program.yaml` 仍在 frontier 分支中，谱系信息就完整保存（`git show branch:file` 读取不需要 checkout）。

### 7.2 并行评估的 worker 模型

评估并发通过 `asyncio.Semaphore` 控制（`src/evaluation/evaluate.py`）：

```python
semaphore = asyncio.Semaphore(max_concurrent)   # 默认 4，loop 内传 concurrency 参数

async def run_one(question, ground_truth) -> EvalResult:
    async with semaphore:                        # 限流
        async with asyncio.timeout(1020):        # 17 分钟/题超时
            trace = cache.get(...) or await agent.run(question)
            cache.set(question, trace, ...)
            return EvalResult(...)

tasks = [run_one(q, gt) for q, gt in items]
results = await tqdm_asyncio.gather(*tasks)      # 并发启动，semaphore 限制实际并发数
```

**运行阶段的并发区分**：

| 阶段 | 并发 | 原因 |
|---|---|---|
| 训练采样（失败检测） | `asyncio.gather(*)` 无 semaphore | 每轮最多 6 个样本，全并发 |
| 验证集评估 | `Semaphore(concurrency=4)` | 验证集可达数百题，需防 API 超载 |
| Proposer 调用 | 串行（单次调用） | 需等待完整失败上下文聚合后才调用 |
| Generator 调用 | 串行（接 Proposer） | 依赖 Proposer 输出 |

**RunCache（`src/cache/run_cache.py`）**：以 `(question, response_model, sdk, model, git_tree_hash)` 为复合键，保存到磁盘 JSON。`git_tree_hash` 覆盖 `.claude/` 目录的树哈希——技能文件变更会改变此哈希，从而强制重跑；仅更新 `program.yaml` score metadata 的 git commit 不影响树哈希，缓存仍有效。

**`eval_full` 的增量持久化**（`evoskill eval` 命令专用）：每完成一道题即追加写入 pickle 文件；断点续跑时跳过已完成的 index，只重新运行超时或报错的题目。

### 7.3 生产部署经验（697 stars 项目观察）

**三种执行模式**：

| 模式 | 命令 | 适用场景 |
|---|---|---|
| 本地直接执行 | `evoskill run` | 开发调试 |
| Docker 容器化（BYOC） | `evoskill run --docker` | 隔离依赖，支持 `DOCKER_HOST=ssh://` 远程服务器 |
| Daytona 托管沙箱 | `evoskill run --daytona` | 云端分布式，资源上限 4 vCPU / 8 GB RAM |

**断点续跑**（生产关键功能）：`loop_checkpoint.json` 在每轮迭代末精确保存 `category_offset` 和 `per_cat_offset`（每类别独立偏移），保证 `--continue` 恢复后采样状态与中断前完全一致，不会重复评估已评估的样本。

**成本追踪**：每轮迭代结束打印本轮成本和累计成本（`$iter_cost / $total_cost`），全部从 `AgentTrace.total_cost_usd` 聚合。

**版本历史**：v0.1.0-alpha → v1.0.0-alpha → v1.1.0（2026-05 发布）。3 个 release 均有对应 git tag。

**benchmark 实测结果**（arXiv 2603.02766）：

| 基准测试 | 起点 | 终点 | 提升 |
|---|---|---|---|
| OfficeQA（财务 QA） | 60.6% | 67.9% | +7.3% |
| SealQA（含噪检索 QA） | 26.6% | 38.7% | +12.1% |
| BrowseComp（零迁移，SealQA 技能） | — | — | +5.3% |

跨任务零迁移（SealQA → BrowseComp）是 EvoSkill 技能泛化能力的重要证据。

---

## Borrow Recommendations（Phase 1 only）

以下模式值得在 cryptotrader-ai spec 016（技能演化）中借鉴：

1. **`ProposerResponse` 二元路由**：`prompt vs skill` 的显式分类让下游生成器路由清晰，可类比为 cryptotrader-ai 中 `strategy_rule vs agent_prompt` 的路由。

2. **`feedback_history.md` 叙事式记忆**：以自然语言 Markdown 记录历次提案及结果，而非结构化 DB——对 LLM 友好，Proposer 可直接读取作为 few-shot 上下文。对应 cryptotrader-ai 的 `learning/reflect.py` 经验记忆写入机制有参考价值。

3. **改进后清空反馈历史**：避免跨阶段污染。当前 `ExperienceMemory` 合并策略可考虑类似的"阶段隔断"设计。

4. **`related_iterations` 显式回溯**：技能提案显式引用触发灵感的历史迭代 ID，比隐式 embedding 检索更可解释，适合需要审计轨迹的交易场景。

5. **截断层级（0/1/2）**：当上下文超长时逐级压缩（全量 → 中等 → 激进截断），可直接用于 cryptotrader-ai 的 GSSC 管道中的 token 预算控制。

6. **git 分支作为程序版本控制**：每次迭代快照存为独立 branch，frontier 打 tag——这是一种"无额外存储开销"的版本化方案，比自定义 DB 更易于调试和回滚。

---

## Notes / Open Questions

- **技能粒度**：EvoSkill 的技能是 agent 级别的操作指令（如"如何调用某工具"），而 cryptotrader-ai 的 spec 016 技能可能更偏向交易策略规则——两者的粒度差异需在 spec 设计阶段明确。
- **评估器设计**：EvoSkill 用 pairwise LLM 比较作为评估器，cryptotrader-ai 有真实回测收益可用——后者的评估信号更强，但延迟也更高，需要权衡。
- **跨 agent 迁移**：EvoSkill 强调技能跨 agent 可迁移性，cryptotrader-ai 是单体系统，此特性不直接适用，但其 frontmatter 标准化思路可用于多模型适配。
- **Phase 2 待深读**：`src/loop/helpers.py`（采样逻辑）、`src/evaluation/`（评分器实现细节）、`examples/officeqa/`（端到端示例）。
