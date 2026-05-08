---
name: Hermes Agent Self-Evolution
url: https://github.com/NousResearch/hermes-agent-self-evolution
license: MIT
tier: 1
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: true
---

# Hermes Agent Self-Evolution — NousResearch

## 架构概览

NousResearch 的 Hermes Agent Self-Evolution 是一个独立优化框架，通过 DSPy + GEPA（遗传-帕累托提示进化）
对 Hermes Agent 的技能文件、工具描述、系统提示和代码进行自动演化。无需 GPU 训练，全程通过 API 调用完成，
每次优化运行成本约 $2–10。

**仓库统计**：2.9k stars，314 forks，MIT 许可，Python 100%。

**目录结构（关键部分）**：

```
evolution/
  core/          # config.py, constraints.py, dataset_builder.py, fitness.py, external_importers.py
  prompts/       # 提示相关组件
  skills/        # 技能定义
  tools/         # 工具集成
  code/          # 代码级演化（Phase 4）
  monitor/       # 监控
generate_report.py
PLAN.md
```

智能体本体在外部 Hermes Agent 安装中（通过环境变量指向），本仓库为独立优化流水线，
通过 `batch_runner.py` + `trajectory.py` 与智能体交互。

---

## 提示组装（Phase 1 重点）

### 系统提示构造（`agent/prompt_builder.py`）

系统提示由两类组件拼接而成，位于 hermes-agent 主仓库（非本优化仓库）：

**可演化组件**（DSPy 参数化，供 GEPA 优化）：

| 组件名 | 用途 | 演化方向 |
|---|---|---|
| `DEFAULT_AGENT_IDENTITY` | 人格与行为特征描述 | 语气、优先级、处理方式 |
| `MEMORY_GUIDANCE` | 持久记忆使用模式说明 | 何时保存、保存什么 |
| `SESSION_SEARCH_GUIDANCE` | 历史上下文检索触发条件 | 检索触发条件本身可演化 |
| `SKILLS_GUIDANCE` | 技能加载与缓存逻辑 | 触发条件与加载时机 |
| `PLATFORM_HINTS` | 平台特定格式规则 | 按平台分别优化 |

**固定组件**（不参与演化）：

- 自动生成的技能索引（skill index）—— 运行时由技能目录派生
- 来自记忆存储的用户记忆块（user memory blocks）—— 用户私有数据
- 项目上下文文件（project context files）—— 项目特定

**演化流程**：优化器通过 Phase 3（系统提示演化）将可演化段落包装为独立 DSPy Signature
字段，每个字段可独立变异，但评估时以完整系统提示为单位打分，防止局部最优破坏整体。

**每段尺寸约束**：不得超过当前版本长度的 120%（`max_prompt_growth=0.2`），
防止演化向冗长方向漂移（"evolutionary drift toward verbose solutions"）。

### 技能注入机制（Phase 1 已实现）

技能以 `SKILL.md` 文件形式存储，路径为 `skills/<category>/<skill-name>/SKILL.md`。注入流程：

1. 技能文件在初始化时作为用户消息加载进上下文窗口
2. GEPA 将技能 body 文本视为可变异字符串参数进行演化（frontmatter 冻结）
3. 演化后的变体通过 PR 部署为新版本——**从不在进行中的会话内热替换**
4. 所有更改在下一次新鲜会话（fresh session）时生效

**尺寸约束**：技能文件默认 ≤15,000 字节（`max_skill_size=15_000`）。

**缓存合规性**：YAML frontmatter 字段（`name:`、`description:`）保持冻结，
只有 body markdown 参与演化，以保证对话缓存（conversation caching）兼容性。

---

## 记忆 ↔ 技能连接（Phase 1，Phase 2 补全）

### SessionDB 双角色（`hermes_state.py`）

`hermes_state.py` 中的 `SessionDB` 对象同时服务于优化流水线和技能上下文：

**用于优化时**：
- 从真实使用模式中挖掘数据，构建评估数据集
- 用户纠错（corrections）标记为"误选信号"（misselection signals）
- 失败会话分析作为失败案例输入演化

**用于技能上下文时**：
- 持久记忆通过 `SESSION_SEARCH_GUIDANCE` 中定义的触发条件被检索
- 记忆质量直接影响技能有效性
- 演化同时改进记忆检索逻辑和技能文本本身

### 记忆–技能耦合路径

```
用户会话
  → trajectory.py（捕获执行轨迹）
  → SessionDB（存储记忆 + 失败案例）
      ├── [优化路径] dataset_builder.py → GEPA 演化技能/提示
      └── [运行路径] SESSION_SEARCH_GUIDANCE 触发 → 检索相关记忆注入上下文
```

记忆检索逻辑本身是一个可演化的提示段（`SESSION_SEARCH_GUIDANCE`），
这意味着系统可以演化"何时检索记忆"的判断逻辑，而不仅仅是"检索什么"。

### 工具描述演化中的记忆影响（Phase 2 补全）

工具 Schema 存于 `tools/registry.py`，描述字段随每次 API 调用发送。
Phase 2 实现的演化要点：

- 将工具 `description` 字符串包装为 DSPy Signature 字段（最大 500 字符）
- 参数说明 `description` 字段独立约束（最大 200 字符）
- **跨工具评估**：所有工具描述必须一起参与评估，防止一个工具的改进
  "窃取"另一个工具的选择份额（cross-tool evaluation 设计）
- **误选信号来源**：SessionDB 中代理选用 `terminal(grep)` 而非 `search_files`
  的历史会话被提取为高价值训练样本
- `registry.register()` 调用结构（参数名称、类型、必填字段）**冻结不变**，
  只有自然语言描述文本参与演化

---

## 进化算法（Phase 2）

### GEPA 核心机制（ICLR 2026 Oral）

GEPA（Genetic-Pareto Prompt Evolution）是 DSPy 内置的反射性提示优化器，
论文于 2025 年 7 月发布，2026 年被 ICLR 接收为 Oral 演讲。
核心思想：将遗传算法的 Pareto 多目标选择与 LLM 自然语言反射（reflection）融合，
在不微调模型权重的前提下优化任意文本参数。

**三大核心机制**：

#### 1. 反射性变异（Reflective Mutation）
每次迭代中，GEPA 从 Pareto 前沿选取候选提示，用 LLM 对最近一批执行轨迹
（rollouts）打分并生成自然语言反馈（"为什么这条提示在这类任务上失败了"），
然后将该反馈注入下一轮变异请求。子代提示从祖先继承，累积多轮高层次教训
（"每次变异的候选提示都来自一个祖先，累积了从观察和 LLM 反馈中得出的高层次教训"）。

**关键区别于 RL**：变异方向来自自然语言推理，而非梯度信号。
以最多 35× 少的 rollouts，在 AIME-2025 上超过 GRPO 约 6%（最高 20%）；
超过 MIPROv2 超过 10%（+12% on AIME-2025）。

#### 2. Pareto 前沿维护（Pareto Frontier）
GEPA 不只保留全局最优一个候选，而是维护一组 **Pareto 候选集**——
"在至少一个评估实例上达到最高分的候选集合"。
从 Pareto 前沿采样下一个待变异候选时，采样概率正比于该候选的 **覆盖度**（coverage），
即它在哪些实例上是最佳的。这保证了探索（不同策略互补）和鲁棒性（防止局部最优）。

#### 3. 系统感知交叉（System-Aware Merge/Crossover）
GEPA 支持跨谱系合并：将来自不同演化分支、在不同实例上表现最优的模块
组合成新候选，比单点变异更快收敛到高质量解。

### 在 Hermes 中的 DSPy 集成

Hermes 通过三步将 GEPA 与技能演化对接（源文件：
`evolution/skills/evolve_skill.py`，`evolution/skills/skill_module.py`）：

**步骤一：将技能包装为 DSPy 模块**（`skill_module.py` 第 84-114 行）

`SkillModule(dspy.Module)` 内嵌 `TaskWithSkill` Signature（三字段：
`skill_instructions` + `task_input` → `output`），将 `self.skill_text`（技能 body）
作为 GEPA 的优化目标参数，frontmatter 冻结不参与。

**步骤二：调用 GEPA 优化器**（`evolve_skill.py` 第 157-166 行）

`dspy.GEPA(metric=skill_fitness_metric, max_steps=iterations)` +
`.compile(baseline_module, trainset=..., valset=...)`。
GEPA 每次迭代对 rollouts 打分、生成反馈、变异 `skill_text`；
Hermes 默认 10 次迭代，种群大小 5。

**步骤三：提取演化结果**（`evolve_skill.py` 第 184-185 行）

`optimized_module.skill_text` 即经 GEPA 优化的文本，
再由 `reassemble_skill(frontmatter, evolved_body)` 重组为完整 `SKILL.md`。

**降级策略**：若 DSPy 版本不含 GEPA，自动切换到
`dspy.MIPROv2(metric=..., auto="light")`，确保生产可用性。

### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `iterations` | 10 | GEPA 迭代轮数（`max_steps`） |
| `population_size` | 5 | 每轮维护的候选数量 |
| `optimizer_model` | `openai/gpt-4.1` | GEPA 反射/变异使用的模型 |
| `eval_model` | `openai/gpt-4.1-mini` | LLM-as-judge 打分使用的模型 |
| `judge_model` | `openai/gpt-4.1` | 数据集生成使用的模型 |

---

## 技能数据结构（Phase 2）

### SKILL.md 完整 Schema

技能文件是存储于 `skills/<category>/<skill-name>/SKILL.md` 的 Markdown 文件，
采用 YAML frontmatter + Markdown body 结构（源码：`skill_module.py` 第 15-55 行）：

```yaml
---
name: github-pr-workflow        # 必填，技能唯一标识符（目录名匹配）
description: 完整的 PR 生命周期...  # 必填，技能用途摘要（≤500 字符）
version: 1.1.0                  # 版本号（可选，Hermes 官方技能使用）
author: Hermes Agent            # 作者（可选）
license: MIT                    # 许可证（可选）
tags: [GitHub, Pull-Requests]   # 标签（可选）
---

# 正文（Body）
## 分支创建
...（完整的 Markdown 程序指令）
```

**硬约束**（由 `constraints.py` 验证）：

| 约束 | 实现 | 失败处理 |
|---|---|---|
| 大小上限 | `len(text) <= max_skill_size`（默认 15,000 字节） | 立即拒绝 |
| 增长上限 | `(新长度 - 基线长度) / 基线长度 <= 0.20` | 立即拒绝 |
| 非空检查 | `text.strip()` 非空 | 立即拒绝 |
| 结构完整性 | frontmatter 含 `---` 分隔符 + `name:` + `description:`，且均在前 500 字符内 | 立即拒绝 |
| 测试通过率 | `pytest tests/ -q`，超时 300 秒，returncode == 0 | 立即拒绝 |

所有约束均为**硬约束**（binary pass/fail），无软约束设计。

### 工具描述演化 Schema

工具描述的演化约束独立于技能（均在 `EvolutionConfig` 中配置）：

| 字段类型 | 约束 | 原因 |
|---|---|---|
| 工具 `description` | ≤ 500 字符 | 每轮 API 调用都发送，字符数乘积累加 |
| 参数 `description` | ≤ 200 字符 | 同上，控制 token 成本 |
| Schema 结构（参数名称、类型） | 完全冻结 | 破坏调用方契约 |

### `find_skill` 双重检索逻辑

技能定位通过两阶段实现（`skill_module.py` 第 58-81 行）：

1. **直接匹配**：在 `skills/` 目录下递归搜索 `SKILL.md`，匹配 `parent.name == skill_name`
2. **模糊匹配**：读取每个技能文件前 500 字符的 frontmatter，检查 `name:` 字段是否匹配

---

## 检索机制（Phase 2）

### 评估数据集三大来源

Hermes 没有使用向量 embedding 进行技能检索，而是通过三种来源构建评估数据集，
再由 GEPA 的反射机制内生地"学会"何时及如何检索技能。

**来源 A：合成生成（首选，解决冷启动）**

使用强模型（默认 `gpt-4.1`）读取技能文件全文，通过 `GenerateTestCases` DSPy Signature
生成 `(task_input, expected_behavior, difficulty, category)` 四元组。
JSON 解析失败时使用正则表达式 `r'\[.*\]'` 回退提取。
默认生成 20 个样本，按 50%/25%/25% 分割为 train/val/holdout（`dataset_builder.py`）。

**来源 B：真实会话挖掘（`external_importers.py`，三个数据源）**

| 导入器 | 数据路径 | 可用信息 |
|---|---|---|
| `ClaudeCodeImporter` | `~/.claude/history.jsonl` | 用户输入（无响应） |
| `CopilotImporter` | `~/.copilot/session-state/*/events.jsonl` | 用户+助手完整对话 |
| `HermesSessionImporter` | `~/.hermes/sessions/*.json` | 用户+助手+工具上下文 |

三个导入器均内置**秘密检测**（`SECRET_PATTERNS`），过滤 API 密钥、Token、私钥等
17 类敏感模式（Anthropic、OpenAI、GitHub、Slack、AWS 等）。

**二阶段相关性过滤**（`RelevanceFilter`）：
1. **低成本关键词预筛**：技能名、单词（>3 字符）及技能文件前 500 字符关键词，要求至少 2 个命中
2. **LLM 相关性打分**：`ScoreRelevance` DSPy Signature，输出 `{relevant, expected_behavior, difficulty, category}` JSON，使用 `gpt-4.1-mini` 打分，JSON 解析采用基于括号深度计数的稳健提取算法

**来源 C：手动黄金集**（Golden Sets）

存于 `datasets/skills/<skill-name>/golden.jsonl` 的手工标注数据，
单文件时自动按 50/25/25 分割，多文件时直接加载 `train.jsonl`/`val.jsonl`/`holdout.jsonl`。

### `SESSION_SEARCH_GUIDANCE` 的可演化性

该提示段决定"智能体何时去搜索历史会话"，本身作为 Phase 3 系统提示演化的目标之一。
这意味着系统不只是优化"检索什么"，而是通过 GEPA **演化"何时触发检索"的判断逻辑**。

---

## 评估（Phase 2）

### 适应度函数详细实现

适应度计算分两层（源文件：`evolution/core/fitness.py`）：

**训练期快速打分**（`skill_fitness_metric`，`fitness.py` 第 107-136 行）：

用于 GEPA 每次迭代中的 rollout 打分，追求速度而非精度：
`score = 0.3 + 0.7 × (expected_words ∩ output_words) / |expected_words|`
纯关键词重叠（词袋）近似，返回 0.0-1.0，非空输出基础分 0.3。

**精确 LLM-as-judge 打分**（`LLMJudge`，第 34-104 行）：

用于 holdout 集最终评估，调用 `gpt-4.1-mini` 作为法官（judge model），
通过 `JudgeSignature` DSPy Signature 输出四个字段：

| 维度 | 权重 | 说明 |
|---|---|---|
| `correctness` | 50% | 输出是否正确解决了任务 |
| `procedure_following` | 30% | 是否遵循了技能的程序/步骤 |
| `conciseness` | 20% | 是否在 token 预算内简洁完成 |
| `length_penalty` | 扣分，最高 -0.3 | 超过 90% 容量时线性递增惩罚 |

**复合分**计算：`composite = 0.5×C + 0.3×P + 0.2×Con - length_penalty`

**长度惩罚斜率**：制品大小超过 `max_size` 的 90% 时，
每额外 1% 容量扣 0.03 分，上限 0.3（即 100% 容量时 -0.3 分）。
这防止演化向冗长方向漂移。

**LLM 输出解析容错**（`_parse_score`，第 139-146 行）：
类型为 `int/float` 时直接转换；为字符串时 `strip()` 后转换；解析失败时默认 0.5（中立分）。

**法官看到的上下文**：`task_input`、`expected_behavior`（评分标准）、
`agent_output`（实际响应）、`skill_text`（代理使用的技能文本）——
最后一项使评判者能直接判断代理是否遵循了技能的指导方针。

### 评估流水线（多门控架构）

Hermes 使用一个精心设计的多层门控流水线（`PLAN.md`），将适应度评估
和基准测试分层递进，控制成本同时确保质量：

```
候选变体
  │
  ├── Gate 1: pytest（必须 100% 通过，300 秒超时）
  │
  ├── Gate 2: TBLite 快速子集（20 任务，~20 分钟）
  │
  ├── 适应度打分：任务专用数据集（快速关键词近似）
  │
  ▼（仅前 3 名候选进入）
  │
  ├── Gate 3: 完整 TBLite（100 任务）
  │
  ├── Gate 4: YC-Bench fast_test（长期连贯性检查）
  │
  ▼
最佳候选 → PR + 完整指标
```

**关键原则**：基准测试是**门控**（gate），不是适应度信号。
一个技能改进了 20% 但 TBLite 下降 5% 的变体会被拒绝。

### 跨工具适应度防"抢占"

Phase 2（工具描述演化）中，评估**始终对所有工具描述一起进行**，而非单独评估。
数据集包含"混淆对"（confusing pairs）——同一任务可能有两个合理工具，
但应优先选其中一个。适应度函数对任何工具选择率的回归施加惩罚，
防止 `search_files` 的描述改进通过"窃取"来自 `read_file` 的选择份额来虚报进步。

### 误选信号（Misselection Signal）

定义：用户或回测纠错事件，标志着代理在某次任务中做出了非最优选择。

来源与量化：
1. SessionDB 中代理用 `terminal(grep)` 而非 `search_files` 的历史记录
2. 用户说"不，用 X 代替"的纠正语句
3. 基准测试中因工具选择失误导致任务失败的轨迹

这些样本被 LLM-as-judge 确认后归入 `source="sessiondb"` 的高价值训练集，
作为 GEPA 反射变异的失败案例输入。

---

## Agent ↔ Skill 边界（Phase 2）

### 两仓库分离架构

```
hermes-agent/                    # 主仓库（MIT）
  skills/<category>/<name>/
    SKILL.md                     ← 演化目标（只读输入，写出到 git 分支）
  tools/registry.py              ← 工具描述注册（只读输入）
  agent/prompt_builder.py        ← 系统提示段（只读输入）
  batch_runner.py                ← 并行评估执行器（被调用）
  agent/trajectory.py            ← 轨迹收集（被读取）
  hermes_state.py (SessionDB)    ← 真实使用数据挖掘（被查询）

hermes-agent-self-evolution/     # 优化仓库（MIT，本仓库）
  evolution/                     ← 优化流水线（无状态，幂等）
  datasets/                      ← 评估数据集（.gitignore）
  output/                        ← 演化结果快照（.gitignore）
```

优化仓库对 hermes-agent 仓库是**只读**的；
它读取当前版本，生成演化变体，输出 git 分支 + PR，
**绝不直接修改 hermes-agent 的工作树**。

### Hermes Agent 接口契约（外部仓库）

`batch_runner.py`（hermes-agent 主仓库）扮演评估执行器角色：
接收一批评估任务，并行运行代理，收集每次执行的轨迹（trajectories）。
`agent/trajectory.py` 负责记录执行路径供 GEPA 的反射性分析使用。

**重要**：`batch_runner.py` 和 `trajectory.py` 均位于 hermes-agent 主仓库，
而非本优化仓库。评估时优化仓库调用主仓库的基础设施，两者通过
`HERMES_AGENT_REPO` 环境变量或约定路径（`~/.hermes/hermes-agent`）耦合。

### 工具注册 vs 技能注册的差异

| 维度 | 工具注册（`tools/registry.py`）| 技能注册（`skills/` 目录）|
|---|---|---|
| 格式 | Python dict（代码中注册） | SKILL.md 文件（文件系统） |
| 演化粒度 | 自然语言 `description` 字段 | 整个 Markdown body |
| 结构约束 | 参数名/类型/必填完全冻结 | frontmatter 冻结，body 自由演化 |
| 部署方式 | 代码更改 → PR → merge | 文件更改 → PR → merge |

### 演化运行时 vs 代理运行时的隔离

**隔离是强制的**，实现机制如下：

1. **时间隔离**：演化只在离线（offline）状态下运行，输出 PR 供人工审核，
   从不在代理运行时期间执行优化
2. **会话隔离**：演化后的版本**只在下一次新鲜会话（fresh session）生效**，
   正在进行中的会话永远使用启动时加载的版本（"No evolved content is ever hot-swapped
   into an active conversation"）
3. **文件系统隔离**：演化结果写入独立的 `output/<skill>/<timestamp>/` 目录，
   通过 `git checkout -b evolve/<target>-<timestamp>` 创建专属分支，
   PR 创建后才影响主仓库

---

## 工程实现（Phase 2）

### 批量运行与并行评估

当前实现的并行度**仅在工具层面**（通过 `batch_runner.py` 并行运行代理任务）。
`evolve_skill.py` 的 holdout 评估循环本身是**顺序执行**（第 213-223 行），
对每个样本依次运行 baseline 和 evolved 模块后取平均。
GEPA 内部的 rollout 并行化由 DSPy 框架负责，与 Hermes 实现无关。

### Git 谱系追踪

每次演化运行产生专属分支（`PLAN.md` 第 706-721 行）：
分支命名为 `evolve/<target>-<timestamp>`，提交信息包含
before/after 得分、GEPA 迭代数、候选评估数、数据集规模。
PR body 附完整 diff、train/val/holdout 三集得分、运行成本和被拒绝的约束违规记录。

**回滚**：`git revert` 即可——演化即是提交历史。

每次运行产生带时间戳快照目录（`evolve_skill.py` 第 256-284 行）：
`output/<skill-name>/<YYYYMMDD_HHMMSS>/`，含 `evolved_skill.md`、
`baseline_skill.md`、`metrics.json`（含耗时、迭代数、模型名）。

### 成本模型（$2-10/run）

成本构成（`generate_report.py` + `PLAN.md` 实测数据）：

| 步骤 | 模型 | 典型调用数 | 成本估算 |
|---|---|---|---|
| 数据集生成（合成） | gpt-4.1 | 1 次调用 × 20 样本 | ~$0.10-0.20 |
| GEPA 反射变异 | gpt-4.1 | 10 迭代 × 5 候选 × N 样本 | ~$1-4 |
| LLM-as-judge 打分 | gpt-4.1-mini | 训练期：关键词近似（免费） | — |
| holdout 评估 | gpt-4.1-mini | 5 holdout 样本 × 2（baseline+evolved） | ~$0.05 |
| 相关性过滤（sessiondb 模式） | gemini-2.5-flash | 最多 150 条候选消息 | ~$0.50-2 |

实测案例（`generate_report.py`）：arxiv 技能（10,175 字符），3 个训练样本，
2 个 holdout，1 轮优化，成本 **< $0.50**，耗时 < 60 秒。
标准 10 迭代完整运行约 $2-10，具体取决于评估数据集大小和模型选择。

### Phase 4 Darwinian Evolver（AGPL v3）

Darwinian Evolver 被设计为**仅通过外部 CLI 调用**，永远不作为 Python 模块导入
（`PLAN.md`："Darwinian Evolver: AGPL v3 ⚠️ — external CLI only, no Python imports"）。

其核心机制（从 PLAN.md 架构推断）：
- 将每个工具实现文件包装为 `GitBasedOrganism`（代码文件即"生物体"）
- 变异操作：LLM 读取具体失败案例后提出代码修改建议（非随机变异）
- 每次变异自动提交到 git 分支，形成完整演化谱系
- 适应度函数：`pytest 通过率`（硬性门控，100% 要求）+ TBLite 基准分 + 具体 bug
  复现脚本是否修复（目标导向适应度）
- 保护约束：函数签名冻结、`registry.register()` 调用冻结、
  错误处理覆盖率不得下降

**对 cryptotrader-ai 的影响**：AGPL v3 意味着任何通过 Python import 方式
集成 Darwinian Evolver 的代码必须以 AGPL v3 开源整个项目。
但其**算法思想**（GitBasedOrganism、测试驱动适应度、目标导向代码变异）
可以自由借鉴并用 MIT 代码重新实现。

---

## 借鉴建议（Phase 1 + Phase 2）

以下建议基于完整研究，针对 cryptotrader-ai spec 016 / spec 018：

### 高优先级借鉴（Phase 1 来源）

1. **分段系统提示（Sectioned System Prompt）**
   将系统提示拆分为具名段落（`MEMORY_GUIDANCE`、`SKILLS_GUIDANCE` 等），
   每段独立可测试、可演化。cryptotrader-ai 的代理提示（agents/base.py 中各代理的系统提示）
   可参照此模式重构，使各段具备独立字符预算和语义约束，
   对应 spec 018 FR 中的"提示段落独立演化"需求。

2. **技能–会话隔离原则**
   演化后的经验规则/技能提示应在下一个交易周期生效，
   不应在当前决策链路（`build_trading_graph` 执行期间）中途注入。
   对应 `nodes/agents.py` 中 GSSC 管道的经验注入时机管控。

3. **记忆检索触发逻辑可演化化**
   `SESSION_SEARCH_GUIDANCE` 的设计思路：把"何时检索历史经验"本身作为可优化参数。
   对应 cryptotrader-ai 的 `learning/verbal.py` + `learning/context.py`，
   可将 `gather_packets()` 的触发条件参数化为 DSPy Signature 字段。

4. **误选信号作为评估数据**（Phase 1 来源）
   回测实际结果 vs. 预测标记为误选信号，自动构建演化数据集。
   与 `ExperienceRule`（`maturity`、`empirical_rate`）高度兼容，
   应用落点：`learning/reflect.py` 的 `_verify_rules()` 可使用此信号校准规则质量。

### 高优先级借鉴（Phase 2 来源）

5. **GEPA Pareto 前沿选择 → 经验规则多目标优化**
   GEPA 的 Pareto 前沿维护机制直接对应 `ExperienceRule` 的多维质量评估：
   `win_rate` 和 `confidence` 可构成双目标 Pareto 前沿，
   防止高胜率低置信度的规则"抢占"低胜率高置信度规则的覆盖份额。
   应用落点：`learning/reflect.py` 的 `_merge_memories()` 中引入 Pareto 选择替代简单加权平均，
   对应 spec 018 FR 中的"规则演化选择策略"。

6. **两阶段数据集构建（合成 + 会话挖掘）**
   Hermes 的合成生成（冷启动）+ SessionDB 挖掘（真实数据）双轨策略，
   与 cryptotrader-ai 的 `_run_graph()` 回测执行轨迹完全类比。
   应用落点：在 `backtest/engine.py` 中记录每个决策周期的 `(market_snapshot, action, outcome)`
   三元组，对应 Hermes 的 `EvalExample(task_input, expected_behavior)`；
   利用 `ExperienceRule` 失败案例作为 GEPA 的反射输入。

7. **多维适应度函数（correctness × procedure × conciseness）**
   Hermes 的 `FitnessScore` 三维加权设计（50%+30%+20%）+长度惩罚斜率，
   可直接移植到 cryptotrader-ai 的经验规则质量评分。
   应用落点：在 `learning/reflect.py` 中将规则评分扩展为
   `(预测准确率 × 0.5) + (程序遵循度 × 0.3) + (表达简洁性 × 0.2) - 长度惩罚`，
   替代当前仅使用 `empirical_rate` 的一维评分。

8. **约束验证器架构（硬约束全部二值化）**
   `ConstraintValidator` 的设计（所有约束硬性通过/拒绝，无软约束）
   可移植到 `ExperienceRule` 的提交前校验层。
   应用落点：在 `learning/reflect.py` 的 `_verify_rules()` 中添加对应的
   硬约束（规则长度上限、样本量下限、语义漂移检测），
   对应 spec 018 中的"规则提交约束"需求。

9. **外部会话导入（ClaudeCodeImporter）**
   Hermes 直接读取 `~/.claude/history.jsonl` 挖掘真实使用样本，
   说明 Claude Code 用户的操作历史可以成为评估数据源。
   对应 spec 018：考虑从 cryptotrader-ai 的实盘/模拟盘交易日志中
   挖掘"用户干预"（人工覆盖 AI 决策）事件作为误选信号。

10. **Git 谱系标准提交格式**
    Hermes 的演化提交格式（含 before/after 分数、迭代数、数据集规模）
    应用于 cryptotrader-ai 的经验规则版本管理：
    每次 `_merge_memories()` 执行后在 `ExperienceRule.source` 字段记录演化元数据。

### 需要适配的差异

- Hermes 以 `SKILL.md` 文本文件存储技能，cryptotrader-ai 使用结构化 `ExperienceRule` 对象；
  需要设计从 `ExperienceRule` 到提示文本的序列化层（`_serialize_rule_to_prompt()`），
  使 GEPA 能将规则 body 视为可变异字符串。
- Hermes 的 `skill_fitness_metric` 使用关键词重叠近似，
  cryptotrader-ai 有更精确的 `empirical_rate` 历史数据可用，应优先使用真实回测结果而非 LLM 近似。
- Hermes 的 15KB 技能文件上限对应 cryptotrader-ai 的 token 预算约束（已有 `_estimate_tokens()`）。
- Hermes 的 Darwinian Evolver（AGPL v3）不可直接集成，
  但其 `GitBasedOrganism` + 测试驱动适应度的算法设计可在 MIT 代码中重新实现。

---

## 说明 / 开放问题

### 已解决（Phase 2）

1. **~~`evolution/core/fitness.py` 未能获取源码~~（已解决）**：
   已确认为 LLM-as-judge（`JudgeSignature` DSPy Signature）+ 关键词近似双层适应度，
   权重为 correctness 50%、procedure_following 30%、conciseness 20%，加长度惩罚。

2. **~~`evolution/prompts/` 目录内容未知~~（已确认为占位符）**：
   该目录的 `__init__.py` 仅含 `"Phase placeholder: prompts evolution."`，
   Phase 3（系统提示演化）尚未实现。

3. **~~`trajectory.py` 轨迹格式未知~~（已部分确认）**：
   `trajectory.py` 位于 hermes-agent 主仓库，由 `batch_runner.py` 调用，
   收集执行路径供 GEPA 反射性分析使用，但具体 schema 仍位于外部仓库。

### 仍未解决

4. **Hermes Agent 本体代码细节**：`agent/prompt_builder.py` 和 `hermes_state.py`（SessionDB）
   位于外部 hermes-agent 仓库，未克隆检查。SessionDB 的确切查询接口
   影响 cryptotrader-ai 是否能参照相同模式构建回测会话索引。

5. **GEPA 内部并行化机制**：DSPy GEPA 内部的 rollout 并行度和批量评估策略
   不在 Hermes 仓库中，依赖 DSPy 库实现。当评估数据集较大（>50 样本）时，
   是否会自动批量化尚不确定。

6. **TBLite / TerminalBench2 基准的具体任务格式**：
   这些基准在 hermes-agent 主仓库中，Phase 2 未覆盖。
   对应 cryptotrader-ai：等价的"基准门控"应使用什么指标（回测夏普比率回归？
   胜率下降阈值？）尚未设计。

7. **Phase 4 Darwinian Evolver 的变异算子细节**：
   Darwinian Evolver 的具体代码变异策略（是 diff-based 还是 rewrite-based？
   如何确定变异范围？）仅从 PLAN.md 架构推断，未能获取源码。
