---
name: EvoSkills
url: https://github.com/EvoScientist/EvoSkills
license: Apache-2.0
tier: 2
last_accessed: 2026-05-07
phase_1_complete: true
phase_2_complete: true
---

# EvoSkills — EvoScientist

> **论文引用**：Lyu et al. (2026) *EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery*，arXiv:2603.08127。
> **基准排名**：2026 年 4 月 18 日 DeepResearch Bench 排名 #1。

---

## 一、Architecture Overview（架构总览）

EvoSkills 是 EvoScientist 框架的官方技能仓库，面向端到端**科研自动化**场景。核心思路：将科研流程拆分为 13 个自包含技能模块（每个模块一个 `SKILL.md`），由三类 Agent 协作驱动。

### 1.1 三类核心 Agent

| Agent | 角色 | 核心职责 |
|-------|------|----------|
| **Researcher Agent (RA)** | 想法生成与排名 | 执行文献树搜索（30-50 篇论文）、生成 3 条创意并通过 Elo 锦标赛排名、触发 IDE/IVE 记忆更新 |
| **Engineer Agent (EA)** | 实验执行 | 运行结构化 4 阶段实验流水线、维护代码轨迹日志、触发 ESE 记忆提取 |
| **Evolution Manager Agent (EMA)** | 跨周期知识管理 | 运行 IDE/IVE/ESE 三协议、维护 M_I + M_E 双存储、生成演化报告 |

### 1.2 全局流水线

```
research-ideation
  ↓ (IDE 更新 M_I)
paper-planning
  ↓
experiment-pipeline ──(失败)──→ experiment-craft ──→ IVE 更新 M_I
  ↓ (成功，ESE 更新 M_E)
paper-writing → paper-review → paper-rebuttal → academic-slides
```

`evo-memory` 技能横贯全程，提供记忆读写反馈闭环。

### 1.3 技术栈

- Python 86.6%，HTML 7.5%，TeX 5.9%
- 基础框架：LangChain DeepAgents（层级 Agent 编排）
- 安装方式：`npx skills add EvoScientist/EvoSkills`（适用 Claude Code、Cursor、Gemini CLI 等）；或 `/install-skill EvoScientist/EvoSkills@skills`（EvoScientist 内置命令）
- MCP 扩展：arXiv 检索、Web 搜索、文档访问等

---

## 二、技能数据结构（SKILL.md 精确格式）

### 2.1 Frontmatter YAML 规范

每个技能的 `SKILL.md` 以如下 frontmatter 开头（完整字段集）：

```yaml
name: <skill-name>
description: "<单段自然语言描述，含以下四个语义区：
  (1) 功能摘要；
  (2) should-trigger 触发场景（'Use when: ...'）；
  (3) should-not-trigger 排除场景（'Do NOT use for: ...'）；
  (4) 路由目标（'use <other-skill> instead'）"
allowed-tools: "<空格分隔的工具白名单>"
metadata:
  author: EvoScientist
  version: '<semver>'
  tags: [<tag1>, <tag2>, ...]
```

**关键设计点**：

1. `description` 字段**同时承担路由器职责**——框架通过解析该字段的自然语言来决定技能是否加载，无独立的 `should-trigger` YAML 键。
2. `allowed-tools` 是**运行时工具白名单**，Agent 框架在加载技能时将工具集限制为该列表；`execute` 工具仅出现在需要运行代码的技能中（experiment-pipeline、experiment-craft 等），纯写作技能（paper-writing、paper-review）不含该工具。
3. `version` 使用语义版本，`research-ideation` 已迭代至 `2.1.0`，其余技能多为 `1.0.0` 或 `1.1.0`。

### 2.2 全量 13 个技能清单（含 `allowed-tools`）

| 技能 | 版本 | allowed-tools | tags |
|------|------|---------------|------|
| `research-ideation` | 2.1.0 | write_file edit_file read_file think_tool execute | core, research, ideation, tournament, proposal |
| `paper-planning` | 1.0.0 | write_file edit_file read_file think_tool | core, research, writing |
| `experiment-pipeline` | 1.0.0 | write_file edit_file read_file think_tool execute | core, experimentation, experiment-design |
| `experiment-craft` | 1.0.0 | write_file edit_file read_file think_tool execute | core, experimentation, experiment-design |
| `experiment-iterative-coder` | 1.0.0 | write_file edit_file read_file think_tool execute | core, code-generation, iteration, refinement |
| `paper-writing` | 1.0.0 | write_file edit_file read_file think_tool | core, research, writing, academic-writing, latex |
| `paper-review` | 1.0.0 | read_file edit_file write_file think_tool | core, writing, academic-writing, peer-review |
| `paper-rebuttal` | 1.0.0 | write_file edit_file read_file think_tool | core, writing, academic-writing, peer-review |
| `academic-slides` | 1.0.0 | write_file, edit_file, read_file, think_tool, execute | core, writing, presentation, academic-writing |
| `evo-memory` | 1.0.0 | write_file edit_file read_file think_tool | core, meta-learning |
| `paper-navigator` | 1.1.0 | write_file edit_file read_file think_tool execute | core, research, literature, papers, search, citation |
| `research-survey` | 1.0.0 | write_file edit_file read_file think_tool | core, research, literature, survey, synthesis |
| `nano-banana` | 1.0.0 | write_file, edit_file, read_file, think_tool, execute | core, presentation, image-generation |

### 2.3 技能路由机制（should-trigger 语法）

框架对每个输入请求执行如下路由逻辑：

1. **关键词匹配**：从 `description` 中提取 `Use when:` 后的短语列表（如 `"update memory"`, `"classify failure"`, `"what worked before"`），对用户输入进行关键词匹配。
2. **Embedding 路由**：将用户输入与每个技能的 `description` 全文计算余弦相似度，若最高分超过阈值则触发。
3. **互斥排除**：命中 `Do NOT use for:` 的请求路由到指定替代技能。
4. **多技能触发时**：当前上下文决定主技能；其他技能以"子技能调用"形式嵌套（如 `research-ideation` 内部调用 `paper-navigator`、`evo-memory`）。

**路由示例**（`evo-memory` 的触发短语）：

```
触发：update memory / classify failure / what worked before /
      research history / evolution / 新周期需要先验知识
排除：running experiments → experiment-pipeline
      debugging code → experiment-craft
      generating ideas → research-ideation
```

### 2.4 技能内部结构

每个技能目录结构固定：

```
skills/<skill-name>/
├── SKILL.md          # 技能主文件（frontmatter + 步骤化工作流 + 启发式原则）
├── assets/           # 模板、图片
└── references/       # 加载进 Agent 上下文的辅助文档
```

`evo-memory/references/` 包含五个协议文件：`ide-protocol.md`、`ive-protocol.md`、`ese-protocol.md`、`memory-schema.md`、`paper-prompts.md`——这些文件在 `evo-memory` 被触发时自动注入 Agent 上下文。

---

## 三、进化算法精确描述（IDE / IVE / ESE）

### 3.1 IDE — Idea Direction Evolution（想法方向演化）

**触发条件**：`research-ideation` Step 5（Elo 锦标赛）完成，`/direction-summary.md` 已保存。

**输入变量**：

```
{user_goal}        ← 原始研究目标（非提案文档全文）
{top_ranked_ideas} ← 锦标赛 top-3 结果（含 Elo 分数）
```

**七步算法**：

1. **读取当前 M_I**：从 `/memory/ideation-memory.md` 载入已有可行方向与失败方向列表。
2. **执行 IDE Prompt**：填入 `{user_goal}` 和 `{top_ranked_ideas}`，生成 DIRECTION SUMMARY（2-3 个方向，每个含 Core Idea + Why Promising + Requirements + Validation Plan）。
3. **抽象到可复用层级**：从具体实验步骤抽象为"技术-领域对"，再进一步抽象为宽泛方向。
   - 示例：`"Llama-2 的 4-bit 量化"` → `"大语言模型推理的精度压缩技术"`
   - 原则：*"抽象程度应使未来研究者能从该方向生成具体想法，而不是直接存储锦标赛方案"*
4. **重叠检测**：与 M_I 现有条目比对，分为精确匹配/部分重叠/新方向三类。
5. **耗尽追踪**：若某方向已在 3+ 个周期后无进展，标记为 `approaching exhaustion`，状态更新为 `approaching exhaustion`。
6. **写入标准化条目**（完整字段集见 §4.2）。
7. **生成演化报告**：保存至 `/memory/evolution-reports/cycle_N_ide.md`，说明新增/更新内容及预期影响。

**条目状态枚举**：`feasible` | `approaching exhaustion` | `claimed territory` | `retry with fixes`

---

### 3.2 IVE — Idea Validation Evolution（想法验证演化）

**双重触发条件**（任一满足即触发）：

| 触发类型 | 条件 |
|----------|------|
| **规则触发** | Engineer Agent 在预算内无法生成可执行代码（任意阶段） |
| **LLM 触发** | 实验完成但提出的方法表现劣于基线 |

**主分类步骤**（Step 1）：

执行 IVE Prompt，填入 `{research_proposal}` 和 `{execution_report}`，产生三种结果：

```
FAILED(NoExecutableWithinBudget)  → 实现失败，标记"retry with fixes"
FAILED(WorseThanBaseline)          → 进入 Step 2 诊断
NOT_FAILED(ValidatedOrInconclusive)→ 不更新 M_I
```

**扩展诊断（Step 2）— 五诊断问题**（当结果为 `FAILED(WorseThanBaseline)` 时）：

| 问题 | 信号类型 |
|------|----------|
| Q1：任意变体显示局部改进？ | 是 → 实现失败信号 |
| Q2：简化问题上假设是否成立？ | 否 → 根本失败信号 |
| Q3：类似方法在已发表研究中成功？ | 是 → 实现失败信号 |
| Q4：不同实现中失败模式一致？ | 是 → 根本失败信号 |
| Q5：轨迹日志中能定位具体 bug？ | 是 → 实现失败信号 |

**决策矩阵**：

| 实现失败信号数 | 根本失败信号数 | 分类结果 |
|---------------|---------------|----------|
| 3-5 | 0-2 | **实现失败**（可重试） |
| 0-2 | 3-5 | **根本失败**（不可重试） |
| 分裂 | 分裂 | 默认判定**实现失败**（保守原则） |

**保守原则**："浪费一次重试的代价远小于永久丢弃有效方向"。

**三连败升级规则**：若同一方向被连续分类为"实现失败"达 3 次，触发"仔细重评估"流程（不直接标记根本失败，而是召集更深入的诊断）。

**M_I 更新规则**：

- 实现失败 → 更新 Feasible Directions 区：增加 `Retry Count`（整数）、`Retry Guidance`、`Countermeasures` 字段
- 根本失败 → 写入 Unsuccessful Directions 区：含完整 5 问诊断答案、Root Cause、Boundary Conditions、Do-Not-Repeat Notes

---

### 3.3 ESE — Experiment Strategy Evolution（实验策略演化）

**触发条件**：`experiment-pipeline` 全部 4 个阶段完成且关口条件全部满足。

**输入**：

```
{research_proposal}  ← 研究提案文本
{trajectories}       ← 来自四个阶段的轨迹日志合集
                       /experiments/stage1_baseline/trajectory.md
                       /experiments/stage2_tuning/trajectory.md
                       /experiments/stage3_method/trajectory.md
                       /experiments/stage4_ablation/trajectory.md
                       /experiments/trajectory-summary.md（可选）
```

**七步算法**：

1. **执行 ESE Prompt**：生成 DATA SUMMARY（预处理/增强/分割策略，含精确超参数）和 MODEL SUMMARY（优化器、学习率调度、训练技巧）。
2. **采集轨迹日志**：聚合上述 4-5 个文件。
3. **识别四类模式**：
   - 数据处理策略（Data Processing）
   - 模型训练策略（Model Training）
   - 调试策略（Debugging）
   - 架构策略（Architecture）
4. **通用性评估**：每条策略分类为 `Broadly applicable`（跨领域）/ `Domain-specific`（该类问题）/ `Highly specific`（精确配置）。
5. **比对现有 M_E 条目**：精确匹配 → 更新证据；矛盾条目 → 追加备注（两条均保留）；相关条目 → 添加交叉引用；新策略 → 追加。
6. **写入标准化条目**（完整字段集见 §4.3）。
7. **生成演化报告**：保存至 `/memory/evolution-reports/cycle_N_ese.md`，含提取数量分布、Broadly applicable 策略清单、识别的矛盾点。

**防过拟合规则**：单次观察标记为 `Single observation`；需 2+ 个周期后方可标记为 `Confirmed [N cycles]`；若后续周期出现矛盾，状态降为 `Contradicted`（不删除，两方观察均保留）。

---

## 四、记忆存储结构（M_I + M_E 精确 Schema）

### 4.1 双存储概览

| 存储 | 路径 | 颗粒度 | 读取时机 | 写入时机 |
|------|------|--------|----------|----------|
| **M_I**（想法记忆） | `/memory/ideation-memory.md` | 方向级 | `research-ideation` Step 0（k_I=2） | IDE（Step 6）/ IVE（失败分类后） |
| **M_E**（实验记忆） | `/memory/experiment-memory.md` | 策略级 | `experiment-pipeline` 开始时（k_E=1） | ESE（成功实验后） |

两个文件均为纯 Markdown，无数据库层，人类可直接阅读和编辑。

### 4.2 M_I 条目 Schema

**Feasible Directions 区（可行方向）**：

```markdown
### <Direction Name>

- Summary: <单句摘要>
- Why Promising: <可行性理由>
- Requirements: <前提条件/假设>
- Validation Plan: <2-4 步验证路径>
- Evidence: <周期来源 + 结果数据>
- Status: feasible | approaching exhaustion | claimed territory | retry with fixes
- Retry Count: <整数，仅 retry with fixes 时存在>
- Retry Guidance: <下次重试的具体指导，仅 retry with fixes 时存在>
- Countermeasures: <防止重蹈覆辙的措施，仅 retry with fixes 时存在>
- Related Entries: <交叉引用其他条目名>
- Retrieval Tags: <逗号分隔标签，供 embedding 检索使用>
- Date Added: YYYY-MM-DD
- Last Updated: YYYY-MM-DD
```

**Unsuccessful Directions 区（失败方向，仅根本失败）**：

```markdown
### <Direction Name>

- Summary: <单句摘要>
- Failure Classification: Fundamental
- Evidence: <周期 + 指标数据>
- Diagnostic Answers: <5 问诊断的完整答案>
- Root Cause: <根本原因分析>
- Boundary Conditions: <该方向在哪些条件下仍可能有效（可选）>
- Countermeasures: <避免再次尝试的具体措施>
- Do-Not-Repeat Notes: <给未来研究者的警示>
- Retrieval Tags: <逗号分隔标签>
- Date Added: YYYY-MM-DD
```

### 4.3 M_E 条目 Schema

```markdown
### <Strategy Name>

- Category: Data Processing | Model Training | Debugging | Architecture
- Context: <适用场景（When to use）>
- Strategy: <可操作的具体指导>
- Evidence: <周期/阶段/尝试编号 + 量化结果>
- Generality: Broadly applicable | Domain-specific | Highly specific
- Confidence: Single observation | Confirmed [N cycles] | Contradicted
- Related Entries: <交叉引用>
- Date Added: YYYY-MM-DD
- Last Updated: YYYY-MM-DD
```

**M_E 四类策略分区**（文件内部结构）：

```
## Data Processing Strategies    ← 核心分区
## Model Training Strategies     ← 核心分区
## Architecture Strategies       ← 扩展分区
## Debugging Strategies          ← 扩展分区
```

### 4.4 记忆文件维护规则

**主动剪枝条件**（满足任一即可归档）：
- 条目在 10 个周期内未被引用
- 策略已被严格更优的替代方案取代
- 所属领域发生根本性转变（Domain Drift）

**版本追踪**：每个记忆文件维护 `Last Updated` 字段和周期计数器；条目修改（而非追加）时，在演化报告中记录变更原因。

**跨领域迁移原则**：策略往往可跨领域复用——评估 `Generality` 时优先考虑普适价值（"训练不稳定性的解决方案在多数深度学习任务中适用"）。

---

## 五、检索机制（Embedding-Based Retrieval）

### 5.1 检索参数

| 参数 | M_I 取值 | M_E 取值 |
|------|----------|----------|
| **k（top-k）** | k_I = **2** | k_E = **1** |
| **相似度度量** | 余弦相似度 | 余弦相似度 |
| **检索维度** | Summary + Retrieval Tags | Context + Retrieval Tags |
| **触发时机** | `research-ideation` Step 0 | `experiment-pipeline` 开始时 |

**注**：论文中提及"Embedding-based retrieval with cosine similarity"，但未指定具体向量模型；代码仓库中 `evo-memory/SKILL.md` 注明"embedding-based retrieval or manual relevance assessment"，说明存在人工回退路径。

### 5.2 注入规则

**M_I 注入**（k_I=2 条目进入 `research-ideation`）：

- 可行方向条目 → 作为 Level 1 种子分支注入想法树，Elo 锦标赛的起点得分基于该方向的历史表现
- 失败方向条目（根本失败）→ 标记为"待剪枝"，在想法树生成时主动排除同类分支

**M_E 注入**（k_E=1 条目进入 `experiment-pipeline`）：

- 选出的策略作为所有 4 个阶段的配置提示上下文
- 同时扫描 Debugging Strategies 区，匹配当前实验领域，注入 Stage 1 的调试预备上下文

### 5.3 多技能触发合并规则

当多个技能描述与用户请求均高度相关时：

1. **主技能**：选择 description 相似度最高者（或关键词最精确匹配者）
2. **子技能调用**：主技能在其步骤序列中通过名字显式调用其他技能（如 `research-ideation` Step 2 内调用 `paper-navigator` Workflow 9；Step 6 调用 `evo-memory` IDE 协议）
3. **互斥保护**：`paper-writing` 与 `paper-review` 互斥——`paper-review` 仅在草稿已完成时触发；`paper-rebuttal` 仅在收到外部审稿意见后触发

---

## 六、实验流水线详解（4 阶段 + 嵌套调试）

### 6.1 四阶段结构

| 阶段 | 目标 | 尝试预算 | 关口条件 |
|------|------|----------|----------|
| Stage 1 | 复现基线代码与指标 | ≤ 20 次 | 指标在报告值 2% 范围内（或报告方差内） |
| Stage 2 | 针对当前环境优化超参数 | ≤ 12 次 | 稳定配置，3 次运行方差 < 5% |
| Stage 3 | 实现并验证新方法 | ≤ 12 次 | 在主指标上优于调优后基线 |
| Stage 4 | 消融实验证明各组件贡献 | ≤ 18 次 | 所有声明均有受控实验支撑 |

**通用尝试循环（Universal Stage Loop）**：每次尝试遵循 `Generate → Execute → Record → Diagnose → Revise` 五步，且每次均记录：假设（Hypothesis）、代码变更摘要、量化结果、假设验证分析。

### 6.2 代码轨迹日志格式

```
## Attempt <N> — Stage <M>

### Hypothesis
<预期结果及推理>

### Code Changes
<关键修改摘要（非完整 diff）>

### Results
<量化指标 + 定性观察 + 具体案例>

### Analysis
<假设是否得到验证；未验证的原因；经验教训>
```

该格式直接被 ESE Prompt 消费，`{trajectories}` 变量即为上述日志的拼接。

### 6.3 嵌套调试（experiment-craft 集成）

失败尝试触发 `experiment-craft` 五步诊断，**不占用阶段预算**：

1. **收集失败案例**：确定失败是系统性还是随机性
2. **找到可工作版本**：通过简化或回退建立基线
3. **桥接差距**：每次只加入一个因素，隔离破坏点
4. **假设与验证**：列出解释，按可能性排序，设计靶向实验确认根因
5. **提出并实施修复**：搜索针对已确认根因的技术方案，验证修复有效性

**核心约束**：每次实验**只改变一个变量**。

### 6.4 experiment-iterative-coder 的评分机制

当主 Agent 委派代码生成任务（`MODE: MORE_EFFORT`），启动此技能：

- **复合评分（0-1.0）**：综合 ruff lint、ruff format、pytest 结果、自我评估
- **硬性上限**：lint 失败时得分封顶 0.4；测试失败时得分封顶 0.6
- **阶段推进条件**：得分 ≥ 0.85 或达到迭代上限（单阶段最多 3 次，全局最多 10 次）
- **任务分解策略**：1 文件 = 1 阶段；2-4 文件 = 2 阶段；5+ 文件 = 3-5 阶段

---

## 七、Agent ↔ Skill 边界与提示组装规则

### 7.1 Skill 作为 Prompt Body

EvoSkills 中所有技能以 **Markdown 文件**形式交付，不含 Python 运行时代码。技能本身即提示指令，代码执行委托给宿主 Agent 框架（Claude Code、Cursor 等）。

**提示组装流程**：

```
1. 框架加载 SKILL.md
2. 提取 allowed-tools → 限制当前 Agent 可调用的工具集
3. 将 SKILL.md 步骤内容 + references/ 辅助文档注入系统提示
4. 在注入内容前插入检索到的记忆条目（M_I 或 M_E 的 top-k）
5. 用户输入作为用户消息附加
```

### 7.2 各 Agent 的职责边界

**Researcher Agent (RA)**：
- 接受 `research-ideation`、`paper-planning`、`paper-navigator`、`research-survey` 技能
- 禁止直接运行代码（`execute` 工具不在其白名单）
- 文献发现**强制**通过 `paper-navigator`（禁止直接 WebSearch/WebFetch）

**Engineer Agent (EA)**：
- 接受 `experiment-pipeline`、`experiment-craft`、`experiment-iterative-coder` 技能
- 拥有 `execute` 工具权限，可运行 Python 代码及 shell 命令
- 每次代码变更须维护轨迹日志（为 ESE 提供结构化消费数据）

**Evolution Manager Agent (EMA)**：
- 接受 `evo-memory` 技能（及其 references/ 下的五个协议文件）
- 无 `execute` 权限（仅文件读写）
- 负责驱动 IDE/IVE/ESE 三协议，生成演化报告

**写作技能（paper-writing / paper-review / paper-rebuttal / academic-slides）**：
- 无 `execute` 权限
- 可以被 RA 或独立 Agent 承接
- `paper-review` 仅在 `paper-writing` 完成后触发；`paper-rebuttal` 仅在外部审稿意见到达后触发

### 7.3 反直觉启发式（内置于 SKILL.md Prompt 体）

各技能 prompt body 中以"原则"形式硬编码了违反直觉的策略：

| 技能 | 代表性原则 |
|------|-----------|
| `paper-writing` | "先写拒信"（在写优点前先模拟一段批评性拒稿意见） |
| `experiment-pipeline` | "固定预算强制系统性思考——第 47 次尝试很少比第 12 次提供更多信息" |
| `evo-memory` | "失败方向比成功方向更有价值——知道什么不该尝试节省的时间更多" |
| `experiment-craft` | "每次只改一个变量" |
| `paper-review` | "拒稿优先模拟——在记录优点之前先写一段批评性拒绝意见" |
| `nano-banana` | "不要重新生成；仅编辑局部更改以保持视觉一致性" |

---

## 八、工程实现细节

### 8.1 存储介质

| 存储对象 | 介质 | 路径 |
|----------|------|------|
| M_I 记忆文件 | 纯 Markdown 文件 | `/memory/ideation-memory.md` |
| M_E 记忆文件 | 纯 Markdown 文件 | `/memory/experiment-memory.md` |
| 演化报告 | 纯 Markdown 文件 | `/memory/evolution-reports/cycle_N_{type}.md` |
| 方向汇总 | 纯 Markdown 文件 | `/direction-summary.md` |
| 实验产物 | 目录结构 | `/experiments/stage{N}_{name}/` |
| 代码轨迹日志 | 纯 Markdown 文件 | `/experiments/stage{N}_{name}/trajectory.md` |
| 迭代日志（iterative-coder）| 纯 Markdown 文件 | `/artifacts/iteration_log.md` |

**无数据库**：所有持久化均为 Markdown 文件，无 SQLite / Vector DB / Redis 依赖。Agent 框架直接通过 `write_file`/`read_file` 工具访问。

### 8.2 演化任务调度

EvoSkills 不依赖独立调度器，演化触发完全**事件驱动**（由主流水线关键节点触发）：

```
research-ideation Step 5 完成 → 触发 IDE
experiment-pipeline 任意阶段预算耗尽（无可执行代码）→ 触发 IVE
experiment-pipeline Stage 3 劣于基线 → 触发 IVE
experiment-pipeline 全部 4 阶段成功 → 触发 ESE
```

无定时任务、无后台线程——所有演化操作在主 Agent 线程内同步执行。

### 8.3 MCP 服务器集成

`mcp/` 目录提供外部能力扩展，按交互方式分两类：

| 类型 | 描述 |
|------|------|
| **Interactive Browser** | 通过浏览器 UI 发现并安装 MCP 服务器（可视化目录） |
| **By Name** | 直接通过服务器名称安装（arXiv、Web Search、文档访问等） |

`paper-navigator` 的文献发现命令（`scholar_search`、`recommend`、`arxiv_monitor` 等）即通过 MCP 服务器提供，不直接调用 WebSearch/WebFetch。

### 8.4 多平台安装

`skills.sh` 脚本（通过 `npx skills add` 调用）支持：

```
Claude Code  → ~/.claude/skills/
Cursor       → .cursor/skills/
OpenCode     → .opencode/skills/
Gemini CLI   → .gemini/skills/
```

安装后，各平台的 Agent 框架自动发现并按需加载 `SKILL.md`。

### 8.5 EvoScientist 主框架架构补充

（来自 EvoScientist 主仓库，为理解 EvoSkills 运行环境提供背景）

- **六个专用子 Agent**：规划、研究、代码生成、调试、分析、技术写作
- **动态系统提示**：基于对话状态的上下文感知提示重写
- **自适应工具选择**：每个 turn 按需过滤工具集，减少认知噪声
- **会话持久化**：`~/.evoscientist/` 存储会话状态，支持跨会话恢复

---

## 九、评估与基准

### 9.1 核心评估维度

`research-ideation` Elo 锦标赛对每个想法在四个维度打分：

| 维度 | 含义 |
|------|------|
| **Novelty（新颖性）** | 区别于已有工作的程度 |
| **Feasibility（可行性）** | 在预算内实现的可能性（非谈判性约束，不可行直接淘汰） |
| **Relevance（相关性）** | 与用户目标的对齐程度 |
| **Clarity（清晰度）** | 方案表达的明确程度 |

Elo 更新公式标准：初始分 1500，K 因子 = 32，两两对比。

### 9.2 系统级基准成绩

| 基准 | 排名 | 时间 |
|------|------|------|
| DeepResearch Bench II | #1 | 2026 年 4 月 |
| AstaBench Code & Execution | #1 | 2026 年 3 月 |
| AstaBench Data Analysis | #1 | 2026 年 3 月 |
| ICAIS 2025 | 最佳论文 + AI 评审奖 | 2025 年 |

### 9.3 记忆有效性指标（论文消融）

论文中的消融研究显示：
- 加入 M_I（IDE + IVE）后，自动评估中的新颖性、可行性、相关性、清晰度均显著提升（超过 7 个对比系统基线）
- 加入 M_E（ESE）后，代码执行成功率实质性提升（论文通过多周期对比验证）

---

## 十、Borrow Recommendations（全量建议，Phase 1 + Phase 2 综合）

### 直接可迁移（结构级）

1. **SKILL.md 的 `allowed-tools` 声明**：每个技能显式白名单工具，可直接用于 cryptotrader-ai 的技能粒度权限控制——不同信号技能开放不同 API 调用权限。

2. **description 字段双重用途设计**（功能摘要 + 路由器）：无需独立路由配置文件，将触发条件和排除条件内嵌于描述字段，降低维护成本。

3. **代码轨迹日志格式**（尝试编号 + 假设 + 变更 + 指标 + 分析）：与 cryptotrader-ai 的 `ExperienceRule` 高度同构，可作为 `ExperienceRule.conditions` + `ExperienceRule.evidence` 的填充模板。

4. **双存储分离设计**（M_I 追踪"方向级"，M_E 追踪"策略级"）：对应 cryptotrader-ai 可拆分为"市场信号方向记忆"（类 M_I）和"交易策略执行记忆"（类 M_E），与现有 `ExperienceMemory.success_patterns` / `forbidden_zones` 分类逻辑吻合。

### 算法级借鉴

5. **IVE 的失败分类（实现失败 vs 根本失败）+ 三连败升级规则**：cryptotrader-ai 的等价物可将交易亏损归因为"执行缺陷"（滑点、延迟、信号延迟）vs"方向错误"（信号逻辑本身失效）；三连败升级规则防止错误放弃有效信号。

6. **五诊断问题的结构化决策矩阵**：可移植为 cryptotrader-ai `IVE` 分支的归因 prompt 模板，输出明确的 `classification: implementation | direction` 标签，直接更新 `ExperienceRule.source`。

7. **IDE 的抽象层级规范**（"技术-领域对"→"宽泛方向"）：cryptotrader-ai 可借鉴用于将具体交易策略（"BTC 5 分钟图 RSI<30 买入"）抽象为可复用方向（"超卖反转策略，适用于高波动资产"），提升 `ExperienceRule` 跨市场泛化能力。

8. **ESE 的防过拟合机制**（Single observation → Confirmed [N cycles] → Contradicted 三态）：直接对应 cryptotrader-ai `ExperienceRule.maturity` 字段的设计逻辑，可引入相同三态状态机。

### 工程级借鉴

9. **top-k 检索超参数**（k_I=2，k_E=1）：cryptotrader-ai 现有 `search_by_regime()` 已有类似设计，可进一步量化对齐 k=2（方向级）/ k=1（策略级）的超参数设计，防止注入过多历史噪声。

10. **事件驱动演化调度**（无后台线程，在主流水线节点同步触发）：简化 cryptotrader-ai 演化任务的架构复杂度——不需要 APScheduler，可在 `nodes/verdict.py` / `nodes/journal.py` 的成功/失败路径末尾同步触发演化更新。

11. **演化报告格式**（What changed + Why + Expected future impact）：改善 cryptotrader-ai 当前 `reflect.py` 的可审计性，便于人工回溯演化历史。

12. **`should-trigger` / `should-not-trigger` 测试集的可测试性设计**：为 spec 016/018 的技能加载条件编写单元测试提供参照——每个技能配一组 positive/negative 示例作为路由测试用例。

---

## 十一、Notes / Open Questions

- **存储介质差异**：EvoSkills 所有记忆为纯 Markdown 文件（无 DB），cryptotrader-ai 已有 SQLite 支持，集成层更重但能力更强；借鉴时可将 Schema（§4.2/4.3）直接映射为 SQLite 表结构。
- **embedding 实现未公开**：论文提及"embedding-based cosine similarity"但未指定向量模型；代码仓库中存在人工评估回退路径，说明框架对 embedding 模型无强依赖。
- **提示驱动 vs 代码驱动**：EvoSkills 的 IDE/IVE/ESE 均以提示词触发（Agent 读取协议文件后生成文本），非 Python 函数；cryptotrader-ai 的 `reflect.py` 是程序化触发——两者目标语义可类比，但实现路径不同，借鉴时需做适配。
- **13 个技能的科研专用性**：所有技能均面向学术科研（arXiv、LaTeX、Elo 排名），与量化交易场景存在领域隔离；借鉴时需将"研究方向"映射为"交易信号方向"，"实验策略"映射为"执行配置策略"。
- **nano-banana 的 Gemini 依赖**：该技能强依赖 Google Gemini API，在 cryptotrader-ai 场景中无对应需求。
- **Skills.sh 安装机制**：通过 `npx skills add` 支持 Claude Code、Cursor、OpenCode、Gemini CLI 等多平台；spec 018 若采用类似安装机制需确认目标平台和权限模型。
