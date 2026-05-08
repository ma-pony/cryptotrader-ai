---
name: EvoSkills
url: https://github.com/EvoScientist/EvoSkills
license: Apache-2.0
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# EvoSkills — EvoScientist

## Architecture Overview

EvoSkills 是 EvoScientist 框架的官方技能仓库，面向端到端**科研自动化**场景。核心思路：将科研流程拆分为 13 个自包含技能模块（每个模块一个 `SKILL.md`），由三类 Agent 协作驱动：

- **Researcher Agent**：执行想法树搜索与 Elo 锦标赛排名
- **Engineer Agent**：运行结构化 4 阶段实验流水线
- **Evolution Manager**：跨研究周期维护持久化记忆机制

整体流水线：`research-ideation → experiment-pipeline / experiment-craft → paper-writing / paper-review / paper-rebuttal → academic-slides`，`evo-memory` 技能横贯全程提供反馈闭环。

技术栈：Python 86.6%，HTML 7.5%，TeX 5.9%。外部能力通过 MCP 服务器扩展（arXiv 检索、Web 搜索、文档访问等）。

## Prompt Assembly（Phase 1）

### 技能格式：SKILL.md

每个技能以单一 `SKILL.md` 文件交付，包含：

1. **Frontmatter**（YAML）：`name`、`author`、`version`、`tags`、`allowed_tools` — 后者显式声明该技能允许调用的工具（如 `write_file, edit_file, read_file, think_tool, execute`）。
2. **描述字段**：用于运行时路由 — 含 `should-trigger` 与 `should-not-trigger` 测试集，Agent 框架据此决定何时加载技能。
3. **步骤化工作流**：以有编号的步骤序列形式嵌入提示指令，例如 `paper-writing` 为 11 步、`experiment-pipeline` 为 4 阶段。
4. **反直觉启发式**：以"原则"形式写入提示体，例如"先写拒信"（`paper-writing`）、"每次只改一个变量"（`experiment-craft`）。
5. **`references/` 子目录**：存放加载进 Agent 上下文的辅助文档；`assets/` 存放模板与图片。

### 提示注入触发点

- **周期开始时**（`research-ideation` 启动）：从 `M_I`（想法记忆）检索 top-k_I=2 可行方向，注入当前提示，同时标记需剪枝的"基本失败"方向。
- **实验阶段开始时**（`experiment-pipeline` Stage 2-3）：从 `M_E`（实验记忆）检索 top-k_E=1 最相关策略，注入配置提示。
- **失败时**：触发 `experiment-craft` 的 5 步诊断流程（观察 → 假设 → 测试 → 结论 → 处方），作为子提示链插入主流。

### 结构化输出格式

- 每阶段产物保存至 `/experiments/stageN_name/`（结果、配置、代码）
- **代码轨迹日志**：记录尝试编号、假设、代码变更、指标、分析 — 形成可被 `evo-memory` 消费的结构化文本
- **流水线追踪器**：跨四阶段监控进展及关口评估结果

## Memory ↔ Skill 连接（Phase 1 lite）

### 双存储结构

| 存储 | 路径 | 内容 |
|------|------|------|
| Ideation Memory (M_I) | `/memory/ideation-memory.md` | 可行方向 + 失败分类（实现失败 vs 根本失败） |
| Experimentation Memory (M_E) | `/memory/experiment-memory.md` | 数据处理策略、模型训练策略、架构策略、调试策略 |

每条记忆条目包含：唯一标识符、时间戳、周期来源、验证证据、分类字段、通用性评估（领域特定 vs 广泛适用）。

### 三种演化机制

**IDE（Idea Direction Evolution）**
- 触发条件：`research-ideation` 完成 Step 5 后
- 操作：用 IDE 提示词从具体想法中抽象出可复用的方向级别表示，更新 M_I

**IVE（Idea Validation Evolution）**
- 触发条件：实验失败（无可执行代码或性能不达标）
- 操作：5 问诊断将失败分类为"实现失败"或"根本失败"；同一方向 3 次以上实现失败则升级重评估；更新 M_I

**ESE（Experiment Strategy Evolution）**
- 触发条件：`experiment-pipeline` 成功完成后
- 操作：从轨迹日志中提取数据处理和模型训练策略，评估通用性，更新 M_E

### 检索机制

采用**基于 Embedding 的余弦相似度**（或人工相关性评估）：将当前上下文与存储条目的 Summary + Retrieval Tags 对比。

记忆维护原则：
- 主动剪枝陈旧条目（10 个周期以上、被取代、领域漂移）
- 跨领域迁移：策略往往可跨领域复用
- 演化报告须解释"为什么"，便于数月后人工回溯

## Phase 2 Placeholders

（空列表 — Phase 2 待执行）

## Borrow Recommendations（Phase 1 only）

1. **SKILL.md 的 `allowed_tools` 声明**：每个技能显式白名单工具，可直接借鉴用于 cryptotrader-ai 的技能粒度权限控制。

2. **代码轨迹日志格式**（尝试编号 + 假设 + 变更 + 指标 + 分析）：与 cryptotrader-ai 的 journal/experience 记忆高度同构，可作为 `ExperienceRule` 条目的填充模板。

3. **双存储分离设计**（M_I 追踪"方向级"知识，M_E 追踪"策略级"知识）：对应 cryptotrader-ai 可拆分为"市场信号方向记忆"和"交易策略执行记忆"，与现有 `ExperienceMemory.success_patterns` / `forbidden_zones` 分类逻辑吻合。

4. **IVE 的失败分类（实现失败 vs 根本失败）**：cryptotrader-ai 的 `IVE` 等价物可将交易亏损归因为"执行缺陷"（滑点、延迟）vs "方向错误"（信号逻辑本身），防止错误丢弃有效方向。

5. **top-k 检索 + 主动剪枝**：现有 `search_by_regime()` + `_filter_records_by_regime` 已有类似设计，可进一步对齐 EvoSkills 的 k_I=2 / k_E=1 超参数设计思路，并引入生命周期剪枝。

6. **`should-trigger` / `should-not-trigger` 测试集**：技能路由的可测试性设计，可为 spec 016 的技能加载条件编写单元测试提供参照。

## Notes / Open Questions

- EvoSkills 所有技能以 Markdown 文件交付（无 Python 运行时代码），技能本身即提示；代码执行委托给 Agent 框架（Claude Code、Cursor 等）。这与 cryptotrader-ai 以 Python 节点为中心的执行模型存在本质差异，借鉴时需做适配。
- 记忆文件为纯 Markdown（`/memory/*.md`），无数据库持久化；cryptotrader-ai 已有 SQLite 支持，集成层更重但能力更强。
- 检索机制在文档中描述为"embedding 余弦相似度或人工评估"，未见具体向量库实现；Phase 2 需确认实际代码路径。
- `evo-memory` 的演化机制（IDE/IVE/ESE）均以提示词触发，非代码触发 — 与 cryptotrader-ai 的 `reflect.py` 程序化触发路径不同，但目标语义可类比。
- 13 个技能的安装通过 `skills.sh` 脚本完成，支持 Claude Code、Cursor、OpenCode、Gemini CLI 等多平台；spec 016 若采用类似机制需确认目标平台。
