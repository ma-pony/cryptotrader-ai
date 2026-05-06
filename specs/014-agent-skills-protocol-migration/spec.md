# Feature Specification: Agent Skills 协议迁移（双层架构）

**Feature Branch**: `014-agent-skills-protocol-migration`
**Created**: 2026-05-06
**Status**: Draft (v2 — 双层架构修订)
**Input**: User description: "重构 trading agent 的结构化经验记忆系统：分两层——`agent_memory/` 保留全部历史记录用于分析（gitignored），`agent_skills/` 是从 memory 整理出的少量高层能力包（git 跟踪、Anthropic Skills 协议格式）"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 双层架构：Memory 与 Skills 解耦（Priority: P1）

作为系统设计者，我希望经验数据明确分为两层：底层 `agent_memory/`（gitignored，永久保留所有 cases / patterns，用于离线分析与反思）+ 高层 `agent_skills/`（git 跟踪，仅 5 个 SKILL.md，是从 memory 整理出的能力描述）。Agent 只读 skills，reflection 写 memory + 整理 skills。

**Why this priority**: 这是整个 feature 的根本架构。混淆 memory 与 skills（之前每条 pattern 当 skill）导致 240+ 文件、git noisy、违反 Anthropic Skills 协议。必须分层。

**Independent Test**:
1. clone 仓库 → `agent_skills/` 5 个 SKILL.md 进 git 可见
2. `agent_memory/` 不在 git 中（`.gitignore` 包含）
3. trading cycle 跑完写入 `agent_memory/<agent>/cases/` 但不污染 git
4. reflection 写入 `agent_memory/<agent>/patterns/` 也不污染 git
5. 运维者手工或触发 LLM 整理 SKILL.md → 该单文件 git diff 可见

**Acceptance Scenarios**:

1. **Given** 空仓库 + 干净 PostgreSQL，**When** 运行 1 个 trading cycle，**Then** `agent_memory/<agent>/cases/` 出现该 cycle 的原始记录文件；`git status` 不显示该文件
2. **Given** 累积 50 个 cycles 后跑 reflection，**When** 反思完成，**Then** `agent_memory/<agent>/patterns/` 出现蒸馏后的 pattern 文件，`agent_skills/` 下 SKILL.md **不变**（整理是独立流程）
3. **Given** 运维者手工编辑 `agent_skills/tech-analysis/SKILL.md`，**When** commit + push，**Then** git history 仅多 1 个文件改动，与 240+ pattern 文件方案对比 noise 大幅降低
4. **Given** 跨机器同步（A 机器训练，B 机器复盘），**When** B 机器 git pull，**Then** B 拿到最新 SKILL.md（足够 agent 决策）；如需深度分析 patterns/cases，B 通过其他渠道（rsync / S3）获取 `agent_memory/`

---

### User Story 2 — Memory 层：原始 cases + 蒸馏 patterns（Priority: P1）

作为反思系统的设计者，每个 trading cycle 完成后系统自动写入 `agent_memory/<agent>/cases/<cycle_id>.md`（原始执行记录，完整快照）；reflection job 周期性读 cases 蒸馏出 patterns 并写入 `agent_memory/<agent>/patterns/<pattern_name>.md`（带 PnL 追踪 + regime tags + maturity）。memory 层数据**永久保留**，不自动清理（用户可定期归档）。

**Why this priority**: memory 是后续所有分析的数据源——反思、调参、复盘、A/B 都从这里读。数据丢失或被自动清理会损害分析能力。

**Independent Test**:
1. 跑 100 个 cycles → `agent_memory/<agent>/cases/` 出现 100 个 .md 文件
2. 跑 reflection → `agent_memory/<agent>/patterns/` 出现新 pattern 或更新
3. `cat agent_memory/tech/cases/2026-05-06-cycle-7ffc.md` 显示完整 cycle 记录（snapshot、verdict、PnL、reasoning 等）
4. `cat agent_memory/tech/patterns/funding_squeeze_long.md` 显示 pattern 元数据 + body

**Acceptance Scenarios**:

1. **Given** trading cycle 完成（含 verdict + 最终平仓 PnL），**When** journal 节点写完，**Then** `agent_memory/<agent>/cases/<cycle_id>.md` 内容完整：snapshot_summary + agent analyses + verdict + risk_gate + execution_status + final_pnl + applied_patterns
2. **Given** 累计 50 个 cases，**When** reflection 触发，**Then** 输出 0~N 个新 pattern 文件 + 0~M 个旧 pattern 的 PnL 更新；4 层防过拟合（regime / sample / global-vs-regional / forbidden adversarial）全保留
3. **Given** 已 deprecated 的 pattern 文件，**When** 60 天后用户想复盘，**Then** 该文件**仍在 archive/ 目录中可读**（不被自动删除）
4. **Given** 用户手工想分析某 case，**When** `cat agent_memory/tech/cases/2026-05-06-cycle-7ffc.md`，**Then** 看到完整结构化数据（YAML frontmatter + markdown body）

---

### User Story 3 — Skills 层：5 个高层能力包（Anthropic 协议）（Priority: P1）

作为 trading agent，我读取的是 5 个**整理后**的 SKILL.md（4 个 agent + 1 个 trading-knowledge），每个文件遵循 Anthropic Skills 协议（frontmatter `name` + `description` + 完整 body 含 active patterns 摘要）。SKILL.md **不是每 cycle 自动更新**，而是周期性（每周或按触发）由人工 / LLM 从 memory/patterns 整理而来。

**Why this priority**: agent 看到的"经验"必须是稳定、整理后的能力包——而非每 cycle 都变化的原始记录。SKILL.md 数量小（5 个）符合 Anthropic Skills 协议本意，与本仓库 `.claude/skills/` 既有实践对齐。

**Independent Test**:
1. `ls agent_skills/` 列出恰好 5 个目录：tech-analysis / chain-analysis / news-analysis / macro-analysis / trading-knowledge
2. 每个目录有且仅有 `SKILL.md`（initial 阶段；将来可加 reference 子文件）
3. 每个 SKILL.md frontmatter 含 `name` 与 `description`（合规 Anthropic 协议）
4. middleware 把 5 个 SKILL.md 的内容注入对应 agent prompt
5. 运维者手工修改任一 SKILL.md → 下一个 cycle agent 看到改动（无需 reflection 触发）

**Acceptance Scenarios**:

1. **Given** 仓库初始化，**When** `ls agent_skills/`，**Then** 显示 5 个目录且总文件数 ≤ 6（5 个 SKILL.md + 可选 .gitkeep）
2. **Given** tech-analysis/SKILL.md 含 12 个 active pattern 摘要 + 4 个 forbidden 摘要，**When** TechAgent 节点运行，**Then** middleware 把整个 SKILL.md body 注入 system prompt
3. **Given** 用户手工编辑 tech-analysis/SKILL.md 加了一句新约束，**When** 下一个 cycle 运行，**Then** TechAgent system prompt 立即反映新内容（无需重启、无 reflection 介入）
4. **Given** SKILL.md frontmatter 有 `manually_edited: true`，**When** 自动整理流程运行（LLM 周期重写），**Then** 该文件被跳过，仅追加新数据到 `<!-- AUTO-DISTILLED-PATTERNS -->` 标记内的区段（如有）

---

### User Story 4 — Middleware 自动注入 + load_skill tool（Priority: P1）

每个 trading agent 节点（tech / chain / news / macro）通过 LangChain `SkillsInjectionMiddleware` 自动加载对应 SKILL.md + trading-knowledge 共享 SKILL.md，注入 system prompt。同时 middleware 注册 `load_skill(name)` tool，agent 可按需重新拉取 skill body（用于 multi-turn 对话场景；本期单 turn 主要用静态注入）。

**Why this priority**: middleware 是把 SKILL.md 文件接到 LangChain 框架的桥梁——没它则 SKILL.md 只是好看不顶用。

**Independent Test**:
1. mock 5 个 SKILL.md，构造一个 trading cycle，断言 4 个 agent 节点 LLM 收到的 system_prompt 含对应 SKILL.md body 与 trading-knowledge body
2. 调用 `load_skill("tech-analysis")` 返回该文件 body
3. 调用 `load_skill("nonexistent")` 返回结构化 error

**Acceptance Scenarios**:

1. **Given** tech-analysis/SKILL.md 与 trading-knowledge/SKILL.md 存在，**When** TechAgent 节点运行，**Then** system prompt 含 2 个 SKILL.md 的 body 内容
2. **Given** verdict 节点（不走 create_agent）需要 skill 内容，**When** 调用 `load_skill("trading-knowledge")` Python 函数，**Then** 返回 SKILL.md body 字符串
3. **Given** agent 调用 `load_skill("nonexistent")`，**When** tool 解析，**Then** 返回 `{"error": "skill_not_found", "name": "nonexistent"}`
4. **Given** middleware 启动，**When** SKILL.md 文件读取成功，**Then** 同一 cycle 内重复读取走进程内缓存（无重复磁盘 IO）

---

### User Story 5 — Verdict 引用 pattern 名称便于归因（Priority: P2）

verdict 节点输出的 reasoning 中显式声明 `applied: <pattern_name>`（pattern 名出现在 SKILL.md 文本里）。reflection job 解析 reasoning 后定位到 `agent_memory/<agent>/patterns/<pattern_name>.md` 更新 PnL track。

**Why this priority**: 实现精确 PnL 归因，让 4 层防过拟合算法基于真信号工作。

**Independent Test**: 跑 50 个 cycles，统计 verdict reasoning 中 `applied:` 频率（≥ 60% 在有 active pattern 时）+ 引用的 pattern 名都能在 memory 中找到对应文件。

**Acceptance Scenarios**:

1. **Given** SKILL.md body 中列出 `funding_squeeze_long`，verdict 引用了它，**When** reasoning 文本生成，**Then** 含 `applied: funding_squeeze_long`
2. **Given** verdict 引用了不存在的 `applied: nonexistent_pattern`，**When** reflection 解析，**Then** 该引用被忽略 + logger.warning（视作 hallucination）
3. **Given** 同一 cycle 引用 3 条 patterns，**When** 平仓 PnL = +120，**Then** 3 条 pattern 文件的 `pnl_track.cases` 各 +1，`avg_pnl` 各按权平均更新

---

### User Story 6 — 旧系统全删（Priority: P2）

迁移完成后旧 GSSC pipeline、`ExperienceMemory` / `ExperienceRule` dataclass、`success_patterns / forbidden_zones / strategic_insights` 三分法、`decision_commits.experience_json` DB 列、`arena experience` CLI 子命令、4 个 GSSC 测试文件**完全删除**——不留 fallback。

**Why this priority**: clean break 让代码库可维护性提升；保留双轨双重维护成本。

**Independent Test**: `grep -rn "ExperienceMemory\|ExperienceRule\|success_patterns\|forbidden_zones\|strategic_insights\|gather_packets\|select_packets\|structure_experience" src/ tests/` 返回 0 结果；空 PostgreSQL 启动 cycle 跑通。

**Acceptance Scenarios**:

1. **Given** 全套测试运行，**When** 上述 grep 执行，**Then** 0 命中
2. **Given** 干净 PostgreSQL 启动，**When** 跑首个 cycle，**Then** 不需要任何 experience 种子数据
3. **Given** alembic-style migration 运行，**When** 检查 `decision_commits` 表结构，**Then** `experience_json` 列已 drop

---

### Edge Cases

- **`agent_memory/` 不存在（首次启动）**：系统 MUST 自动创建目录树（含 4 个 agent 子目录 + cases/ + patterns/ + archive/），不报错
- **`agent_skills/` 某 SKILL.md 缺失**：middleware 跳过该 skill 注入 + logger.warning，其他 skill 正常注入；agent 仍可工作
- **SKILL.md frontmatter 损坏**：跳过该文件 + warning，加载其他 skills
- **memory 数据量爆炸**（cases/ 文件 > 10K）：不自动清理；用户可手工归档（如压缩到 `agent_memory/<agent>/archive/2026-Q1.tar.gz`），或后续 follow-up 加 retention 策略
- **`load_skill` 调用频率失控**：同一 cycle 调用 > 10 次时返回 `rate_limit_per_cycle` error
- **reflection 失败**：不阻塞下一个 cycle；下一个 cycle 仍读最新版 memory（reflection 失败前）
- **pattern 名冲突（同一 agent 同名）**：reflection 写入前检测，若旧 pattern frontmatter `manually_edited: true` 仅更新 `pnl_track`，否则全量 rewrite
- **跨 agent 同名 pattern**：天然路径隔离（`tech/patterns/X.md` ≠ `chain/patterns/X.md`）；引用时 reasoning 中以 agent 上下文消歧（verdict 节点显式用 agent 前缀如 `applied: tech::X`）
- **manually_edited 设置方式**：用户手工编辑 SKILL.md 或 pattern 文件后**自行**在 frontmatter 加 `manually_edited: true`，本期不实现自动检测

## Requirements *(mandatory)*

### Functional Requirements

#### 双层架构

- **FR-001**: 系统 MUST 在仓库根创建两个顶级目录：`agent_memory/`（**gitignored**）与 `agent_skills/`（**git 跟踪**）
- **FR-002**: `.gitignore` MUST 包含 `agent_memory/`，确保该目录及子内容不进 git
- **FR-003**: `agent_memory/` 下 MUST 有 4 个 agent 子目录（tech / chain / news / macro），每个子目录含 `cases/`、`patterns/`、`archive/` 三个二级目录
- **FR-004**: `agent_skills/` 下 MUST 有恰好 5 个 skill 目录（tech-analysis / chain-analysis / news-analysis / macro-analysis / trading-knowledge），每个目录至少含 1 个 `SKILL.md` 文件
- **FR-005**: `agent_memory/` 数据 MUST 永久保留（不自动删除）；deprecated 的 pattern 文件移到 `agent_memory/<agent>/archive/`，仍可读取

#### Memory 层（cases + patterns）

- **FR-006**: 每个 trading cycle 完成时（含 journal_trade 节点）MUST 写入 `agent_memory/<agent>/cases/<cycle_id>.md`（每个 agent 独立一份）；文件含 frontmatter（cycle_id / timestamp / pair / verdict_action / final_pnl）+ markdown body（agent_analysis + verdict_reasoning + applied_patterns）
- **FR-007**: cycle 写 cases 失败 MUST NOT 阻塞 cycle 主流程，logger.warning 记录后继续
- **FR-008**: reflection job MUST 周期性（按 `[experience] every_n_cycles`）读 `agent_memory/<agent>/cases/` 蒸馏出 patterns，写入 `agent_memory/<agent>/patterns/<pattern_name>.md`
- **FR-009**: pattern 文件 MUST 有 frontmatter（name / description / regime_tags / pnl_track / maturity / created / source_cycles / version）+ body（条件 / 案例 / 例外）
- **FR-010**: reflection 算法 MUST 完整保留 4 层防过拟合（pure-PnL 驱动）：
  - **L1 — regime-aware 胜率统计**：仅同 regime 历史样本
  - **L2 — 最少样本量门槛**：< N（默认 5）case 不晋升 maturity
  - **L3 — 全局 vs 区段差距**：区段必须显著优于全局基线
  - **L4 — 对手验证（forbidden 专属）**：相反方向有亏损证据
  - 不引入时间衰减层
- **FR-011**: skill maturity MUST 演化（observed → probationary → active → deprecated），由 PnL track 触发；deprecated 的 pattern 文件 MUST 移到 `archive/`，不删除
- **FR-012**: reflection 失败 MUST NOT 阻塞下一个 trading cycle
- **FR-013**: reflection 写文件 MUST 使用原子写（temp + rename）+ `fcntl.flock` 排他锁；archive 操作用 atomic rename

#### Skills 层（5 个 SKILL.md）

- **FR-014**: 每个 SKILL.md MUST 遵循 Anthropic Skills 协议格式：frontmatter 含 `name`（与目录名一致）+ `description`（一句话能力摘要）+ body（agent 角色 + active patterns 摘要 + forbidden zones 摘要 + 使用规则）
- **FR-015**: SKILL.md **不在每 cycle 自动更新**——是周期性整理产物（手工编辑、LLM 触发整理、或半自动 PR）
- **FR-016**: 系统 MUST 提供 CLI 命令 `arena skills curate <skill_name> [--llm]` 触发整理：读 `agent_memory/<agent>/patterns/` active 状态的 patterns + 当前 SKILL.md → 输出新 SKILL.md 草稿
- **FR-017**: SKILL.md frontmatter 含 `manually_edited: true` 时，整理流程 MUST 跳过该文件或仅追加到指定区段（如 `<!-- AUTO-DISTILLED-PATTERNS -->` 标记内）

#### 加载与注入

- **FR-018**: 系统 MUST 提供 `SkillsInjectionMiddleware`（继承 LangChain `AgentMiddleware`），可挂载到 `create_agent` 的 `middleware=` 参数；参考 https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant
- **FR-019**: 4 个 agent 节点（tech_analyze / chain_analyze / news_analyze / macro_analyze）MUST 通过 middleware 自动加载并注入对应 SKILL.md + trading-knowledge SKILL.md
- **FR-020**: middleware MUST 在 `wrap_model_call` 时把 SKILL.md body 拼接到 `request.system_message.content_blocks`
- **FR-021**: middleware MUST 注册 `load_skill(name: str) -> str` tool（通过 `AgentMiddleware.tools` 类变量）；agent 可按需重新拉取任一 skill body
- **FR-022**: `load_skill` MUST 是 LangChain tool + 普通 Python 函数双接口（同实现），便于 verdict 节点等不走 create_agent 的代码直接调
- **FR-023**: `load_skill(name)` 仅需 1 参数（5 个 skill name 之一）；不需要 `agent::pattern` 嵌套形式（patterns 在 SKILL.md body 里）
- **FR-024**: middleware 加载 SKILL.md 失败（文件不存在 / frontmatter 损坏）MUST 跳过 + warning，不阻塞 cycle
- **FR-025**: `load_skill` rate-limit：同一 cycle 同一 trace_id 调用 > 10 次返回 `rate_limit_per_cycle` error

#### Verdict 显式归因

- **FR-026**: verdict 节点 prompt MUST 强制要求 LLM 在 reasoning 中以 `applied: <pattern_name>` 或 `applied: <agent>::<pattern_name>` 形式显式声明应用的 pattern
- **FR-027**: reflection job 解析 verdict.reasoning 时 MUST 提取所有 `applied:` 引用，定位到 `agent_memory/<agent>/patterns/<pattern_name>.md` 并更新 `pnl_track`
- **FR-028**: 引用了不存在的 pattern 时，reflection MUST 跳过 + logger.warning，不影响其他引用归因

#### 删除清单

- **FR-029**: 系统 MUST 删除 `src/cryptotrader/learning/context.py`（GSSC pipeline）整个文件
- **FR-030**: 系统 MUST 删除 `models.py` 中 `ExperienceMemory` 与 `ExperienceRule` dataclass
- **FR-031**: 系统 MUST 删除 `decision_commits.experience_json` 列（含 auto-migration），`JournalStore` 不再写该列
- **FR-032**: 系统 MUST 移除 `arena experience distill / show / merge / sessions` 4 个 CLI 子命令
- **FR-033**: 系统 MUST 删除 4 个 GSSC 相关测试文件，新增 ≥ 25 个测试覆盖：memory 写入（≥ 6）、reflection 写文件（≥ 6）、middleware 注入（≥ 5）、load_skill tool（≥ 4）、防过拟合算法等价（≥ 4）

#### 非功能性

- **FR-034**: 不引入新 runtime 依赖（不引入向量库、Rust crates、外部 OpenViking server）
- **FR-035**: 兼容现有 LangChain 1.2+ `create_agent` 接口；middleware 集成不破坏现有 fallback / streaming
- **FR-036**: 仅 drop `decision_commits.experience_json` 一列，保留其他列与现有数据完整性

### Key Entities

- **Skill**：高层能力包（5 个之一）。映射为 `agent_skills/<skill-name>/SKILL.md`。属性：name、description（frontmatter）+ body（含 active patterns 摘要 + forbidden 摘要 + agent role + usage rules）
- **PatternRecord**（memory 层数据，**非 Skill**）：单条蒸馏出的 pattern。映射为 `agent_memory/<agent>/patterns/<name>.md`。属性：name、description、regime_tags、pnl_track、maturity、source_cycles、body
- **CaseRecord**（memory 层数据）：单个 trading cycle 的原始执行记录。映射为 `agent_memory/<agent>/cases/<cycle_id>.md`。属性：cycle_id、timestamp、pair、snapshot_summary、agent_analysis、verdict、risk_gate、execution_status、final_pnl、applied_patterns
- **AgentSkillSet**：单 agent 一次 cycle 加载的 skill 集合（含 own SKILL.md + trading-knowledge SKILL.md）
- **ReflectionRun**：一次 reflection 任务的结构化日志（处理 cases 数 / 创建 patterns / 更新 / archive）
- **CurationRun**：一次 SKILL.md 整理任务的日志（人工或 LLM 触发；输入 patterns 输出 SKILL.md 草稿）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**：迁移完成后单个 trading cycle 中 4 个 agent 的 prompt token 数总和（含 load_skill tool 调用累计）较旧系统下降 ≥ 30%
- **SC-002**：`agent_skills/` 进 git 文件总数 ≤ 6（5 个 SKILL.md + 可选 .gitkeep）；运维者用 `cat agent_skills/<skill>/SKILL.md` ≤ 30 秒可审查任一 skill
- **SC-003**：完成 ≥ 14 天运行后，`agent_memory/<agent>/cases/` 累计 cycles 数 ≥ trading cycle 实际执行次数（不丢数据）
- **SC-004**：完成 ≥ 14 天 reflection 运行后，`agent_memory/<agent>/patterns/` 自动产生 ≥ 1 条 maturity=active 的 pattern（验证防过拟合算法在新存储下仍可正常晋升）
- **SC-005**：`arena skills curate` 命令运行后，输出的 SKILL.md 草稿 frontmatter 合规、body 包含至少一条 active pattern 的引用
- **SC-006**：完成 ≥ 14 天后 verdict reasoning 中 `applied:` 引用率 ≥ 60%（在有 active pattern 时）
- **SC-007**：删除旧代码后 `grep -rn "ExperienceMemory\|gather_packets" src/ tests/` 返回 0 结果；干净 PostgreSQL 启动 cycle 不需种子数据
- **SC-008**：全套件测试通过率 100%；新增测试覆盖 memory 写入（≥6）、reflection（≥6）、middleware（≥5）、load_skill（≥4）、防过拟合等价（≥4）共 ≥ 25 项
- **SC-009**：`load_skill` tool 在 ≥ 14 天 reflection 运行后被实际调用（频率 > 0），`skill_not_found` error 比例 < 20%
- **SC-010**：reflection job 失败注入测试中 trading cycle 完成率 100%（reflection 异常不阻塞）

## Assumptions

- 4 个 agent 当前已存在且 prompt 模板可清晰提取到 SKILL.md（无需重写决策逻辑）
- LangChain 1.2+ `create_agent` 的 `middleware=` 参数稳定；`AgentMiddleware.wrap_model_call` 与 `AgentMiddleware.tools` 类变量行为符合 https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant 描述
- 当前 4 层防过拟合算法在 `learning/reflect.py` 可被剥离为不依赖 DB 的纯函数
- `agent_memory/` 数据保留无明确容量上限——预期长期积累 ~50 cycles/天 × 365 天 = ~18K cycles 文件 / 年 / agent，约 ~70K 文件 / 年总量；filesystem inode 与 git status 性能不敏感（gitignored）
- `agent_skills/` 5 个 SKILL.md 总规模 ≤ 50KB；git 仓库容量影响忽略
- regime tags 集合（trending_up / range_bound / etc）继承 `learning/regime.py` 当前定义
- reflection 触发频率沿用 `[experience] every_n_cycles` 配置
- SKILL.md 整理（curation）频率与 reflection 解耦：reflection 自动 + 频繁；curation 较慢（每周或手工）
- 旧 `experience_json` 数据完全丢弃；新系统冷启动可接受（前 2 周 patterns 为空）
- 前端 `/decisions/:id` 详情页可降级（不展示 experience 字段）

## Out of Scope

- 旧 `experience_json` → memory 文件的数据迁移：明确不做
- A/B 模式（同时跑 legacy + skills 双轨）：不做
- 向量检索 / embedding：不引入
- 部署 OpenViking server / VikingDB：不引入
- `agent_memory/` 自动 retention 策略（按时间删除老文件）：本期保留全部，retention 留 follow-up
- `agent_memory/` 跨机器同步（如 rsync / S3 备份）：不在本 spec 范围
- 前端 UI 展示 cases / patterns：不做（agent_memory 是分析数据，不在 web UI）
- LLM-driven SKILL.md 自动整理流程的实现细节：本期 CLI 命令存在，但具体 prompt 设计 + 评估机制留 follow-up
- Cross-trading-pair 经验泛化：保持当前行为
- `load_skill` 响应缓存：每次调用都直读文件
