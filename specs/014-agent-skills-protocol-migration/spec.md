# Feature Specification: Agent Skills 协议迁移（双层架构）

**Feature Branch**: `014-agent-skills-protocol-migration`
**Created**: 2026-05-06
**Status**: Draft (v3 — 动态发现 + scope 字段)
**Input**: User description: "重构 trading agent 的结构化经验记忆系统：分两层——`agent_memory/` 保留全部历史记录用于分析（gitignored），`agent_skills/` 是从 memory 整理出的少量高层能力包（git 跟踪、Anthropic Skills 协议格式）"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 双层架构：Memory 与 Skills 解耦（Priority: P1）

作为系统设计者，我希望经验数据明确分为两层：底层 `agent_memory/`（gitignored，永久保留所有 cases / patterns，用于离线分析与反思）+ 高层 `agent_skills/`（git 跟踪，initial 5 个 SKILL.md、运行时可增长，是从 memory 整理出的能力描述）。Agent 只读 skills，reflection 写 memory + 整理 skills。

**Why this priority**: 这是整个 feature 的根本架构。混淆 memory 与 skills（之前每条 pattern 当 skill）导致 240+ 文件、git noisy、违反 Anthropic Skills 协议。必须分层。

**Independent Test**:
1. clone 仓库 → `agent_skills/` initial 5 个 SKILL.md 进 git 可见（运行时可由用户或 propose-new 增加）
2. `agent_memory/` 不在 git 中（`.gitignore` 包含）
3. trading cycle 跑完写入 `agent_memory/cases/` 但不污染 git
4. reflection 写入 `agent_memory/<agent>/patterns/` 也不污染 git
5. 运维者手工或触发 LLM 整理 SKILL.md → 该单文件 git diff 可见

**Acceptance Scenarios**:

1. **Given** 空仓库 + 干净 PostgreSQL，**When** 运行 1 个 trading cycle，**Then** `agent_memory/cases/` 出现该 cycle 的原始记录文件（per-cycle 单文件）；`git status` 不显示该文件
2. **Given** 累积 50 个 cycles 后跑 reflection，**When** 反思完成，**Then** `agent_memory/<agent>/patterns/` 出现蒸馏后的 pattern 文件，`agent_skills/` 下 SKILL.md **不变**（整理是独立流程）
3. **Given** 运维者手工编辑 `agent_skills/tech-analysis/SKILL.md`，**When** commit + push，**Then** git history 仅多 1 个文件改动，与 240+ pattern 文件方案对比 noise 大幅降低
4. **Given** 跨机器同步（A 机器训练，B 机器复盘），**When** B 机器 git pull，**Then** B 拿到最新 SKILL.md（足够 agent 决策）；如需深度分析 patterns/cases，B 通过其他渠道（rsync / S3）获取 `agent_memory/`

---

### User Story 2 — Memory 层：原始 cases + 蒸馏 patterns（Priority: P1）

作为反思系统的设计者，每个 trading cycle 完成后系统自动写入 `agent_memory/cases/<cycle_id>.md`（per-cycle 单文件，含 4 agent 各自 analysis + 综合 verdict）；reflection job 周期性读 cases 蒸馏出 patterns 并写入 `agent_memory/<agent>/patterns/<pattern_name>.md`（带 PnL 追踪 + regime tags + maturity，按 agent 分隔因为 patterns 是 agent 独有领域知识）。memory 层数据**永久保留**，不自动清理（用户可定期归档）。

**Why this priority**: memory 是后续所有分析的数据源——反思、调参、复盘、A/B 都从这里读。数据丢失或被自动清理会损害分析能力。

**Independent Test**:
1. 跑 100 个 cycles → `agent_memory/cases/` 出现 100 个 .md 文件（per-cycle 单文件）
2. 跑 reflection → `agent_memory/<agent>/patterns/` 出现新 pattern 或更新
3. `cat agent_memory/cases/2026-05-06-cycle-7ffc.md` 显示完整 cycle 记录（snapshot、4 agent analyses、verdict、PnL、reasoning 等）
4. `cat agent_memory/tech/patterns/funding_squeeze_long.md` 显示 pattern 元数据 + body

**Acceptance Scenarios**:

1. **Given** trading cycle 完成（含 verdict + 最终平仓 PnL），**When** journal 节点写完，**Then** `agent_memory/cases/<cycle_id>.md` 内容完整：snapshot_summary + 4 agent analyses + verdict + risk_gate + execution_status + final_pnl + applied_patterns
2. **Given** 累计 50 个 cases，**When** reflection 触发，**Then** 输出 0~N 个新 pattern 文件 + 0~M 个旧 pattern 的 PnL 更新；4 层防过拟合（regime / sample / global-vs-regional / forbidden adversarial）全保留
3. **Given** 已 deprecated 的 pattern 文件，**When** 60 天后用户想复盘，**Then** 该文件**仍在 archive/ 目录中可读**（不被自动删除）
4. **Given** 用户手工想分析某 case，**When** `cat agent_memory/cases/2026-05-06-cycle-7ffc.md`，**Then** 看到完整结构化数据（YAML frontmatter + markdown body）

---

### User Story 3 — Skills 层：≥5 个高层能力包（Anthropic 协议）（Priority: P1）

作为 trading agent，我读取的是**整理后**的 SKILL.md（initial 5 个：4 个 agent + 1 个 trading-knowledge；运行时可增长），每个文件遵循 Anthropic Skills 协议（frontmatter `name` + `description` + `scope` + 完整 body 含 active patterns 摘要）。SKILL.md **不是每 cycle 自动更新**，而是周期性（每周或按触发）由人工 / LLM 从 memory/patterns 整理而来。

**Why this priority**: agent 看到的"经验"必须是稳定、整理后的能力包——而非每 cycle 都变化的原始记录。SKILL.md 数量保持小（high-level capabilities，initial 5）符合 Anthropic Skills 协议本意，与本仓库 `.claude/skills/` 既有实践对齐。

**Independent Test**:
1. `ls agent_skills/` 列出 ≥ 5 个目录，initial 含 tech-analysis / chain-analysis / news-analysis / macro-analysis / trading-knowledge
2. 每个目录有且仅有 `SKILL.md`（initial 阶段；将来可加 reference 子文件）
3. 每个 SKILL.md frontmatter 含 `name`、`description` 与 `scope`（合规 Anthropic 协议 + 本扩展字段）
4. middleware 通过 `scope` frontmatter 动态发现并注入到对应 agent prompt
5. 运维者手工修改任一 SKILL.md → 下一个 cycle agent 看到改动（无需 reflection 触发）

**Acceptance Scenarios**:

1. **Given** 仓库初始化，**When** `ls agent_skills/`，**Then** 显示 initial 5 个目录（运行时可增长）；每个目录仅 1 个 SKILL.md（+ 可选 .gitkeep）
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

- **`agent_memory/` 不存在（首次启动）**：系统 MUST 自动创建目录树（顶级 `cases/` + 4 个 agent 子目录各含 `patterns/` 与 `archive/`，agent 列表来源于项目级常量 `VALID_AGENT_IDS`，见 FR-004c），不报错
- **`agent_skills/` 某 SKILL.md 缺失**：middleware 跳过该 skill 注入 + logger.warning，其他 skill 正常注入；agent 仍可工作
- **SKILL.md frontmatter 损坏**：跳过该文件 + warning，加载其他 skills
- **memory 数据量爆炸**（cases/ 文件 > 10K）：不自动清理；用户可手工归档（如压缩到 `agent_memory/archive/2026-Q1.tar.gz`），或后续 follow-up 加 retention 策略
- **`load_skill` 调用频率失控**：同一 cycle 调用 > 10 次时返回 `rate_limit_per_cycle` error
- **reflection 失败**：不阻塞下一个 cycle；下一个 cycle 仍读最新版 memory（reflection 失败前）
- **pattern 名冲突（同一 agent 同名）**：reflection 写入前检测，若旧 pattern frontmatter `manually_edited: true` 仅更新 `pnl_track`，否则全量 rewrite
- **跨 agent 同名 pattern**：天然路径隔离（`tech/patterns/X.md` ≠ `chain/patterns/X.md`）；引用规则见 FR-026（self-agent context 用 bare name；cross-agent context / verdict 节点用 `<agent>::<pattern>` 前缀形式）
- **manually_edited 设置方式**：用户手工编辑 SKILL.md 或 pattern 文件后**自行**在 frontmatter 加 `manually_edited: true`，本期不实现自动检测

## Requirements *(mandatory)*

### Functional Requirements

#### 双层架构

- **FR-001**: 系统 MUST 在仓库根创建两个顶级目录：`agent_memory/`（**gitignored**）与 `agent_skills/`（**git 跟踪**）
- **FR-002**: `.gitignore` MUST 包含 `agent_memory/`，确保该目录及子内容不进 git
- **FR-003**: `agent_memory/` 下 MUST 包含：(1) 顶级 `cases/` 目录（per-cycle 单文件，跨 agent 共享）；(2) 4 个 agent 子目录（tech / chain / news / macro），每个含 `patterns/` 与 `archive/` 二级目录（per-agent，因为 patterns 是 agent 独有的领域知识蒸馏）。该 4-agent 子目录布局与项目级常量 `VALID_AGENT_IDS = {"tech", "chain", "news", "macro"}` 对齐——见 FR-004c
- **FR-004**: `agent_skills/` 下 MUST 至少含 5 个 initial skill 目录（tech-analysis / chain-analysis / news-analysis / macro-analysis / trading-knowledge）；目录结构扁平（`agent_skills/<name>/SKILL.md`）；**支持运行时增长**——用户或 `arena skills propose-new` 可在该目录下新建 skill，middleware 自动发现，无需修改代码或重启
- **FR-004a**: 每个 SKILL.md 的 frontmatter MUST 含 `scope` 字段，取值 `shared`（跨 agent 注入）或 `agent:<agent_id>`（仅注入到指定 agent，agent_id ∈ {tech, chain, news, macro}）
- **FR-004b**: 系统 MUST NOT 硬编码 skill name → agent 的映射表；middleware 通过扫描 `agent_skills/*/SKILL.md` 的 frontmatter `scope` 字段做 runtime 发现
- **FR-004c**: `VALID_AGENT_IDS = {"tech", "chain", "news", "macro"}` 是**项目级常量**（与 `nodes/agents.py` / verdict / risk gate 等多处保持一致），不通过 scope 字段动态化；本 feature 不修改 agent 列表。FR-004b 禁止的是 *skill→agent 映射*的硬编码，不是 agent 列表本身——后者是项目结构常量
- **FR-005**: `agent_memory/` 数据 MUST 永久保留（不自动删除）；deprecated 的 pattern 文件移到 `agent_memory/<agent>/archive/`，仍可读取

#### Memory 层（cases + patterns）

- **FR-006**: 每个 trading cycle 完成时（含 journal_trade 节点）MUST 写入**单一** `agent_memory/cases/<cycle_id>.md` 文件（**per-cycle，非 per-agent**）；文件含 frontmatter（cycle_id / timestamp / pair / verdict_action / final_pnl）+ markdown body（4 个 agent 各自的 analysis 段落 + 综合 verdict_reasoning + 全部 applied_patterns 列表，按 `<agent>::<pattern>` 形式列出）。理由：verdict 是跨 4 agent 综合产物，per-cycle 单文件可避免 verdict_reasoning 重复存储与蒸馏时的去重问题；reflection 解析 applied_patterns 时按 `<agent>::` 前缀分发到对应 agent 的 PnL track。
- **FR-007**: cycle 写 cases 失败 MUST NOT 阻塞 cycle 主流程，logger.warning 记录后继续
- **FR-008**: reflection job MUST 周期性（按 `[experience] every_n_cycles`）读 `agent_memory/cases/` 蒸馏出 patterns，按 `applied_patterns` 中 `<agent>::` 前缀分发到对应 agent，写入 `agent_memory/<agent>/patterns/<pattern_name>.md`
- **FR-009**: pattern 文件 MUST 有 frontmatter（name / description / regime_tags / pnl_track / maturity / created / source_cycles / version）+ body（条件 / 案例 / 例外）
- **FR-010**: reflection 算法 MUST 完整保留 4 层防过拟合（pure-PnL 驱动）：
  - **L1 — regime-aware 胜率统计**：仅同 regime 历史样本
  - **L2 — 最少样本量门槛**：< N（默认 5）case 不晋升 maturity
  - **L3 — 全局 vs 区段差距**：区段必须显著优于全局基线
  - **L4 — 对手验证（forbidden 专属）**：相反方向有亏损证据
  - 不引入时间衰减层
- **FR-011**: skill maturity MUST 演化（observed → probationary → active → deprecated），由 PnL track 触发；deprecated 的 pattern 文件 MUST 移到 `archive/`，不删除
- **FR-012**: reflection 失败 MUST NOT 阻塞下一个 trading cycle
- **FR-013**: reflection 写文件 MUST 使用原子写（temp + rename）+ `threading.Lock` 进程内排他锁（单进程 single-writer 模型已足够，与 research.md R4 一致）；archive 操作用 atomic rename。如未来扩展为多进程写入，再升级为 `fcntl.flock`

#### Skills 层（initial 5 个 SKILL.md，运行时动态发现）

- **FR-014**: 每个 SKILL.md MUST 遵循 Anthropic Skills 协议格式：frontmatter 含 `name`（与目录名一致）+ `description`（一句话能力摘要）+ body（agent 角色 + active patterns 摘要 + forbidden zones 摘要 + 使用规则）
- **FR-015**: SKILL.md **不在每 cycle 自动更新**——是周期性整理产物（手工编辑、LLM 触发整理、或半自动 PR）
- **FR-016**: 系统 MUST 提供 CLI 命令 `arena skills curate <skill_name> [--llm]` 触发整理：读 `agent_memory/<agent>/patterns/` active 状态的 patterns + 当前 SKILL.md → 输出新 SKILL.md 草稿
- **FR-016a**: 系统 MUST 提供 CLI 命令 `arena skills propose-new [--scope <shared|agent:<id>>]` 触发**新 skill 提议**：分析 active patterns 找出共同 regime / theme 的子集，输出建议的新 SKILL.md 草稿到 stdout 或 `agent_skills/<proposed-name>/SKILL.md.draft`；**不自动创建 skill 文件**——需用户 review + manual save。**输入数据范围**：`--scope agent:<id>` 仅分析该 agent 的 `agent_memory/<id>/patterns/`（active 状态）；`--scope shared` 跨 4 agent 全部 active patterns 找跨域共性（regime/theme 重叠度高的子集）
- **FR-017**: SKILL.md frontmatter 含 `manually_edited: true` 时，整理流程 MUST 跳过该文件或仅追加到指定区段（如 `<!-- AUTO-DISTILLED-PATTERNS -->` 标记内）
- **FR-017a**: 用户手动创建新 SKILL.md（如 `agent_skills/momentum-trader/SKILL.md` 含合规 frontmatter）后，**下一个 trading cycle** MUST 自动 discovery 并按 `scope` 注入对应 agent prompt——无需重启服务、无需修改代码

#### 加载与注入

- **FR-018**: 系统 MUST 提供 `SkillsInjectionMiddleware`（继承 LangChain `AgentMiddleware`），可挂载到 `create_agent` 的 `middleware=` 参数；参考 https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant
- **FR-019**: 4 个 agent 节点（tech_analyze / chain_analyze / news_analyze / macro_analyze）MUST 通过 middleware 自动 discover 并注入：(1) 所有 `scope: shared` 的 skills；(2) 所有 `scope: agent:<self>` 的 skills（自身相关）
- **FR-019a**: middleware discovery 算法 MUST 扫描 `agent_skills/*/SKILL.md`，按 frontmatter `scope` 过滤；使用**进程内 LRU 缓存**（最大 N 个 SKILL.md 解析结果，由文件 mtime 失效——即每次访问对比缓存条目时间戳与磁盘 mtime，不一致则重新加载）；新增 / 删除 / 修改 skill 文件后下个 cycle 自动反映，无需重启
- **FR-020**: middleware MUST 在 `wrap_model_call` 时把所有匹配 skill 的 body 拼接到 `request.system_message.content_blocks`
- **FR-021**: middleware MUST 注册 `load_skill(name: str) -> str` tool（通过 `AgentMiddleware.tools` 类变量）；agent 可按需重新拉取任一 skill body
- **FR-022**: `load_skill` MUST 是 LangChain tool + 普通 Python 函数双接口（同实现），便于 verdict 节点等不走 create_agent 的代码直接调
- **FR-023**: `load_skill(name)` 仅需 1 参数（任一已注册的 skill name，runtime 扩展）；不需要 `agent::pattern` 嵌套形式（patterns 在 SKILL.md body 里）
- **FR-024**: middleware 加载 SKILL.md 失败（文件不存在 / frontmatter 损坏）MUST 跳过 + warning，不阻塞 cycle
- **FR-025**: `load_skill` rate-limit：同一 cycle 同一 trace_id 调用 > 10 次返回 `rate_limit_per_cycle` error
- **FR-025a**: `load_skill` tool 每次调用 MUST 通过 `metrics_collector` 增加 counter（label：`name`、`result ∈ {ok, skill_not_found, corrupt_file, rate_limit, dir_missing}`），用于 SC-009 频率统计与异常监控

#### Verdict 显式归因

- **FR-026**: verdict 节点 prompt MUST 强制要求 LLM 在 reasoning 中显式声明应用的 pattern，遵循以下命名规则：
  - **self-agent 上下文**（4 个 analysis agent 在自身 prompt 内引用本 agent 的 pattern）：bare name 如 `applied: funding_squeeze_long`，默认归属为发起 agent
  - **cross-agent 上下文**（verdict 节点综合 4 agent 输出时）：MUST 使用前缀形式 `applied: <agent>::<pattern_name>`（例如 `applied: tech::funding_squeeze_long`），消除跨 agent 同名歧义
  - **reflection 解析规则**：bare name 按发起 agent 解析；带前缀按 `<agent>` 解析；若 bare name 在多 agent 同时存在且无前缀，记 logger.warning 并跳过该次 PnL 归因
- **FR-026a** (added 2026-05-08): server-side enforcement — verdict 节点 MUST 在 LLM 输出落库前检查 reasoning 是否包含至少一个 `applied:` 引用；directional verdict (long/short/close) 缺失时 MUST 将 verdict.confidence × 0.5 并在 `verdict.guardrails` 列表中记录 `missing_applied`。理由：实盘观测 prompt 单靠"MUST 强制要求"达不到 100% 引用率，server-side ×0.5 惩罚把无归因 verdict 的实际下单尺寸压到原来 ¼（信心 ramp 联动），既保留 LLM 自由度又使无 PnL 归因路径不可能造成大仓损失。Hold action 不受此规则约束（无可归因路径）。
- **FR-027**: reflection job 解析 verdict.reasoning 时 MUST 提取所有 `applied:` 引用，定位到 `agent_memory/<agent>/patterns/<pattern_name>.md` 并更新 `pnl_track`
- **FR-028**: 引用了不存在的 pattern 时，reflection MUST 跳过 + logger.warning，不影响其他引用归因

#### 删除清单

- **FR-029**: 系统 MUST 删除 `src/cryptotrader/learning/context.py`（GSSC pipeline）整个文件
- **FR-030**: 系统 MUST 删除 `models.py` 中 `ExperienceMemory` 与 `ExperienceRule` dataclass
- **FR-031**: 系统 MUST 删除 `decision_commits` 表上的 GSSC 历史列（实际列名为 `experience_memory`，旧文档误称 `experience_json`；migration 同时尝试 drop 两个列名以兼容历史 schema），`JournalStore` 不再写该列
- **FR-032**: 系统 MUST 移除 `arena experience distill / show / merge / sessions` 4 个 CLI 子命令
- **FR-033**: 系统 MUST 删除 4 个 GSSC 相关测试文件，新增 ≥ 25 个测试覆盖：memory 写入（≥ 6）、reflection 写文件（≥ 6）、middleware 注入（≥ 5）、load_skill tool（≥ 4）、防过拟合算法等价（≥ 4）

#### 非功能性

- **FR-034**: 不引入新 runtime 依赖（不引入向量库、Rust crates、外部 OpenViking server）
- **FR-035**: 兼容现有 LangChain 1.2+ `create_agent` 接口；middleware 集成不破坏现有 fallback / streaming
- **FR-036**: 仅 drop `decision_commits` 上的 GSSC 列（`experience_memory` / `experience_json` 兼容名），保留其他列与现有数据完整性

### Key Entities

- **Skill**：高层能力包（initial 5 个，运行时可增长）。映射为 `agent_skills/<skill-name>/SKILL.md`。属性：name、description、scope（frontmatter）+ body（含 active patterns 摘要 + forbidden 摘要 + agent role + usage rules）
- **PatternRecord**（memory 层数据，**非 Skill**）：单条蒸馏出的 pattern。映射为 `agent_memory/<agent>/patterns/<name>.md`。属性：name、description、regime_tags、pnl_track、maturity、source_cycles、body
- **CaseRecord**（memory 层数据）：单个 trading cycle 的原始执行记录（**per-cycle 单文件**，跨 4 agent 共享）。映射为 `agent_memory/cases/<cycle_id>.md`。属性：cycle_id、timestamp、pair、snapshot_summary、agent_analyses（4 agent 各自段落）、verdict、risk_gate、execution_status、final_pnl、applied_patterns（按 `<agent>::<pattern>` 形式列出）
- **AgentSkillSet**：单 agent 一次 cycle 加载的 skill 集合，**动态 list[Skill]**，含所有 frontmatter `scope == "shared"` 与 `scope == f"agent:{self_id}"` 的 skills；initial 状态通常 2 个（own + shared），运行时可多于 2 个（用户或 propose-new 增加）
- **ReflectionRun**：一次 reflection 任务的结构化日志（处理 cases 数 / 创建 patterns / 更新 / archive）
- **CurationRun**：一次 SKILL.md 整理任务的日志（人工或 LLM 触发；输入 patterns 输出 SKILL.md 草稿）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**：迁移完成后单个 trading cycle 中 4 个 agent 的 prompt token 数总和（含 load_skill tool 调用累计）较旧系统下降 ≥ 30%。**Baseline 测量要求**：实施前 MUST 在旧系统跑 5 个 trading cycle，记录每个 agent 的 prompt token 数中位数与总和（含 GSSC pipeline 注入），结果记录在 PR 描述与 plan.md 中作为对照基线
- **SC-002**：`agent_skills/` 初始进 git 文件总数 ≤ 6（initial 5 个 SKILL.md + 可选 .gitkeep）；运行时可增长（用户或 LLM 提议创建新 skill），但每个新 skill 仍是 1 个 SKILL.md 文件——文件数与功能数线性，绝不会再退化为"一条 pattern 一个文件"；运维者用 `cat agent_skills/<name>/SKILL.md` ≤ 30 秒可审查任一 skill
- **SC-003**：完成 ≥ 14 天运行后，`agent_memory/cases/` 累计 cycles 数 ≥ trading cycle 实际执行次数（不丢数据）
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
- `agent_skills/` initial 5 个 SKILL.md 总规模 ≤ 50KB；运行时增长后单个 SKILL.md 仍应 ≤ 10KB；git 仓库容量影响忽略
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
- Skill 自动拆分 / 合并（auto split-merge）：当 skill body 过大 OR 多个 skill patterns 重叠时建议拆分/合并——本期不实现，留 follow-up
- 自动创建新 skill 文件：`arena skills propose-new` 仅输出 draft，不直接 write `agent_skills/`；用户必须 review + manual save
- Skill 命名冲突自动解决（如同名 SKILL.md 已存在时如何处理）：本期 propose-new 仅输出 stdout/draft，文件冲突时 fail；未来可加交互式 prompt
- Cross-trading-pair 经验泛化：保持当前行为
- `load_skill` 响应缓存：每次调用都直读文件
