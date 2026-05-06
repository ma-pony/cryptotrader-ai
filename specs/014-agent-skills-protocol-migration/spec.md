# Feature Specification: Agent Skills 协议迁移

**Feature Branch**: `014-agent-skills-protocol-migration`
**Created**: 2026-05-06
**Status**: Draft
**Input**: User description: "重构 trading agent 的结构化经验记忆系统：用基于文件的 Anthropic Skills 协议替代当前 DB-based ExperienceMemory"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 文件即记忆：人工审查 + git 追溯（Priority: P1）

作为运维者，我希望 trading agent 的"经验"以人类可读的 markdown 文件形式存在仓库里，每一条 pattern / forbidden / instruction / shared knowledge 都有单独文件并随 reflection 自动更新；任何一次记忆改动（新增、PnL 反馈更新、自动废弃）都通过 git diff 能看见，无需查 DB。

**Why this priority**: 当前 DB JSONB 单列存储让经验**对外不可见**——查不到、改不动、不能 PR review。文件存储是后续所有改进的前置条件，没它则单写者模型、middleware 注入、Anthropic 协议对齐都无从谈起。

**Independent Test**: 即便 LangChain middleware 还没接入、agent 还在用旧注入逻辑，也能：(1) 启动 reflection job 写出第一批 markdown 文件；(2) 用 `cat agent_skills/tech/patterns/*.md` 直接审查；(3) 手动改一条 instructions 然后看到下一个 cycle agent 拿到的是新版本。MVP 价值即"经验透明化"，独立可交付。

**Acceptance Scenarios**:

1. **Given** 系统冷启动（无任何蒸馏记录），**When** reflection job 第一次运行 ≥ 3 个 trading cycle 后，**Then** `agent_skills/{agent}/patterns/` 下出现至少一个有效 markdown 文件，含合规 frontmatter（name、agent、description、regime_tags、pnl_track、maturity）+ body
2. **Given** 一个已存在的 pattern 文件 `funding_squeeze_long.md`，**When** reflection job 检测到该 pattern 命中过 5 个新案例且 win_rate 跌至 35%，**Then** 该文件 frontmatter 的 `pnl_track.cases` 加 5 且 `maturity` 改为 `deprecated`，文件被移到 `archive/` 子目录
3. **Given** 用户手工编辑 `tech/instructions.md` 加了一条新约束并提交 git，**When** 下一个 trading cycle 运行，**Then** TechAgent 的 system prompt 包含该新约束（无需重启、无需 DB 操作）
4. **Given** 仓库 clone 到一台新机器、空 PostgreSQL，**When** 启动 scheduler 跑第一个 cycle，**Then** agent 行为与原机器一致——所有"经验"来源都在 git 跟踪的文件里，零状态依赖于 DB

---

### User Story 2 — Middleware 自动注入：每个 agent 节点拿自己的 skills（Priority: P1）

作为开发者，我希望每个 agent 节点（tech / chain / news / macro）通过 LangChain `create_agent` 的 middleware 机制**自动**加载属于自己的 skills，按当前 regime 过滤后注入到 system prompt——agent 节点代码里不需要显式调任何 GSSC 或 selector 函数。

**Why this priority**: 这是把"经验作为文件"和"agent 实际能用上"连起来的桥梁。没有它，文件只是好看不顶用；有了它，注入逻辑下沉到框架层，agent 业务代码极简。

**Independent Test**: 写一个集成测试：mock 4 个 agent 各 5 条 skills（含不同 regime_tags），构造一个 `range_bound + low_funding` 的 ArenaState，运行 graph 一个完整 cycle，断言：(a) tech 节点 LLM 收到的 system_prompt 含 tech 的 5 条中匹配 regime 的若干条 description；(b) chain 节点收到 chain 的；(c) 各自互不混杂；(d) shared/ 下的 knowledge 4 个节点都收到。

**Acceptance Scenarios**:

1. **Given** `agent_skills/tech/patterns/` 下 10 条 patterns，其中 4 条含 `regime_tags: [trending_up]`、3 条 `[range_bound]`、3 条 `[high_funding]`，**When** 当前 regime 为 `[range_bound]` 的 trading cycle 运行 tech 节点，**Then** 注入 system prompt 的 patterns descriptions 仅包含 `range_bound` 的 3 条（其余被过滤）
2. **Given** `agent_skills/shared/funding_rate.md` 含 funding rate 阈值定义，**When** 任意 agent 节点运行，**Then** 4 个节点的 system prompt 都包含 funding_rate.md 的 description
3. **Given** middleware 加载 tech 的 skills 时遭遇某个文件 frontmatter 损坏，**When** 节点继续运行，**Then** 该损坏文件被跳过且 logger.warning 输出文件路径 + 解析错误，其他文件正常注入，cycle 不崩
4. **Given** middleware 注入完毕，**When** 检查 prompt token 数，**Then** 同一 regime 下的 token 总数较旧 GSSC 系统下降 ≥ 30%

---

### User Story 3 — 单写者反思：reflection 改写文件 + 防过拟合保留（Priority: P1）

作为系统设计者，我希望反思流程从"DB upsert 旧 ExperienceMemory" 切换到"文件读写 + 单写者保证"——只有 reflection job 写文件，4 个 agent 节点 read-only；保留当前 4 层防过拟合算法（regime filter、verify、maturity assignment、对手验证），保留 PnL-based maturity 演化（observed → probationary → active → deprecated）。明确不引入时间衰减层。

**Why this priority**: 防过拟合是当前系统的核心交易特化 know-how，不能在迁移中丢失；单写者模型是文件存储下避免 race condition 的最简方案。这两条决定了"迁移后系统是否仍然可信"。

**Independent Test**: 构造一组合成历史 commits（含已知胜负 PnL 分布），跑 reflection job，对比新算法与旧算法在以下 4 个 anti-overfitting 维度的输出是否数值等价：(1) regime-aware 胜率统计、(2) 最少样本量门槛、(3) 全局 vs 区段统计差距识别、(4) 对手验证（forbidden 是否真有相反方向证据）。等价即视为 know-how 无损迁移。

**Acceptance Scenarios**:

1. **Given** 一批合成 commits 包含一个明显但仅在 trending_up regime 出现的 pattern，**When** reflection job 运行，**Then** 该 pattern 写入文件时 `regime_tags` 仅含 `trending_up`（非 `[trending_up, range_bound, ...]` 全打），`maturity=probationary`（样本不足以晋升 active）
2. **Given** 已 active 的 pattern 因后 30 个 cycle 命中胜率跌破 maturity 降级阈值，**When** reflection job 运行，**Then** 该 pattern 文件被移到 `archive/`，frontmatter `maturity=deprecated`，原文件路径不再被 loader 加载
3. **Given** reflection job 在写第 3 个文件时进程被强杀，**When** 系统重启后下一次 cycle 运行，**Then** 已写入的 2 个文件保持完好（frontmatter 合法、可被 parse），未写完的第 3 个不存在或可被识别为部分写入并跳过；trading cycle 不受影响
4. **Given** 4 个 agent 节点正在并行加载 skills，**When** reflection job 同时启动想写文件，**Then** reflection job 等待短暂时间窗后写入；agent 节点本次 cycle 看到的是写入前的旧版本（不混读）

---

### User Story 4 — Verdict 显式归因：reasoning 含 `applied: <skill>`（Priority: P2）

作为反思系统的设计者，我希望 verdict 节点输出的 reasoning 文本里**显式声明**当前决策应用了哪些 skills（如 `applied: funding_squeeze_long`），便于 reflection job 在 PnL 反馈时**精确**归因到具体 skill 而不是粗略地"那个 cycle 用过的所有 skills 都加 1 分"。

**Why this priority**: 没它系统也能跑（粗归因仍可工作），但有它能让 PnL 反馈精度大幅提升——deprecate 决策更准、新 pattern 晋升更稳。属于精度提升而非功能性。

**Independent Test**: 跑 50 个 trading cycle，统计 verdict reasoning 中 `applied:` 出现频率 + 引用的 skill_name 是否能在 `agent_skills/` 中找到对应文件。频率 ≥ 60% 视为格式合规。

**Acceptance Scenarios**:

1. **Given** verdict 决定 short ETH 时引用了 `crowded_long_squeeze` pattern，**When** verdict reasoning 文本生成，**Then** 文本中包含字符串 `applied: crowded_long_squeeze`（也允许 `applied: tech::crowded_long_squeeze` 跨 agent 显式形式）
2. **Given** verdict 引用了一个不存在的 skill_name（hallucination），**When** reflection job 解析 reasoning，**Then** 该引用被忽略并 logger.warning，不影响存在 skill 的归因更新
3. **Given** 同一 cycle reasoning 引用了 3 条不同 skills，**When** 该 cycle 平仓 PnL 为 +120 USDT，**Then** 3 条 skills 的 `pnl_track.cases` 各 +1，`avg_pnl` 各按 +120 / 现有 case 数加权更新

---

### User Story 5 — 旧系统全删：clean break（Priority: P2）

作为代码维护者，我希望迁移完成后旧的 GSSC pipeline、`ExperienceMemory` / `ExperienceRule` dataclass、`success_patterns / forbidden_zones / strategic_insights` 三分法、`decision_commits.experience_json` DB 列、`arena experience distill/show/merge/sessions` CLI 子命令以及 4 个 GSSC 测试文件**完全删除**——不留任何 fallback / legacy 代码路径，新系统从零冷启动。

**Why this priority**: 留兼容层意味着双重维护成本；clean break 让代码库真正变小、维护性变好（review 目标 d）。但这是工程整洁度而非 PnL 改进，故 P2。

**Independent Test**: PR diff 显示 `learning/context.py`、`models.py` 中相关 dataclass、`journal/store.py` 中 experience_json 列、CLI 命令注册全部消失；`grep -r "ExperienceMemory" src/ tests/` 返回 0 结果；空 PostgreSQL 启动后 trading cycle 跑通。

**Acceptance Scenarios**:

1. **Given** 全套测试运行，**When** `grep -rn "ExperienceMemory\|ExperienceRule\|success_patterns\|forbidden_zones\|strategic_insights\|gather_packets\|select_packets\|structure_experience" src/ tests/` 执行，**Then** 0 命中
2. **Given** 一个干净 PostgreSQL 实例，**When** 启动 scheduler 跑首个 cycle，**Then** 不需任何"先创建 experience_json 数据"的种子步骤，cycle 正常完成
3. **Given** alembic-style migration 运行，**When** 检查 `decision_commits` 表结构，**Then** `experience_json` 列已被 drop（PostgreSQL `\d decision_commits` 验证），且现有 commits 行不丢失

---

### Edge Cases

- **空目录冷启动**：`agent_skills/{agent}/patterns/` 下无文件时，loader 返回空列表，agent 系统 prompt 里 patterns 段为空字符串或省略——agent 不应崩溃，应当输出基于纯 instructions + market data 的分析
- **frontmatter 损坏**：YAML 解析错误的文件被跳过 + warning，不阻塞同目录其他文件加载
- **`regime_tags` 为空**：被视为"任何 regime 都适用"——加载时不被 regime 过滤掉
- **同 agent 同名 pattern 冲突**：reflection 写入前检测同名文件，若 frontmatter `manually_edited: true` 则只更新 `pnl_track`、保留 body；否则全量 rewrite
- **跨 agent 同名 pattern**：路径自然隔离（`tech/patterns/X.md` vs `chain/patterns/X.md`），引用时用 `{agent}::{name}` 消歧
- **shared/ 文件被 agent 私有覆盖意图**：禁止；shared/ 严格只读，agent 不能 override
- **reflection job 失败**：trading cycle 必须**继续运行**，使用上一次成功蒸馏的快照；reflection 失败不阻塞下一个 cycle
- **频繁 git diff noise**：reflection 的自动写入应单独 commit（如有 commit 流程），与代码 commit 物理分离；不在本 spec 强制要求

## Requirements *(mandatory)*

### Functional Requirements

#### 存储层

- **FR-001**: 系统 MUST 在仓库根目录创建 `agent_skills/` 目录，与 `src/`、`config/` 同级，受 git 跟踪
- **FR-002**: 目录结构 MUST 按 4 个 agent（tech / chain / news / macro）分子目录，每个子目录含 `instructions.md`、`patterns/`、`forbidden/`、`archive/` 四类
- **FR-003**: 系统 MUST 提供 `agent_skills/shared/` 目录用于跨 agent 共享的领域常识（funding_rate / regime_definitions / 等）
- **FR-004**: 每个 skill 文件 MUST 是合法 markdown 文件，含 YAML frontmatter（name、agent、description、regime_tags、pnl_track、maturity、created、source_commits、version）+ body 段
- **FR-005**: 文件命名 MUST 遵循 `{agent}/{kind}/{snake_case_name}.md` 格式；跨 agent 引用 MUST 使用 `{agent}::{name}` 消歧

#### 加载与注入

- **FR-006**: 系统 MUST 提供一个 `SkillsInjectionMiddleware`，可挂载到 LangChain `create_agent` 的 `middleware=` 参数
- **FR-007**: 4 个 agent 节点（tech_analyze / chain_analyze / news_analyze / macro_analyze）MUST 通过该 middleware 自动加载并注入对应 agent 的 skills
- **FR-008**: 注入策略 MUST 是**静态注入**（每次 LLM 调用前拼接到 system_prompt），非 tool-calling 动态加载
- **FR-009**: 系统 MUST 按当前 regime_tags 过滤 patterns / forbidden（仅注入 regime 匹配或 regime_tags 为空的 skill），instructions / shared 不过滤
- **FR-010**: 注入到 prompt 的 MUST 是 skill 的 description（来自 frontmatter），非完整 body
- **FR-011**: shared/ 目录的所有文件 MUST 同时注入到 4 个 agent 节点的 prompt
- **FR-012**: 加载过程中遇到 frontmatter 损坏的文件 MUST 跳过且 logger.warning，不阻塞其他文件加载
- **FR-013**: 加载结果 MUST 是 read-only——agent 节点禁止写入 `agent_skills/`

#### 反思与写文件

- **FR-014**: 系统 MUST 提供一个 reflection job（替代当前 `learning/reflect.py` 的 DB upsert 路径），输入为最近 N 个 commits，输出为对 `agent_skills/{agent}/` 下文件的增删改
- **FR-015**: reflection job MUST 是单写者——同一时刻只能有一个实例运行（通过文件锁、运行时 mutex 或外部 cron 编排实现）
- **FR-016**: reflection 算法 MUST 完整保留 4 层防过拟合机制（pure-PnL 驱动，无时间维度），具体包括：
  - **L1 — regime-aware 胜率统计**：仅统计与该 pattern 同 regime tag 的历史样本，不能用全局胜率混淆
  - **L2 — 最少样本量门槛**：低于 N 个独立 case（当前 N=5）的候选 pattern 不晋升 maturity
  - **L3 — 全局 vs 区段差距识别**：候选 pattern 必须在区段（regime 内）显著优于全局基线，否则视为过拟合噪声
  - **L4 — 对手验证（forbidden 专属）**：候选 forbidden_zone 必须有"反方向操作真的亏损"的相反方向证据，不仅"该方向操作不盈利"
  - 注：明确不引入"时间衰减"层（基于 last_active 自动降级 maturity）。理由：pattern 可能仅因对应 regime 长期未出现而无 `applied`，并非 pattern 本身失效；纯 PnL 反馈触发的 L1+L3 已能识别真正失效的 pattern
- **FR-017**: skill maturity 演化 MUST 保留当前 4 级（observed → probationary → active → deprecated），由 PnL track 触发；deprecated 的文件 MUST 移到 `archive/` 子目录而非删除
- **FR-018**: reflection 失败 MUST NOT 阻塞下一个 trading cycle——cycle 使用上一次成功的蒸馏快照（即文件系统当前状态）。"失败"包括：进程崩溃、文件 IO 异常、解析异常、4 层防过拟合算法异常
- **FR-018a**: reflection 写文件 MUST 使用原子写（写到 temp file 然后 atomic rename），保证部分写入不会留下损坏文件；archive 操作 MUST 使用 git mv 或 atomic rename
- **FR-019**: reflection 检测到 frontmatter 含 `manually_edited: true` 的文件时，MUST 仅更新 `pnl_track` 与 `last_active`，禁止覆写 body 或其他 frontmatter 字段
- **FR-019a**: `manually_edited: true` 字段 MUST 由用户**手工添加**到 frontmatter（用户编辑 body 后自行设置）；本期不实现"自动检测 body diff 然后加 flag"的 pre-commit hook 等机制——保留为 follow-up

#### Verdict 显式归因

- **FR-020**: verdict 节点的 prompt MUST 强制要求 LLM 在 reasoning 文本中以 `applied: <skill_name>` 或 `applied: <agent>::<skill_name>` 形式声明应用了哪些 skills（如未应用任何 skill 则可省略）
- **FR-021**: reflection job 解析 verdict reasoning 时 MUST 提取所有 `applied: ...` 引用，并将该 cycle 的 PnL 反馈精确归因到对应 skill 文件
- **FR-022**: 引用了不存在的 skill_name 时，reflection MUST 跳过该引用并 logger.warning（视作 hallucination），不影响其他存在 skill 的归因

#### 删除清单（清旧）

- **FR-023**: 系统 MUST 删除 `src/cryptotrader/learning/context.py`（GSSC pipeline）整个文件
- **FR-024**: 系统 MUST 从 `models.py` 删除 `ExperienceMemory` 与 `ExperienceRule` dataclass 定义
- **FR-025**: 系统 MUST 删除 `decision_commits.experience_json` 列（含相应 alembic-style auto-migration），且 `JournalStore` 的写入路径不再写该列
- **FR-026**: 系统 MUST 从 `arena` CLI 移除 `experience distill / show / merge / sessions` 4 个子命令
- **FR-027**: 系统 MUST 删除当前 4 个 GSSC 相关测试文件，新增 ≥ 20 个测试覆盖 loader / middleware / reflection writer / 防过拟合算法等价性

#### 非功能性

- **FR-028**: 系统 MUST 不引入新的 runtime 依赖（不引入向量库、Rust crates、外部 OpenViking server 等）
- **FR-029**: 系统 MUST 兼容现有 LangChain 1.2+ `create_agent` 接口；用 `middleware=` 参数集成不破坏现有 fallback / streaming 能力
- **FR-030**: 系统 MUST 保留 `decision_commits` 表的其他列与现有数据完整性（仅 drop `experience_json`）

### Key Entities

- **Skill**：一条经验记录的运行时表示。属性包括：所属 agent（tech / chain / news / macro / shared）、kind（pattern / forbidden / instruction / knowledge）、name、description（一句话摘要）、body（完整内容）、regime_tags、pnl_track（cases / win_rate / avg_pnl / last_active）、maturity（observed / probationary / active / deprecated）、source_commit_hashes、version。映射为单个 markdown 文件
- **AgentSkillSet**：单个 agent 节点在一次 cycle 中加载的所有 skills 的集合，含 instructions（单条）、patterns（多条，已 regime 过滤）、forbidden（多条，已 regime 过滤）、knowledge（来自 shared，多条不过滤）
- **ReflectionRun**：一次反思任务的执行记录，输入为 commits 范围与 regime 上下文，输出为对 `agent_skills/` 的增删改清单（创建、frontmatter 更新、归档）
- **AppliedSkillReference**：verdict reasoning 中 `applied: <name>` 形式的归因记录，与具体 commit 的 PnL 关联，供 reflection 精确归因

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**：迁移完成后，单个 trading cycle 中 4 个 agent 的 prompt token 数总和较当前系统下降 ≥ 30%（基线测量取迁移前最近 100 个 cycle 的 token 中位数）
- **SC-002**：迁移完成后，运维者无需查询 PostgreSQL 即可在 ≤ 30 秒内审查任意 agent 当前所有 active patterns（通过 `cat agent_skills/{agent}/patterns/*.md`）
- **SC-003**：完成 ≥ 14 天 reflection 运行后，reflection job 自动产生 ≥ 1 条 maturity=active 的 pattern（验证防过拟合算法在新存储下仍可正常晋升 skill）
- **SC-004**：完成 ≥ 14 天 reflection 运行后，verdict reasoning 中 `applied: <skill>` 引用率 ≥ 60%（在有 active pattern 可用的 cycle 中）
- **SC-005**：删除旧代码后 `grep -rn "ExperienceMemory\|ExperienceRule\|success_patterns\|gather_packets\|select_packets\|structure_experience" src/ tests/` 返回 0 结果；干净 PostgreSQL 启动 trading cycle 不需任何 experience 种子数据
- **SC-006**：全套件测试通过率 100%；新增测试覆盖加载（≥ 6）、middleware 注入（≥ 4）、reflection 写文件（≥ 6）、防过拟合算法等价（≥ 4）共 ≥ 20 项
- **SC-007**：reflection job 失败注入测试中，trading cycle 完成率 100%（即 reflection 异常不阻塞下一个 cycle）
- **SC-008**：从仓库 clone 到运行起 trading cycle 的时间（含 docker compose up + arena migrate + arena run），相比当前系统不变或下降（不引入额外初始化负担）

## Assumptions

- 4 个 agent 当前已存在且其 prompt 模板内容可清晰提取到 `instructions.md`（无需重写 agent 决策逻辑）
- LangChain 1.2+ `create_agent` 的 `middleware=` 参数稳定可用，且支持在 LLM 调用前拦截/修改 system_prompt（已通过 `inspect.signature` 验证存在）
- 当前防过拟合算法的核心实现在 `learning/reflect.py` 中，可被剥离为不依赖 DB 的纯函数；本期采纳其中 4 层（不含时间衰减）
- `decision_commits.experience_json` 列的旧数据**不需要**被迁移到文件——新系统冷启动可接受，前 2 周经验为空
- 前端 `/decisions/:id` 详情页面在迁移期间可降级（不展示 experience_json 字段），是可接受的业务影响
- regime_tags 的取值集合（trending_up / range_bound / low_funding / high_funding / etc）保持当前 `learning/regime.py` 的定义不变
- reflection job 的触发频率与当前一致（每 N 个 cycle 运行一次，N 由 `[experience] every_n_cycles` 配置）
- `agent_skills/` 目录下 markdown 文件的总规模在长期运行（6 个月）后预计不超过 ~500 个文件、~5MB——git 仓库可承受
- 旧 commits 中已有的 `experience_json` 数据在迁移后允许丢失；前端任何依赖该字段的组件可降级或移除

## Out of Scope

- 数据迁移（旧 `experience_json` → 新文件）：明确不做，新系统冷启动
- A/B 模式比较框架（同时跑 legacy + skills 双轨）：不做，单线推进
- 向量检索 / embedding-based 相似度：不引入
- 部署 OpenViking server / VikingDB / Rust crates：不引入
- Anthropic Skills API 的 tool-calling 动态加载（agent 自己 `load_skill()`）：本期采用静态注入，将来可作为 follow-up
- Cross-trading-pair 的经验泛化（同 agent 在 BTC vs ETH 上分别有独立 skills 还是共享）：保持当前行为（按 agent 不按 pair）
- 自动化的 skill 命名冲突解决（同 agent 同 kind 下重名）：reflection 写入前手动检测 + 仅更新 `pnl_track`，更复杂的 merge / split 不在范围
