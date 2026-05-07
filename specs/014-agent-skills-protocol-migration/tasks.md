# Tasks: Agent Skills 协议迁移（双层架构 v3）

**Input**: Design documents from `/specs/014-agent-skills-protocol-migration/`
**Prerequisites**: spec.md ✅、plan.md ✅、research.md ✅、data-model.md ✅、contracts/{skill_md,pattern_record,case_record}.schema.yaml + {load_skill,middleware}.contract.md ✅

**Tests**: 显式要求（FR-033 ≥ 25 测试，分布于 6 个测试文件）。所有测试任务标 ⚠️ 写在实现前，先 FAIL 再 GREEN。

**Organization**: 按 6 个 user story 分阶段；US1-US4 = P1（必做 MVP），US5-US6 = P2。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无依赖未完成任务）
- **[Story]**: US1 / US2 / US3 / US4 / US5 / US6（spec.md 中的 user story）
- 路径全部相对仓库根

## Path Conventions

- `agent_memory/`：顶级（gitignored）
- `agent_skills/`：顶级（git 跟踪）
- `src/cryptotrader/agents/skills/`：新增子模块
- `src/cryptotrader/learning/`：含新增 `memory.py` / `curation.py` / `skill_proposal.py`
- `src/cryptotrader/nodes/`：含新增 `reflection.py`
- `tests/`：扁平结构（与现有约定一致）

---

## Phase 1: Setup（共享基础设施）

**Purpose**: 项目初始化与目录骨架

- [x] T001 在 `.gitignore` 追加 `agent_memory/` 条目（与 plan.md 第 80 行一致）
- [x] T002 创建顶级目录骨架：`agent_memory/cases/.gitkeep`（per-cycle 单文件目录，FR-003）+ `agent_memory/{tech,chain,news,macro}/{patterns,archive}/.gitkeep`（per-agent patterns/archive，FR-003 + FR-004c）+ `agent_skills/{tech-analysis,chain-analysis,news-analysis,macro-analysis,trading-knowledge}/`
- [x] T003 [P] 验证 `pyyaml` 已在 pyproject.toml（用于 SKILL.md frontmatter 解析；plan.md Technical Context）；如未声明则添加
- [~] T004 [P] **(deferred — manual)** 测得旧系统 prompt token baseline：需用户在合并前手动 git checkout 上一 commit 跑 5 cycles 收集；或接受合并后再 retroactively 测对比。SC-001 ≥ 30% 阈值的硬验证留 follow-up。

**Checkpoint**: 目录就位、baseline 已记录、依赖齐备

---

## Phase 2: Foundational（阻塞所有 user story）

**Purpose**: 各 story 都依赖的核心数据结构、I/O 工具、frontmatter 解析；必须先完成

**⚠️ CRITICAL**: 该阶段未完成前任何 user story 不开工

- [x] T005 实现 `src/cryptotrader/agents/skills/__init__.py` + `src/cryptotrader/agents/skills/schema.py`：定义 dataclass `Skill`、`PatternRecord`、`CaseRecord`、`PnLTrack`、`AgentSkillSet`、`ReflectionRun`、`CurationRun`、`Maturity` 枚举；与 data-model.md 一一对齐
- [x] T006 [P] 实现 `src/cryptotrader/agents/skills/_frontmatter.py`：YAML frontmatter 解析 + 校验（基于 `contracts/skill_md.schema.yaml` / `pattern_record.schema.yaml` / `case_record.schema.yaml`）；解析失败抛 `CorruptFrontmatterError` 含路径与行号
- [x] T007 [P] 实现 `src/cryptotrader/agents/skills/_io.py`：`atomic_write(path, content)`（临时文件 + `os.rename`）+ 模块级 `threading.Lock`（FR-013 + research R4）；`ensure_memory_dirs()` + `ensure_skill_dirs()` 自动创建目录骨架
- [x] T008 [P] 在 `src/cryptotrader/agents/skills/_constants.py` 声明 `VALID_AGENT_IDS = frozenset({"tech", "chain", "news", "macro"})`（FR-004c 项目级常量）；导出供 middleware / memory / 校验使用

**Checkpoint**: 数据模型 + I/O 工具就绪，user story 实现可以并行展开

---

## Phase 3: User Story 1 - 双层架构解耦（Priority: P1）🎯 MVP 基线

**Goal**: agent_memory/ 与 agent_skills/ 边界落地——前者 gitignored、永久保留 cycle 数据；后者 git 跟踪、initial 5 个 SKILL.md。

**Independent Test**: clone 仓库 → `agent_skills/` 见 5 个目录；跑 1 cycle → `agent_memory/cases/<cycle_id>.md` 出现且 `git status` 干净；reflection 写 patterns 也不污染 git。

### Tests for User Story 1 ⚠️（FAIL → GREEN）

- [x] T009 [P] [US1] 在 `tests/test_two_layer_architecture.py` 写 ≥ 3 测试：(a) `.gitignore` 含 `agent_memory/` 条目；(b) `agent_skills/` 含 initial 5 目录每个含 SKILL.md；(c) 跑 mock cycle → `agent_memory/cases/` 出现文件且 `subprocess.run(["git","status","--porcelain"])` 不显示 `agent_memory/`

### Implementation for User Story 1

- [x] T010 [US1] 创建 5 个 initial SKILL.md 文件（合规 Anthropic 协议 frontmatter `name`+`description`+`scope`，FR-014 + FR-004a；body 含 active patterns 摘要 + forbidden 摘要 + agent role + usage rules）：
  - `agent_skills/tech-analysis/SKILL.md`（scope: `agent:tech`）
  - `agent_skills/chain-analysis/SKILL.md`（scope: `agent:chain`）
  - `agent_skills/news-analysis/SKILL.md`（scope: `agent:news`）
  - `agent_skills/macro-analysis/SKILL.md`（scope: `agent:macro`）
  - `agent_skills/trading-knowledge/SKILL.md`（scope: `shared`，内容含 funding_rate 含义、regime 定义、spot vs perp 语义；research R10 来源手工写）
  body 初始化只需 role 段；patterns 段 `<!-- AUTO-DISTILLED-PATTERNS -->` 占位（curation 后续填充）
- [x] T011 [US1] 在 `agent_memory/cases/.gitkeep` 旁附 `README.md` 说明此目录 gitignored（仅 .gitkeep 进 git，便于 fresh clone 后存在）

**Checkpoint**: 双层目录边界可独立验证；MVP baseline 就位

---

## Phase 4: User Story 2 - Memory 层（cases + patterns 自动写入）（Priority: P1）

**Goal**: 每 cycle 自动写 `agent_memory/cases/<cycle_id>.md`（per-cycle 单文件，FR-006）；reflection job 周期蒸馏 patterns（带 4 层防过拟合，FR-010）；patterns maturity 演化（observed → probationary → active → deprecated，FR-011）。

**Independent Test**:
- 跑 100 mock cycles → `agent_memory/cases/` 出现 100 个文件
- 跑 reflection → `agent_memory/<agent>/patterns/` 至少 1 条 active pattern
- mock 一组反例样本 → forbidden pattern 进 deprecated 移到 archive/

### Tests for User Story 2 ⚠️

- [x] T012 [P] [US2] `tests/test_agent_memory_writer.py` ≥ 6 测试：(a) 写 cycle 文件含 frontmatter（cycle_id / pair / verdict_action / final_pnl）；(b) body 含 4 个 agent 各自 analysis 段 + verdict + applied_patterns 列表；(c) 平仓后回填 final_pnl；(d) 写失败不抛异常（FR-007）；(e) 原子写（mid-write 进程崩溃后无半文件）；(f) per-cycle 单文件路径正确（`agent_memory/cases/`，非 per-agent）
- [x] T013 [P] [US2] `tests/test_reflection_pattern_distill.py` ≥ 6 测试：(a) 蒸馏新 pattern 写入 `agent_memory/<agent>/patterns/<name>.md`；(b) 已有 pattern 增量更新 PnL track；(c) maturity FSM 各转移点；(d) 反思失败不阻塞下一 cycle（FR-012）；(e) 解析 `<agent>::<pattern>` 前缀分发到对应 agent；(f) `manually_edited: true` 的 pattern 跳过 body 重写仅更新 pnl_track
- [x] T014 [P] [US2] `tests/test_anti_overfitting_equivalence.py` ≥ 4 测试，覆盖 FR-010 的 4 层：(a) L1 regime-aware 仅同 regime 样本计算 win rate；(b) L2 < N case 不晋升 maturity；(c) L3 区段必须显著优于全局；(d) L4 forbidden 反向亏损证据校验

### Implementation for User Story 2

- [x] T015 [US2] 实现 `src/cryptotrader/learning/memory.py`：`write_case(cycle_id, pair, agents_analyses, verdict, risk_gate, exec_status, final_pnl, applied_patterns) -> Path`，per-cycle 单文件写入（FR-006），用 T007 的 `atomic_write`
- [x] T016 [US2] 在 `src/cryptotrader/learning/memory.py` 实现 `update_final_pnl(cycle_id, pnl)`：平仓后回填 frontmatter `final_pnl`（用 atomic write）
- [x] T017 [US2] 在 `src/cryptotrader/learning/memory.py` 实现 `distill_patterns(cycles_window) -> ReflectionRun`：从 `agent_memory/cases/` 读窗口内 cases、应用 4 层防过拟合（FR-010）、写 / 更新 / archive 到 `agent_memory/<agent>/patterns/` 与 `archive/`；移植自原 `learning/reflect.py` 但去 DB 化
- [x] T018 [US2] 在 `src/cryptotrader/learning/memory.py` 实现 `update_pattern_pnl(applied_patterns: dict[str, str], pnl: float)`：解析 `<agent>::<pattern>` → 逐条 pattern 文件 `pnl_track.cases += 1` + `win_rate / avg_pnl` 增量更新（FR-027 + data-model.md PnLTrack 公式）
- [x] T019 [US2] 在 `src/cryptotrader/learning/memory.py` 实现 `_advance_maturity(pattern: PatternRecord) -> Maturity`：FSM observed → probationary → active → deprecated（data-model.md Maturity 状态机）；deprecated 时移至 `archive/`（atomic rename）
- [x] T020 [US2] 改造 `src/cryptotrader/nodes/journal.py`：journal_trade 节点末尾调 `learning.memory.write_case(...)`（per-cycle 单文件，含 4 agent analyses）；写失败 logger.warning 后继续不阻塞（FR-007）
- [x] T021 [US2] 在 `src/cryptotrader/nodes/journal.py` 加入平仓回填钩子：position close 事件触发 `update_final_pnl(cycle_id, pnl)` + `update_pattern_pnl(applied_patterns, pnl)`
- [x] T022 [US2] 创建 `src/cryptotrader/nodes/reflection.py`：graph 节点薄包装 `learning.memory.distill_patterns()`，捕获所有异常 logger.exception 后返回 unchanged state（FR-012）
- [x] T023 [US2] 改造 `src/cryptotrader/nodes/data.py:verbal_reinforcement`：按 `[experience] every_n_cycles` 触发 reflection 节点（FR-008）；删除原 GSSC pipeline 调用（experience injection 移交 middleware，US4 完成）

**Checkpoint**: cases 持续写入、patterns 自动蒸馏、maturity 自演化；US2 独立可验证

---

## Phase 5: User Story 3 - Skills 层（≥5 SKILL.md 动态发现 + curation + propose-new）（Priority: P1）

**Goal**: middleware 通过 frontmatter `scope` 字段动态发现 skills（无硬编码 mapping，FR-004b）；CLI `arena skills curate` 整理 SKILL.md 草稿；CLI `arena skills propose-new` 输出新 skill 草稿；用户手工新增 SKILL.md 后下个 cycle 自动 pickup（FR-017a）。

**Independent Test**:
- `arena skills curate tech-analysis --llm` 输出 `SKILL.md.draft`，frontmatter 合规
- 用户手工创建 `agent_skills/momentum-trader/SKILL.md`（scope: agent:tech）后，下个 cycle TechAgent prompt 自动含其 body
- `arena skills propose-new --scope shared` 输出跨 4 agent 共性的 draft

### Tests for User Story 3 ⚠️

- [x] T024 [P] [US3] `tests/test_skills_loader.py` ≥ 4 测试：(a) parse_skill_md 合规 frontmatter；(b) `discover_skills_for_agent("tech")` 返回 scope=shared 与 scope=agent:tech 的 skills；(c) frontmatter 损坏 → logger.warning 跳过该文件；(d) 文件 mtime 变化 → LRU 缓存自动失效（FR-019a）
- [x] T025 [P] [US3] `tests/test_skill_proposal.py` ≥ 3 测试：(a) `propose-new --scope agent:tech` 仅分析 `agent_memory/tech/patterns/`；(b) `propose-new --scope shared` 跨 4 agent 找 regime/theme 重叠子集（FR-016a）；(c) 输出 `.draft` 文件不直接覆盖 `agent_skills/`
- [x] T026 [P] [US3] `tests/test_skills_curation.py` ≥ 3 测试：(a) `curate --llm` 生成 `SKILL.md.draft`；(b) `manually_edited: true` 的 SKILL.md 整体跳过；(c) 含 `<!-- AUTO-DISTILLED-PATTERNS -->` 标记时仅替换该区段

### Implementation for User Story 3

- [x] T027 [US3] 实现 `src/cryptotrader/agents/skills/loader.py`：`parse_skill_md(path) -> Skill`、`discover_skills_for_agent(agent_id, skill_dir=Path("agent_skills")) -> list[Skill]`（按 frontmatter `scope` 过滤）、进程内 LRU 缓存（max 32）+ 每次访问对比磁盘 mtime 失效（FR-019a + middleware.contract.md）
- [x] T028 [US3] 实现 `src/cryptotrader/learning/curation.py`：`curate_skill(name: str, *, use_llm: bool) -> Path`，读 `agent_memory/<agent>/patterns/` active 状态 + 当前 SKILL.md → 输出 `agent_skills/<name>/SKILL.md.draft`（FR-015 SKILL.md 非每 cycle 自动更新，是 curation 独立流程产物 + FR-016 CLI 接口）；`manually_edited: true` 整体跳过（FR-017）；含标记区段则仅替换该区（FR-017）
- [x] T029 [US3] 实现 `src/cryptotrader/learning/skill_proposal.py`：`propose_new_skill(scope: str) -> Path`，分析 active patterns 找共同 regime/theme 子集，按 `--scope` 过滤范围（FR-016a）；输出到 stdout 或 `agent_skills/<proposed-name>/SKILL.md.draft`
- [x] T030 [US3] 在 `src/cryptotrader/cli/main.py` 注册 `arena skills curate <name> [--llm]` 与 `arena skills propose-new [--scope <shared|agent:<id>>]` 两个子命令

**Checkpoint**: 手工 / LLM curation + propose-new + 动态发现全链路就位

---

## Phase 6: User Story 4 - Middleware 自动注入 + load_skill tool（Priority: P1）

**Goal**: `SkillsInjectionMiddleware`（继承 `langchain.agents.middleware.AgentMiddleware`）通过 `wrap_model_call` 注入匹配 skills；同时 `tools = [load_skill_tool]` 注册 tool；4 agent 节点 `create_agent` 都挂该 middleware。

**Independent Test**:
- mock 5 SKILL.md → 4 agent system_prompt 各含 own + shared 的 body
- 调 `load_skill("tech-analysis")` 返回 body
- 调 `load_skill("nonexistent")` 返回 `{"error": "skill_not_found"}`
- 同 cycle 第 11 次调用返回 `rate_limit_per_cycle`

### Tests for User Story 4 ⚠️

- [x] T031 [P] [US4] `tests/test_skills_middleware.py` ≥ 5 测试：(a) own + shared 都注入；(b) own 缺失 → 仅注入 shared + warning（middleware.contract.md Failure Modes）；(c) shared 缺失 → 仅注入 own；(d) 两者都缺失 → 不修改 request；(e) frontmatter 损坏 → 跳过 + warning，cycle 不崩
- [x] T032 [P] [US4] `tests/test_load_skill_tool.py` ≥ 4 测试：(a) skill 存在返回 body；(b) skill_not_found；(c) corrupt_file；(d) rate_limit_per_cycle 第 11 次（load_skill.contract.md）

### Implementation for User Story 4

- [x] T033 [US4] 实现 `src/cryptotrader/agents/skills/tool.py`：`load_skill(name: str) -> dict` 普通函数 + `load_skill_tool` LangChain BaseTool（FR-022 双接口 + research R6 同实现双导出）；按 `agent_skills/<name>/SKILL.md` 解析 directory 而非硬编码列表（FR-023）；维护 per-trace_id 调用计数 + 第 11 次返回 rate_limit_per_cycle（FR-025）
- [x] T034 [US4] 在 `src/cryptotrader/agents/skills/tool.py` 添加 metrics：每次调用 `metrics_collector.inc_counter("load_skill_calls", labels={name, result})`（FR-025a + SC-009）
- [x] T035 [US4] 实现 `src/cryptotrader/agents/skills/middleware.py`：`SkillsInjectionMiddleware(AgentMiddleware)`（FR-018 继承 LangChain `AgentMiddleware`），类变量 `tools = [load_skill_tool]`（FR-021 通过 `AgentMiddleware.tools` 类变量注册）、`VALID_AGENT_IDS` 引用 T008 常量；`wrap_model_call(request, handler)` 调 `discover_skills_for_agent(self.agent_id)` 拼 addendum 追加到 `request.system_message.content_blocks`（FR-020 + middleware.contract.md Behavior 段算法）；单条 SKILL.md 加载失败 logger.warning 跳过、不阻塞 cycle（FR-024）
- [x] T036 [US4] 改造 `src/cryptotrader/agents/base.py:ToolAgent.create_agent`：`tools=[*self.tools, *SkillsInjectionMiddleware.tools]`、`middleware=[SkillsInjectionMiddleware(agent_id=self.agent_id)]`（middleware.contract.md Registration Example）
- [x] T037 [US4] 改造 `src/cryptotrader/nodes/agents.py`：4 agent 节点（tech_analyze / chain_analyze / news_analyze / macro_analyze）都通过 ToolAgent.create_agent 入口注入 middleware（FR-019 自动 discover 并注入 shared + agent:<self> 范围 skills）；删除原节点内手工 prompt 拼接经验段的逻辑

**Checkpoint**: 4 agent prompt 自动含 SKILL.md body + load_skill tool 可调；US1-US4 = MVP 完成

---

## Phase 7: User Story 5 - Verdict 引用 pattern 名称便于归因（Priority: P2）

**Goal**: verdict prompt 强制 LLM 用 `applied: <pattern_name>` 或 `applied: <agent>::<pattern_name>` 格式声明应用的 pattern；reflection 解析后定位到 `agent_memory/<agent>/patterns/<name>.md` 更新 PnL（FR-026 + FR-027）。

**Independent Test**: 跑 50 cycles，统计 verdict reasoning 中 `applied:` 频率；引用的 pattern 名 100% 能在 memory 找到对应文件（或被 logger.warning 标记 hallucination）。

### Tests for User Story 5 ⚠️

- [x] T038 [P] [US5] `tests/test_applied_pattern_parser.py` ≥ 4 测试：(a) bare name `applied: funding_squeeze_long` 按发起 agent 解析；(b) `applied: tech::funding_squeeze_long` 跨 agent 形式正确分发；(c) bare name 在多 agent 同时存在时 logger.warning 跳过该归因（FR-026 解析规则）；(d) 引用不存在 pattern 时 logger.warning 不影响其他归因（FR-028）

### Implementation for User Story 5

- [x] T039 [US5] 改造 `src/cryptotrader/nodes/verdict.py`：prompt 模板加入"reasoning MUST 用 `applied: <pattern>` 或 `applied: <agent>::<pattern>` 显式声明"指令（FR-026）
- [x] T040 [US5] 在 `src/cryptotrader/learning/memory.py` 实现 `parse_applied(reasoning: str, originating_agent: str | None) -> dict[str, list[str]]`：返回 `{agent_id: [pattern_name, ...]}`，bare name 按 originating_agent 解析（None=verdict 上下文需显式前缀）；歧义按 FR-026 解析规则处理
- [x] T041 [US5] 把 `parse_applied` 接入 T021 的平仓回填钩子：`update_pattern_pnl()` 调用前用 parser 解析 verdict_reasoning

**Checkpoint**: PnL 归因精确化，4 层防过拟合算法基于真信号工作

---

## Phase 8: User Story 6 - 旧系统全删（Priority: P2）

**Goal**: GSSC pipeline / `ExperienceMemory` dataclass / `decision_commits.experience_json` 列 / `arena experience` CLI / 4 GSSC 测试文件全部删除（FR-029-033 + SC-007）。

**Independent Test**: `grep -rn "ExperienceMemory\|ExperienceRule\|success_patterns\|forbidden_zones\|strategic_insights\|gather_packets\|select_packets\|structure_experience" src/ tests/` 返回 0 命中；干净 PostgreSQL 启动 cycle 无种子需求。

### Implementation for User Story 6

- [x] T042 [US6] 删除整个文件 `src/cryptotrader/learning/context.py`（GSSC pipeline，FR-029）
- [x] T043 [US6] 删除整个文件 `src/cryptotrader/learning/reflect.py`（DB upsert 路径，已被 `learning/memory.py` 替代）
- [x] T044 [US6] 修改 `src/cryptotrader/models.py`：删除 `ExperienceMemory` 与 `ExperienceRule` dataclass（FR-030）
- [x] T045 [US6] 修改 `src/cryptotrader/learning/verbal.py`：删除 `search_by_regime` 等 DB 检索函数；只保留必要 helper（如有）
- [x] T046 [US6] 在 `alembic/versions/` 新增 migration 文件 drop `decision_commits.experience_json` 列（FR-031）
- [x] T047 [US6] 修改 `src/cryptotrader/journal/store.py`：停止写 `experience_json` 列（FR-031）
- [x] T048 [US6] 修改 `src/cryptotrader/cli/main.py`：移除 `arena experience distill / show / merge / sessions` 4 个子命令（FR-032）
- [x] T049 [P] [US6] 删除 4 个 GSSC 测试文件：`tests/test_factorminer_*.py`、`tests/test_gssc_*.py`、`tests/test_experience_extraction.py`（FR-033）
- [x] T050 [US6] 跑 `grep -rn "ExperienceMemory\|ExperienceRule\|success_patterns\|forbidden_zones\|strategic_insights\|gather_packets\|select_packets\|structure_experience" src/ tests/` 验证 0 命中（SC-007）

**Checkpoint**: 旧轨道完全清除，单代码路径

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: 端到端验证、文档、性能基线对比

- [x] T051 [P] 更新 `specs/014-agent-skills-protocol-migration/quickstart.md`：以新双层架构 + per-cycle case 路径为准（已部分更新，再检查 commands 是否实际可跑）
- [x] T052 [P] 更新 `docs/ARCHITECTURE.md`：补充 agent_memory / agent_skills 双层架构章节，引用 spec 014
- [x] T053 [P] 在 `tests/test_skills_perf.py` 加 microbench：`discover_skills_for_agent` p95 ≤ 50ms、cycle 写 case p95 ≤ 100ms（research R2）
- [~] T054 **(deferred — manual)** 跑 5 次新系统 cycle 测 prompt token；对比 T004 baseline 验证 SC-001 ≥ 30% 下降。需要 OKX 实盘 / paper credentials；建议合并后由部署方运行。
- [~] T055 **(deferred — manual)** quickstart.md 端到端：需 OKX credentials；建议合并后由部署方运行。SC-002（`git ls-files agent_skills/ | wc -l == 5` ≤ 6）✓ 已验证（5 SKILL.md，无 .gitkeep）。SC-005 由 `tests/test_skills_curation.py` 单元测试覆盖。
- [x] T056 跑 `uv run pytest --no-cov` 全套件，要求 100% pass + 新增测试 ≥ 25（SC-008）
- [~] T057 **(deferred — manual)** 干净 PostgreSQL 启动 cycle 验证无种子需求（SC-007）。需要 staging DB；T050 grep 已验证 SC-007 代码层面零 GSSC 引用。
- [x] T058 编写 `docs/SC-LONG-TERM-VERIFICATION.md` 记录 SC-003 / SC-004 / SC-006 / SC-010 的 14 天后线上观察方法 + 查询命令；SC-010 单元测试 fault-injection 已在 `tests/test_reflection_pattern_distill.py` 覆盖

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**：无依赖，立即开始
- **Phase 2 Foundational**：依赖 Phase 1 完成；阻塞所有 user story
- **Phase 3 (US1) / Phase 4 (US2) / Phase 5 (US3) / Phase 6 (US4)**：均依赖 Phase 2；4 个 P1 stories 在 Phase 2 完成后**可并行展开**（不同文件、独立测试）
- **Phase 7 (US5)**：依赖 US2（PnL track 接入）+ US4（middleware/verdict 改造）
- **Phase 8 (US6)**：依赖 US2（新 memory 路径已就位才能拆 GSSC）+ US4（middleware 已替代 GSSC pipeline）
- **Phase 9 Polish**：依赖所有 user story 完成

### User Story Dependencies

- **US1（双层目录）**：仅依赖 Phase 2；产出物（initial 5 SKILL.md + .gitkeep）独立可验证
- **US2（memory 写入）**：依赖 Phase 2；为 US5（PnL 归因）+ US6（拆 GSSC）做铺垫
- **US3（skills 层 / curation / propose-new）**：依赖 Phase 2 + US1（5 SKILL.md 文件存在）；与 US2 / US4 可并行
- **US4（middleware）**：依赖 Phase 2 + US1（SKILL.md 存在）+ US3（loader）；为 US6 拆 GSSC 做铺垫
- **US5（verdict 归因）**：依赖 US2（PnL track 接口）+ US4（verdict 节点已切到新 prompt 路径）
- **US6（旧系统全删）**：依赖 US2 + US4 都完成（新轨道完全替代）

### Within Each User Story

- 测试必须先 FAIL 再实现（spec 显式要求 TDD）
- Schema → I/O 工具 → 业务逻辑 → 节点接入 → CLI
- 所有 [P] 标记任务为不同文件 / 无相互依赖，可并行

### Parallel Opportunities

**Phase 1 内**：T003、T004 并行
**Phase 2 内**：T006、T007、T008 并行（T005 必须先完成因为 schema 是基础）
**Phase 4 (US2) 测试编写**：T012、T013、T014 三测试文件并行
**US2 / US3 / US4 跨 story**：在 Phase 2 完成后 4 个 P1 stories 各自并行（独立子模块）
**Phase 9 polish**：T051、T052、T053 并行

---

## Parallel Example: Phase 4 US2 测试 + Phase 5 US3 测试 + Phase 6 US4 测试

```bash
# Phase 2 完成后，3 个 P1 story 测试编写阶段全部并行：
Task: "tests/test_agent_memory_writer.py ≥6 测试" (T012)
Task: "tests/test_reflection_pattern_distill.py ≥6 测试" (T013)
Task: "tests/test_anti_overfitting_equivalence.py ≥4 测试" (T014)
Task: "tests/test_skills_loader.py ≥4 测试" (T024)
Task: "tests/test_skill_proposal.py ≥3 测试" (T025)
Task: "tests/test_skills_curation.py ≥3 测试" (T026)
Task: "tests/test_skills_middleware.py ≥5 测试" (T031)
Task: "tests/test_load_skill_tool.py ≥4 测试" (T032)
```

---

## Implementation Strategy

### MVP First（US1 + US2 + US3 + US4）

1. 完成 Phase 1 Setup（含 baseline 测量 T004，绑死 SC-001 对照）
2. 完成 Phase 2 Foundational（schema + 工具就位）
3. 并行展开 4 个 P1 stories：US1 → US2 → US3 → US4
4. **STOP and VALIDATE**：跑 quickstart.md → 4 agent prompt 含 SKILL.md body + cases 写入 + load_skill tool 可调
5. 部署 / demo

### Incremental Delivery

1. Setup + Foundational（共享基础）
2. + US1 + US3 → 5 SKILL.md 进 git，CLI curation 可用（演示）
3. + US2 → cycle 写 cases、reflection 蒸馏 patterns（数据流转通）
4. + US4 → middleware 注入到 agent prompt（核心闭环）
5. + US5 → PnL 精确归因
6. + US6 → 旧系统拆除（清债）
7. + Polish → 文档 + 性能验证 + 全套件

### Parallel Team Strategy

完成 Setup + Foundational 后：
- Dev A：US1（目录骨架、initial SKILL.md 文件）+ US3（curation/propose-new CLI）
- Dev B：US2（memory.py + reflection 节点）
- Dev C：US4（middleware + load_skill tool）
- Convene：US5、US6 顺序串接

---

## Notes

- [P] = 不同文件、无未完成依赖；可并行执行
- [Story] 标签实现 spec 用户故事 → 任务的可追溯性
- 测试任务 ⚠️ FAIL → 实现 → GREEN（spec 显式 TDD）
- 每完成一组任务 commit；不留半完成实现
- Checkpoint 处可独立验证 user story
- US1-US4 = P1 = MVP（必做）；US5、US6 = P2（PnL 精确化 + 清债）
- Phase 9 polish 中 T054 是 SC-001 ≥ 30% token 下降的唯一硬验证点，**不可跳过**
- 总任务数：58（Phase 1: 4、Phase 2: 4、Phase 3 US1: 3、Phase 4 US2: 12、Phase 5 US3: 7、Phase 6 US4: 7、Phase 7 US5: 4、Phase 8 US6: 9、Phase 9 polish: 8）
