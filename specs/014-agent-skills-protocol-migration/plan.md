# Implementation Plan: Agent Skills 协议迁移

**Branch**: `014-agent-skills-protocol-migration` | **Date**: 2026-05-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/014-agent-skills-protocol-migration/spec.md`

## Summary

把当前 DB-backed `ExperienceMemory` 系统替换为基于文件的 Anthropic Skills 协议实现。
4 个 trading agent 节点（tech / chain / news / macro）通过自定义 `SkillsInjectionMiddleware`（继承 LangChain `AgentMiddleware`）在 LLM 调用前自动加载并 regime-过滤注入对应 agent 的 skill descriptions；同时注册 `load_skill(name)` tool 让 agent 按需拉取完整 body。Reflection job 改为单写者文件读写，保留 4 层防过拟合算法 + PnL-based maturity。旧 GSSC pipeline、`ExperienceMemory` / `ExperienceRule` dataclass、`decision_commits.experience_json` DB 列、`arena experience` CLI 子命令、4 个 GSSC 测试文件**全部删除**——无 fallback 无迁移，新系统从零冷启动。

## Technical Context

**Language/Version**: Python 3.12（项目当前已用，pyproject.toml 锁 `>=3.10`，实际开发 3.12）
**Primary Dependencies**:
- LangChain `>=1.2.10` + `langchain-core >=1.2.17`（已在 pyproject.toml；提供 `create_agent` + `AgentMiddleware`）
- `langgraph >=1.0.10`（已用于 graph 编排）
- `pyyaml`（frontmatter 解析；项目已传依，需确认版本）

**Storage**:
- 文件系统：`agent_skills/` 顶级目录（git 跟踪）
- PostgreSQL：保留所有现有表，仅 drop `decision_commits.experience_json` 列

**Testing**: pytest 9.x（`uv run pytest --no-cov`，套件当前 ~2003 测试），asyncio_mode=auto，新增 ≥ 20 测试

**Target Platform**: macOS（开发）+ Linux container（部署）；scheduler 由 launchd / systemd 管理

**Project Type**: 单后端 Python 项目（cryptotrader-ai 主仓），含独立 web 前端（不涉及本 feature）

**Performance Goals**:
- prompt token 较旧系统下降 ≥ 30%（spec SC-001）
- `load_agent_skills(agent, regime)` p95 ≤ 100ms（gap REVIEW-5 解决）
- `load_skill(name)` 单次调用 p95 ≤ 50ms（文件 IO + YAML parse）

**Constraints**:
- reflection 失败 NOT 阻塞 trading cycle（FR-018）
- 不引入新 runtime 依赖（FR-028）—— 不允许 vector DB / Rust / OpenViking server
- 兼容现有 `create_agent` 调用：`agents/base.py:ToolAgent` 已用其 `tools=` + `system_prompt=` 参数
- 单写者：reflection job 写文件时 4 个 agent 节点不能并发改文件（用 fcntl.flock 实现）

**Scale/Scope**:
- 4 个 agent × ~30 patterns × 2 (patterns + forbidden) ≈ 240 文件长期上限
- shared/ ~10 个文件
- 单 cycle 4 个 agent × `load_agent_skills` 调用 = 4 次目录扫描 + ~50 文件 parse / cycle
- `load_skill` 调用频率：未知（agent 自主决定），spec rate-limit ≤ 10 次/cycle

## Constitution Check

`.specify/memory/constitution.md` 仍是模板未填写状态。**跳过 constitution gate**——本 feature 不引入与项目根本性原则冲突的设计（不破坏既有架构，仅替换一个子系统）。

## Project Structure

### Documentation (this feature)

```text
specs/014-agent-skills-protocol-migration/
├── spec.md                         # ✅ 已就绪
├── plan.md                         # 本文件
├── research.md                     # Phase 0 输出（resolve REVIEW-2/5/6 + AMBI-1/2）
├── data-model.md                   # Phase 1 输出（Skill / AgentSkillSet / Frontmatter schema）
├── contracts/                      # Phase 1 输出
│   ├── skill_frontmatter.schema.yaml
│   ├── load_skill.contract.md      # tool 输入输出契约
│   └── middleware.contract.md      # SkillsInjectionMiddleware 接口
├── quickstart.md                   # Phase 1 输出（开发者 setup steps）
├── checklists/requirements.md      # ✅ 已就绪
├── REVIEW-SPEC.md                  # ✅ 已就绪
└── tasks.md                        # /speckit-tasks 输出（不在本命令范围）
```

### Source Code (repository root)

```text
cryptotrader-ai/
├── agent_skills/                                # ⭐ 新增顶级目录（git 跟踪）
│   ├── tech/
│   │   ├── instructions.md                      # tech 行为约束
│   │   ├── patterns/                            # 蒸馏出的可复用规则（初期空）
│   │   ├── forbidden/                           # 蒸馏出的禁忌（初期空）
│   │   └── archive/                             # deprecated skills 移此（初期空）
│   ├── chain/{instructions.md, patterns/, forbidden/, archive/}
│   ├── news/{instructions.md, patterns/, forbidden/, archive/}
│   ├── macro/{instructions.md, patterns/, forbidden/, archive/}
│   └── shared/                                  # 跨 agent 领域常识
│       ├── funding_rate.md                      # 替代 base.py:35-37 硬编码
│       ├── regime_definitions.md
│       └── trading_pair_semantics.md
│
├── src/cryptotrader/
│   ├── agents/
│   │   ├── base.py                              # 改：ToolAgent 注册 SkillsInjectionMiddleware
│   │   ├── skills/                              # ⭐ 新增子模块
│   │   │   ├── __init__.py
│   │   │   ├── loader.py                        # load_agent_skills, parse_skill (~150 行)
│   │   │   ├── middleware.py                    # SkillsInjectionMiddleware (~80 行)
│   │   │   ├── tool.py                          # load_skill tool 实现 (~60 行)
│   │   │   └── schema.py                        # Skill / AgentSkillSet / FrontmatterV1 dataclass (~60 行)
│   │   ├── tech.py / chain.py / news.py / macro.py     # 改：删除内嵌 prompt 字符串（搬到 instructions.md）
│   │   └── verdict.py（如有）                    # 改：prompt 加 "applied: <skill>" 格式要求
│   ├── learning/
│   │   ├── skills.py                            # ⭐ 新增（替代 reflect.py）
│   │   │                                        #   - 4 层防过拟合算法（迁自 reflect.py）
│   │   │                                        #   - 文件原子写 + atomic rename
│   │   │                                        #   - fcntl.flock 单写者
│   │   ├── regime.py                            # 保留不动
│   │   └── reflect.py                           # ❌ 删除（DB upsert 路径）
│   │   └── context.py                           # ❌ 删除（GSSC pipeline 整文件）
│   │   └── verbal.py                            # 改：删 search_by_regime DB 检索
│   ├── nodes/
│   │   ├── agents.py                            # 改：4 个 agent 节点改用 middleware 注入
│   │   ├── data.py                              # 改：删 verbal_reinforcement 中 experience injection
│   │   └── verdict.py                           # 改：prompt 加 applied: 格式要求
│   ├── journal/
│   │   └── store.py                             # 改：drop experience_json 列（auto-migration）
│   ├── models.py                                # 改：删 ExperienceMemory / ExperienceRule
│   └── cli/main.py                              # 改：移除 arena experience subcommands
│
└── tests/
    ├── test_agent_skills_loader.py              # ⭐ 新增（≥6 测试）
    ├── test_skills_middleware.py                # ⭐ 新增（≥4 测试）
    ├── test_skills_reflection.py                # ⭐ 新增（≥6 测试）
    ├── test_skills_anti_overfitting.py          # ⭐ 新增（≥4 测试，等价性验证）
    ├── test_load_skill_tool.py                  # ⭐ 新增（≥4 测试，spec User Story 2 场景 5-8）
    ├── test_experience_extraction.py            # ❌ 删除（GSSC 测试）
    ├── test_factorminer_*.py（如有）             # ❌ 删除
    ├── test_gssc_*.py（如有）                    # ❌ 删除
    └── test_*.py                                # 其他保留
```

**Structure Decision**:

- 新增 `agent_skills/` 顶级目录（git 跟踪，与 `src/`、`config/` 同级），符合 spec FR-001
- 在 `src/cryptotrader/agents/` 下新增 `skills/` 子模块，集中放置 loader / middleware / tool / schema 四个模块；与 `agents/base.py` 同级，便于后者直接 import
- 反思核心搬到 `learning/skills.py`，与 `learning/regime.py` 协作；旧 `learning/reflect.py` 与 `learning/context.py` 整体删除
- 测试不分子目录（项目惯例）；新文件以 `test_skills_*` / `test_load_skill_*` 命名前缀

## Phase 0: Outline & Research

下面 5 个 spec review 推迟项 + 1 个新引入决策，需要在本阶段澄清并写入 `research.md`。

### R1（resolves REVIEW-2）：Reflection job 触发机制

**问题**：当前 `nodes/data.py:verbal_reinforcement` 通过 `[experience] every_n_cycles` 在 graph 中触发反思；删了 GSSC 后 reflection 在哪里跑？

**候选方案**：
- (a) 保留在 graph 里：新建 `nodes/reflection.py` 节点，被 `nodes/data.py` 按 `every_n_cycles` 调用——耦合 trading cycle，简单
- (b) 独立 cron job：launchd / systemd timer 定时跑 `arena reflect` CLI——解耦，但多一层运维
- (c) Scheduler post-cycle hook：scheduler 完成 N 个 cycle 后调 reflection——介于 (a)/(b) 之间

**推荐**：(a) 简单 + 与 cycle 同进程，符合"单写者"语义（不需要跨进程文件锁）。失败仅捕获在该节点不阻塞下游。

### R2（resolves REVIEW-5）：性能基线与 cycle latency SC

**问题**：spec 没有 cycle latency 保证。新加 `load_agent_skills` 调用每 cycle ~4 次磁盘扫描，需估算开销。

**做法**：
- 测一次基线：当前系统 single-cycle wall-clock 中位数（已有 `metrics.cycle_duration_ms`）
- 实现 `load_agent_skills` 后做 microbench：~50 文件读 + YAML parse 单进程 SSD 上目标 < 50ms

**新 SC**（plan.md 自加，不回写 spec）：`load_agent_skills` p95 ≤ 100ms；trading cycle 总耗时较基线增量 ≤ 5%

**缓解策略**：进程内 LRU 缓存（key = `(agent, mtime_max)`），仅在 reflection 写入后失效

### R3（resolves REVIEW-6 + FR-008b）：跨 agent 命名消歧

**问题**：`applied: funding_squeeze_long` 在 tech / chain 都有同名 pattern 时如何 reflection 归因？

**决定**：
- `applied:` 引用：reflection 解析时同名跨 agent 视为**ambiguous**，跳过该引用并 logger.warning（FR-022 已部分覆盖）
- `load_skill(name)` tool：相同规则——同名跨 agent 时返回 ambiguous error（已在 FR-008b + Acceptance Scenario 8 定义）
- 推荐 **agent 在 prompt 里看到的 description 列表自动包含 `agent::` 前缀**——避免 agent 用简短形式

**实现**：middleware 渲染 description 时 force prefix `{agent}::{name}`

### R4（resolves AMBI-1）："短暂时间窗"具体值

**问题**：FR-019 提到 reflection 写入与 agent 读取的时间窗未定。

**决定**：
- 单写者用 `fcntl.flock` 排他锁，无固定 wait window
- agent 节点 read 时同样用 `fcntl.flock` shared 锁
- reflection 等待 shared 锁释放（写入开始）→ 获取 exclusive 锁 → 写入 → 释放
- agent cycle 内连续读多个文件应在循环外**一次性**获取 shared 锁、释放

### R5（resolves AMBI-2）：启动时间基线

**问题**：SC-008 要求"启动时间不变或下降"，但无基线值。

**做法**：迁移前测 5 次"clone → docker compose up → arena migrate → arena run BTC/USDT 完成"全流程时间，取中位数。Phase 1 之后 implementation 完成时再测一次对比。

**预测**：新增 `agent_skills/` 17 个 markdown 文件 + 8 个 .gitkeep 大约 ~50KB；初始化耗时增量 < 1s，完全在噪声内。

### R6（新增决策）：`load_skill` tool 是否传给所有 4 个 agent + verdict

**问题**：spec 说 4 个 agent 节点和 verdict 节点都能调 `load_skill`。但 verdict 节点目前在用 `make_verdict()` 不走 `create_agent`；如何接入 middleware？

**决定**：
- 4 个 analysis agent 通过 `create_agent` 走 middleware
- verdict 节点不接 middleware，但**直接接收**已 load 完的 skill descriptions（passed via state）；必要时通过命令式 `load_skill(name)` 函数调用（同源代码）
- 即 `load_skill` 既是 LangChain tool（agent 用），也是普通 Python 函数（verdict / reflection 用）——双接口设计

**`research.md` 输出格式（Phase 0 完成后）**

```markdown
## R1: Reflection job 触发机制
- Decision: 在 graph 中作为新 node（`nodes/reflection.py`），由 `nodes/data.py:verbal_reinforcement` 按 every_n_cycles 调用
- Rationale: 单进程 + 单写者最简，与现状最少差异
- Alternatives: 独立 cron（增运维）；scheduler hook（介于 a/b，无明显优势）

## R2: 性能基线与新 SC
... (类似格式)

## R3 ~ R6: ...
```

## Phase 1: Design & Contracts

### Phase 1.1: data-model.md（实体设计）

```markdown
## 实体清单

### Skill（运行时 dataclass，对应单个 .md 文件）
- `agent: Literal["tech", "chain", "news", "macro", "shared"]`
- `kind: Literal["pattern", "forbidden", "instruction", "knowledge"]`
- `name: str`（snake_case，文件名）
- `description: str`（1 句话摘要，frontmatter）
- `body: str`（markdown 正文）
- `regime_tags: list[str]`
- `pnl_track: PnLTrack`（cases / win_rate / avg_pnl / last_active）
- `maturity: Literal["observed", "probationary", "active", "deprecated"]`
- `manually_edited: bool`（默认 False）
- `created: datetime`
- `source_commits: list[str]`
- `version: str`（默认 "1.0"）
- `file_path: Path`（运行时反向引用）

### PnLTrack
- `cases: int`
- `win_rate: float`（0-1）
- `avg_pnl: float`（USDT）
- `last_active: datetime | None`

### AgentSkillSet（单 agent 一次 cycle 加载的全集）
- `agent: str`
- `instructions: Skill | None`（单条）
- `patterns: list[Skill]`（已 regime 过滤）
- `forbidden: list[Skill]`（已 regime 过滤）
- `knowledge: list[Skill]`（来自 shared/，不过滤）
- `regime_tags: list[str]`（构造时输入）

### ReflectionRun（一次反思的结构化日志）
- `started_at: datetime`
- `finished_at: datetime | None`
- `commits_window: tuple[str, str]`（first_hash, last_hash）
- `created_skills: list[str]`（file paths）
- `updated_skills: list[str]`
- `archived_skills: list[str]`
- `errors: list[dict]`

### LoadSkillRequest / LoadSkillResponse（tool I/O）
- Request: `name: str`（短或 `agent::` 形式）
- Response: `{"name": str, "agent": str, "body": str}` 或 `{"error": str, ...}`
```

### Phase 1.2: contracts/

#### contracts/skill_frontmatter.schema.yaml

```yaml
$schema: "http://json-schema.org/draft-07/schema#"
title: SkillFrontmatter
type: object
required: [name, agent, description, regime_tags, pnl_track, maturity, version]
properties:
  name: { type: string, pattern: "^[a-z][a-z0-9_]*$" }
  agent: { enum: [tech, chain, news, macro, shared] }
  description: { type: string, minLength: 30, maxLength: 500 }
  regime_tags:
    type: array
    items: { type: string }
  pnl_track:
    type: object
    properties:
      cases: { type: integer, minimum: 0 }
      win_rate: { type: number, minimum: 0, maximum: 1 }
      avg_pnl: { type: number }
      last_active: { type: string, format: date-time, nullable: true }
  maturity: { enum: [observed, probationary, active, deprecated] }
  manually_edited: { type: boolean, default: false }
  created: { type: string, format: date-time }
  source_commits:
    type: array
    items: { type: string, pattern: "^[0-9a-f]{16,64}$" }
  version: { type: string, pattern: "^\\d+\\.\\d+$" }
```

#### contracts/load_skill.contract.md

```markdown
# load_skill Tool Contract

## Tool registration
- Class: `cryptotrader.agents.skills.tool.load_skill`
- Registered via: `SkillsInjectionMiddleware.tools = [load_skill]`
- Type: `langchain.tools.BaseTool` (Pydantic args schema)

## Input
- `name: str` — `<name>` (same agent) or `<agent>::<name>` (cross-agent)

## Output (success)
```json
{"name": "...", "agent": "...", "kind": "pattern", "body": "..."}
```

## Output (errors)
- skill_not_found: `{"error": "skill_not_found", "name": "..."}`
- ambiguous_name: `{"error": "ambiguous_name", "candidates": ["tech::x", "chain::x"]}`
- corrupt_file: `{"error": "corrupt_file", "path": "...", "details": "..."}`
- rate_limit_per_cycle: `{"error": "rate_limit_per_cycle", "limit": 10}`

## Performance
- p95 ≤ 50ms (single file IO + YAML parse)

## Side effects
- Increments per-cycle call counter (state-bound)
- Read-only on `agent_skills/`; never writes
```

#### contracts/middleware.contract.md

```markdown
# SkillsInjectionMiddleware Contract

## Class
```python
class SkillsInjectionMiddleware(AgentMiddleware):
    tools = [load_skill]

    def __init__(self, agent_id: str, skill_dir: Path = Path("agent_skills")):
        ...

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        ...
```

## Behavior
1. Compute current regime_tags from `request.runtime_context["snapshot"]`
2. Call `load_agent_skills(self.agent_id, regime_tags)` to build AgentSkillSet
3. Render to markdown text:
   - `## Instructions\n{instructions.body}\n`
   - `## Available Patterns\n{name}: {description}\n...`
   - `## Forbidden Zones\n{name}: {description}\n...`
   - `## Knowledge\n{shared.description}\n...`
   - `## Loading Rule\nUse load_skill(name) tool to retrieve full body of any item above.`
4. Append to `request.system_message.content_blocks`
5. Forward to `handler(modified_request)`
```

### Phase 1.3: quickstart.md（开发者 setup steps）

```markdown
# Agent Skills Quickstart

## Run Phase 1 implementation locally

```bash
git checkout 014-agent-skills-protocol-migration

# 1. Install deps
uv sync

# 2. Run new tests for the skills layer
uv run pytest tests/test_agent_skills_loader.py tests/test_skills_middleware.py -v

# 3. Run a full cycle (paper mode) to verify middleware injection
uv run arena run --pair BTC/USDT --mode paper

# 4. Inspect injected prompt (debug mode logs system_message)
LOG_LEVEL=DEBUG uv run arena run --pair BTC/USDT --mode paper 2>&1 | grep "system_message"

# 5. Run reflection job manually (one-off CLI)
uv run arena reflect --commits-since=24h
```

## File layout to expect
- `agent_skills/{tech,chain,news,macro}/instructions.md` — agent role text
- `agent_skills/{...}/patterns/` — initially empty
- `agent_skills/shared/{funding_rate,regime_definitions,trading_pair_semantics}.md`
```

### Phase 1.4: Agent context update

跑 `update-agent-context.sh claude` 后**立即恢复 CLAUDE.md**（`git checkout -- CLAUDE.md`）——CLAUDE.md 是用户维护文件，不允许 spec-kit 自动生成段落覆盖。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| 新增 `src/cryptotrader/agents/skills/` 子模块（4 文件 vs 单文件） | loader / middleware / tool / schema 四种关切分离更易测、易扩展 | 单文件 ~350 行可行但难以独立测各 part |
| 双接口（`load_skill` 既是 LangChain tool 又是 Python 函数）| 4 个 analysis agent 用 tool；verdict 不走 create_agent 直接调函数 | 强制 verdict 也走 create_agent 改造范围太大 |
| 用 `fcntl.flock` 而非纯 Python mutex | 跨进程安全（reflection job 可能未来独立 cron）| Python mutex 仅同进程；本期单进程 OK，但 fcntl 不增成本，提前预备 |
