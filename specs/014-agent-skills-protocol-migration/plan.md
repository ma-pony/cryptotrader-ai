# Implementation Plan: Agent Skills 协议迁移（双层架构 v2）

**Branch**: `014-agent-skills-protocol-migration` | **Date**: 2026-05-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/014-agent-skills-protocol-migration/spec.md`

## Summary

双层架构实现：
- **Memory 层**（`agent_memory/`，gitignored）：每 cycle 写 cases，reflection 蒸馏 patterns，永久保留用于分析
- **Skills 层**（`agent_skills/`，git 跟踪，5 个 SKILL.md）：从 memory 整理出的高层能力包，遵循 Anthropic Skills 协议；agent 唯一直接读取的产物
- **整理流程**（curation）：手工 + LLM 触发，把 active patterns 整理进 SKILL.md（与 reflection 解耦，频率较低）
- LangChain `SkillsInjectionMiddleware` 把 5 个 SKILL.md 注入对应 agent prompt + 注册 `load_skill(name)` tool
- 旧 GSSC pipeline / `ExperienceMemory` dataclass / `decision_commits.experience_json` 列 / `arena experience` CLI / 4 GSSC 测试文件全删

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**:
- LangChain `>=1.2.10`、`langchain-core >=1.2.17`（已依，提供 `create_agent` + `AgentMiddleware`）
- `langgraph >=1.0.10`
- `pyyaml`（frontmatter 解析）

**Storage**:
- 文件系统：
  - `agent_memory/`（gitignored，永久保留）—— cases + patterns + archive
  - `agent_skills/`（git 跟踪）—— 5 个 SKILL.md
- PostgreSQL：drop `decision_commits.experience_json` 列；其他表不动

**Testing**: pytest 9.x（`uv run pytest --no-cov`），asyncio_mode=auto，新增 ≥ 25 测试

**Target Platform**: macOS（开发）+ Linux container（部署）

**Project Type**: 单后端 Python 项目（cryptotrader-ai 主仓）

**Performance Goals**:
- prompt token 较旧系统下降 ≥ 30%（spec SC-001）
- `load_agent_skills(agent)` p95 ≤ 50ms（仅 1-2 个 SKILL.md 文件读取）
- cycle 写 case：p95 ≤ 100ms

**Constraints**:
- reflection 失败 NOT 阻塞 cycle（FR-012）
- cycle 写 case 失败 NOT 阻塞 cycle 主流程（FR-007）
- 不引入新 runtime 依赖（FR-034）

**Scale/Scope**:
- `agent_memory/` 长期：~50 cycles/天 × 365 天 × 4 agents = ~73K case 文件 / 年；patterns ~100 条 / agent
- `agent_skills/` 始终：5 个 SKILL.md，总 ≤ 50KB
- middleware 加载：每 cycle 4 个 agent × 2 个 SKILL.md（own + trading-knowledge）= 8 次文件读

## Constitution Check

`.specify/memory/constitution.md` 仍是模板未填写。**跳过 constitution gate**。

## Project Structure

### Documentation

```text
specs/014-agent-skills-protocol-migration/
├── spec.md                         # ✅ v2 双层架构
├── plan.md                         # 本文件
├── research.md                     # 已有，需更新双层架构相关 R 项
├── data-model.md                   # 重写（Skill / PatternRecord / CaseRecord 三个核心实体）
├── contracts/
│   ├── skill_md.schema.yaml        # SKILL.md frontmatter (5 个文件用)
│   ├── pattern_record.schema.yaml  # patterns/*.md frontmatter (memory 层数据)
│   ├── case_record.schema.yaml     # cases/*.md frontmatter (memory 层数据)
│   ├── load_skill.contract.md      # tool I/O
│   └── middleware.contract.md      # SkillsInjectionMiddleware
├── quickstart.md                   # 更新
├── checklists/requirements.md      # 已有
├── REVIEW-SPEC.md                  # 已有
└── tasks.md                        # /speckit-tasks 输出
```

### Source Code

```text
cryptotrader-ai/
├── .gitignore                                   # 新增 agent_memory/ 条目
│
├── agent_memory/                                # ⭐ 顶级目录（gitignored）
│   ├── tech/{cases,patterns,archive}/
│   ├── chain/{cases,patterns,archive}/
│   ├── news/{cases,patterns,archive}/
│   └── macro/{cases,patterns,archive}/
│
├── agent_skills/                                # ⭐ 顶级目录（git 跟踪）
│   ├── tech-analysis/SKILL.md                   # initial 手工写
│   ├── chain-analysis/SKILL.md
│   ├── news-analysis/SKILL.md
│   ├── macro-analysis/SKILL.md
│   └── trading-knowledge/SKILL.md               # 替代之前 shared/ 的 funding_rate / regime_def 等
│
├── src/cryptotrader/
│   ├── agents/
│   │   ├── base.py                              # 改：ToolAgent.create_agent 注册 SkillsInjectionMiddleware
│   │   ├── skills/                              # ⭐ 新增子模块
│   │   │   ├── __init__.py
│   │   │   ├── loader.py                        # load_skill, parse_skill_md (~80 行)
│   │   │   ├── middleware.py                    # SkillsInjectionMiddleware (~100 行)
│   │   │   ├── tool.py                          # load_skill tool 定义 (~50 行)
│   │   │   └── schema.py                        # Skill / PatternRecord / CaseRecord dataclass (~80 行)
│   │   ├── tech.py / chain.py / news.py / macro.py     # 改：删除 prompt 字符串（搬到 SKILL.md）
│   ├── learning/
│   │   ├── memory.py                            # ⭐ 新增 — case 写入 + patterns 蒸馏 + archive (~250 行)
│   │   │                                        # 含 4 层防过拟合算法（迁自 reflect.py）
│   │   ├── curation.py                          # ⭐ 新增 — SKILL.md 整理（手工 + LLM 触发） (~120 行)
│   │   ├── regime.py                            # 保留不动
│   │   ├── reflect.py                           # ❌ 删除（DB upsert 路径）
│   │   ├── context.py                           # ❌ 删除（GSSC pipeline 整文件）
│   │   └── verbal.py                            # 改：删除 search_by_regime 等 DB 检索
│   ├── nodes/
│   │   ├── agents.py                            # 改：4 个 agent 节点改用 middleware 注入
│   │   ├── data.py                              # 改：删 verbal_reinforcement 中 experience injection
│   │   ├── reflection.py                        # ⭐ 新增 — 包装 learning/memory 反思流程为 graph 节点
│   │   ├── journal.py                           # 改：新增 case 写入到 agent_memory/<agent>/cases/
│   │   └── verdict.py                           # 改：prompt 加 applied: 格式要求
│   ├── journal/
│   │   └── store.py                             # 改：drop experience_json 列（auto-migration）
│   ├── models.py                                # 改：删除 ExperienceMemory / ExperienceRule
│   └── cli/main.py                              # 改：移除 arena experience；新增 arena skills curate
│
└── tests/
    ├── test_agent_memory_writer.py              # ⭐ ≥6 测试 (case 写入)
    ├── test_skills_loader.py                    # ⭐ ≥4 测试 (SKILL.md 加载)
    ├── test_skills_middleware.py                # ⭐ ≥5 测试 (middleware 注入)
    ├── test_load_skill_tool.py                  # ⭐ ≥4 测试 (tool I/O)
    ├── test_reflection_pattern_distill.py       # ⭐ ≥6 测试 (memory 蒸馏)
    ├── test_anti_overfitting_equivalence.py     # ⭐ ≥4 测试 (4 层等价)
    ├── test_skills_curation.py                  # ⭐ ≥3 测试 (CLI 整理流程)
    ├── test_experience_extraction.py            # ❌ 删除
    ├── test_factorminer_*.py / test_gssc_*.py   # ❌ 删除
    └── 其他保留
```

**Structure Decision**:
- 两个顶级目录与 `src/`、`config/` 同级；明确分层（memory 数据 / skills 能力包）
- `agent_skills/` 子模块在 `src/cryptotrader/agents/skills/`：loader / middleware / tool / schema 四模块切关切
- learning 层重组：`memory.py`（cases + patterns）+ `curation.py`（SKILL.md 整理）替代旧 `context.py` + `reflect.py`
- nodes 层新增 `reflection.py`，作为 graph 中调用 `learning/memory.py:run_reflection` 的薄包装

## Phase 0: Outline & Research（更新）

之前 R1-R6 决策大部分仍有效，**双层架构修订需更新**：

### R1（更新）：Reflection 触发位置 vs Curation 触发位置
- **Reflection（memory 层蒸馏）**：在 graph 中作为 `nodes/reflection.py`，由 `nodes/data.py:verbal_reinforcement` 按 `every_n_cycles` 触发——与 cycle 同进程、单写者最简
- **Curation（SKILL.md 整理）**：与 cycle 解耦——通过 CLI `arena skills curate <name>` 手工触发；本期不上 cron。Phase 2 follow-up 可加 `--all` 批量自动化

### R2（不变）：性能 SC
- `load_agent_skills(agent)` p95 ≤ 50ms（仅 1-2 个 SKILL.md 文件读，比之前估算更快）
- cycle 写 case p95 ≤ 100ms
- cycle 总耗时较基线增量 ≤ 5%

### R3（重新简化）：命名消歧
双层架构下命名空间天然清晰：
- 5 个 skill name：`tech-analysis` / `chain-analysis` / `news-analysis` / `macro-analysis` / `trading-knowledge`，全局唯一
- patterns 内部命名：`<agent>/patterns/<name>.md`，路径隔离
- `load_skill(name)` 只接 5 个 skill name 之一，无歧义
- verdict reasoning `applied: <pattern_name>`：在该 agent 的 patterns/ 内查找；跨 agent 引用用 `applied: <agent>::<pattern_name>` 形式（FR-026）

### R4（更新）：文件锁
- 进程内 `threading.Lock` 即可——本期 reflection / curation 都在同进程
- 写文件用临时文件 + `os.rename` 保证原子
- 跨进程锁（fcntl.flock）作为 follow-up，等 curation 走独立 cron 时再加

### R5（不变）：启动时间基线，实施前测 5 次

### R6（简化）：load_skill 双接口
- LangChain `BaseTool` + 普通 Python 函数双导出（同实现）
- 4 个 agent 节点通过 middleware 自动获得 tool 调用能力
- verdict / curation 直接调函数

### R7（新）：Curation 的 LLM 整理
- `arena skills curate <name> --llm` 调用 LLM 读 active patterns + 当前 SKILL.md → 输出新 SKILL.md 草稿
- 不直接 overwrite——输出到 stdout 或新文件 `SKILL.md.draft`，用户 diff 后手工 merge
- 本 phase 实现 CLI 框架 + 简单 LLM 调用；prompt 模板优化留 follow-up

### R8（新）：cases 文件命名与内容
- 命名：`agent_memory/<agent>/cases/<YYYY-MM-DD>-cycle-<commit_hash[:8]>.md`
- frontmatter：cycle_id / timestamp / pair / verdict_action / final_pnl
- body：snapshot summary（YAML） + agent_analysis（markdown）+ verdict reasoning + applied_patterns 列表

## Phase 1: Design & Contracts

### 1.1 data-model.md（更新）

**3 个核心实体**：

```python
@dataclass
class Skill:
    """高层能力包，1 个 SKILL.md，5 个之一"""
    name: str                    # tech-analysis / chain-analysis / etc
    description: str             # frontmatter
    body: str                    # markdown content
    file_path: Path
    manually_edited: bool

@dataclass
class PatternRecord:
    """memory 层数据，agent_memory/<agent>/patterns/<name>.md"""
    name: str
    agent: str                   # tech / chain / news / macro
    description: str
    body: str
    regime_tags: list[str]
    pnl_track: PnLTrack          # cases / win_rate / avg_pnl / last_active
    maturity: Literal["observed", "probationary", "active", "deprecated"]
    source_cycles: list[str]     # cycle_ids that contributed
    created: datetime
    file_path: Path

@dataclass
class CaseRecord:
    """memory 层数据，agent_memory/<agent>/cases/<cycle_id>.md，永久保留"""
    cycle_id: str
    timestamp: datetime
    pair: str
    agent: str
    snapshot_summary: dict
    agent_analysis: str
    verdict_action: Literal["long", "short", "hold", "close"]
    verdict_reasoning: str
    applied_patterns: list[str]  # 从 verdict.reasoning 解析
    risk_gate_passed: bool
    execution_status: dict | None
    final_pnl: float | None      # 平仓后回填
    file_path: Path
```

**辅助实体**：`AgentSkillSet`（middleware 加载结果）、`ReflectionRun`（反思日志）、`CurationRun`（整理日志）

### 1.2 contracts/

**3 个 schema 文件**：

- `skill_md.schema.yaml`：SKILL.md frontmatter（极简，遵循 Anthropic 协议）
  ```yaml
  required: [name, description]
  properties:
    name: { pattern: "^[a-z-]+$" }
    description: { minLength: 30, maxLength: 500 }
    manually_edited: { type: boolean, default: false }
    version: { default: "1.0" }
  ```

- `pattern_record.schema.yaml`：memory/patterns 的 frontmatter（含 regime / pnl_track / maturity 等）

- `case_record.schema.yaml`：memory/cases 的 frontmatter（cycle_id / pair / verdict / final_pnl 等）

**2 个 contract**：
- `load_skill.contract.md`：单参数（5 个 skill name 之一）+ 4 类 error
- `middleware.contract.md`：`wrap_model_call` + `tools = [load_skill]` 注册

### 1.3 quickstart.md（更新）

```bash
# Phase 1 验证
uv run pytest tests/test_agent_memory_writer.py tests/test_skills_loader.py tests/test_skills_middleware.py -v

# 跑 cycle 看 memory 写入
uv run arena run --pair BTC/USDT --mode paper
ls agent_memory/tech/cases/   # 应有新文件

# 跑 reflection 看 pattern 蒸馏
uv run arena reflect --commits-since=24h
ls agent_memory/tech/patterns/

# 整理 SKILL.md（手工 review）
uv run arena skills curate tech-analysis --llm
diff agent_skills/tech-analysis/SKILL.md.draft agent_skills/tech-analysis/SKILL.md
mv agent_skills/tech-analysis/SKILL.md.draft agent_skills/tech-analysis/SKILL.md
```

### 1.4 Agent context update

跑 `update-agent-context.sh claude` → 立即 `git checkout -- CLAUDE.md` 还原。

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| 两个顶级目录（agent_memory + agent_skills） | 数据/能力分离；memory gitignored，skills 进 git | 单顶级目录嵌套（`agent_skills/<skill>/memory/`）会让 gitignore 规则复杂、git diff 时混淆 |
| `learning/memory.py` + `learning/curation.py` 拆两文件 | reflection（频繁、自动）vs curation（慢、手工/LLM）频率与触发完全不同 | 单文件可行但难以独立测试 |
| `nodes/reflection.py` 独立节点（vs 嵌入 nodes/data.py） | 独立节点有自己的 logging / metrics / 失败隔离 | 嵌入 nodes/data 会混淆 graph 拓扑 |
| `arena skills curate` 新 CLI 命令 | 用户需要手工触发整理（user 强调"也可以自定义"） | 自动 cron-only 流程剥夺用户控制权 |
