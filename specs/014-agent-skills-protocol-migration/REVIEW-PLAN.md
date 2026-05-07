# Review Guide: Agent Skills 协议迁移（双层架构 v3）

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-05-06

---

## What This Spec Does

把现在 trading bot 用的 DB-backed `ExperienceMemory` 系统（GSSC pipeline + `decision_commits.experience_json` 列）整体替换为基于文件的双层架构：底层 `agent_memory/`（gitignored，永久保留 cases + 蒸馏出的 patterns，用于反思与离线分析），顶层 `agent_skills/`（git 跟踪，遵循 Anthropic Skills 协议的少量高层能力包，agent 真正读取的产物）。注入路径换成 LangChain `AgentMiddleware.wrap_model_call`，删除整个 GSSC pipeline。

**In scope:**
- 双层目录 + `.gitignore` 规则（[FR-001 ~ FR-005](spec.md#functional-requirements)）
- Memory 层数据流：cycle → cases，reflection → patterns，4 层防过拟合（[FR-006 ~ FR-013](spec.md#memory-层cases--patterns)）
- Skills 层 + curation + propose-new CLI + 动态发现（[FR-014 ~ FR-017a](spec.md#skills-层initial-5-个-skillmd运行时动态发现)）
- `SkillsInjectionMiddleware` + `load_skill` tool 双接口（[FR-018 ~ FR-025a](spec.md#加载与注入)）
- Verdict `applied:` 归因约定（[FR-026 ~ FR-028](spec.md#verdict-显式归因)）
- 旧 GSSC / `experience_json` 列 / `arena experience` CLI 全删（[FR-029 ~ FR-033](spec.md#删除清单)）

**Out of scope（明确写在 spec 里）:**
- 旧 `experience_json` 数据 → memory 文件迁移（用户决定冷启动）
- A/B 双轨、向量检索、OpenViking server、retention 自动清理、跨机器同步、前端展示 cases / patterns、cron 化 curation、skill auto-merge / split。详见 [Out of Scope](spec.md#out-of-scope)

## Bigger Picture

这是 trading bot 经验记忆系统的**第二次大重写**。第一次是 2026-03-12 把 `success_patterns/forbidden_zones/strategic_insights` 三分法升级到 FactorMiner-style regime-aware 蒸馏（见 MEMORY.md "Experience Memory Restructure"）；那次仍是 DB-backed，且每条经验当一个 skill 文件存（240+ 文件 / git 噪声大）。这次方向反过来——把 skill 数量收敛到 high-level 能力包（initial 5），把每条经验降回数据层（patterns/cases 文件，gitignored）。

**为什么是现在做**：(a) Anthropic 官方 Skills 协议在 2026-Q1 渐成事实标准，本仓库 `.claude/skills/speckit-*` 已对齐该协议；(b) LangChain 1.2 的 `AgentMiddleware` API 比之前手工拼 prompt 干净很多；(c) 用户在 brainstorm 阶段提了 `OpenViking` 路线但深入分析后明确"过度集成 不要"——本 spec 是不引入外部 server 的本地化版本。

**关联系统**：4 个 analysis agent（[nodes/agents.py](../../src/cryptotrader/nodes/agents.py)）+ verdict 节点 + reflection 节点 + journal 节点都被改造。`learning/regime.py` 保留不动，是这次的"稳定锚点"。`learning/context.py`（GSSC）+ `learning/reflect.py`（DB upsert）整体删除。

如果 Anthropic Skills 协议未来加 `argument-hint` 等扩展字段，本设计的 `additionalProperties: true` 已经留好空间。

---

## Spec Review Guide (30 minutes)

### 理解整体路线（8 min）

读 [spec.md User Story 1](spec.md#user-story-1--双层架构memory-与-skills-解耦priority-p1) 与 [User Story 3](spec.md#user-story-3--skills-层5-个高层能力包anthropic-协议priority-p1)。**Memory vs Skill 的边界**是这次设计的核心——读完两段后请回答：

- 一条 active pattern 进 SKILL.md 后，原 `agent_memory/<agent>/patterns/<name>.md` 是否仍保留？（spec 默认是的——curation 是"复制 + 摘要"，不是"搬家"）这套冗余对你来说是**特性**还是**复杂度**？
- Curation 触发频率是"每周或按需手工"（[research R1](research.md#r1reflection-触发-vs-curation-触发双层架构修订)），意味着 SKILL.md 与 patterns 的内容会**长期不一致**。这种不一致对运维心智负担如何？是否需要一个 `arena skills check-drift` 命令？（spec Out of Scope 明确不做，需要你确认）

### 关键决策需要你拍板（12 min）

**Decision 1：case 文件粒度——per-cycle 单文件**（[FR-006](spec.md#fr-006)）

review v3 的关键修复（CRITICAL-3）。原稿是"每个 agent 独立一份 cases/<cycle_id>.md"——4 个 agent / cycle = 4 文件，verdict_reasoning 重复存。改为 `agent_memory/cases/<cycle_id>.md` 单文件，4 agent analyses 各占一段，verdict 只一份。
- 问题：reflection 蒸馏 patterns 时按 `<agent>::<pattern>` 前缀分发到 `agent_memory/<agent>/patterns/`（[FR-008](spec.md#fr-008)）。如果将来想做"跨 agent 同 pattern"（如 tech 的 funding_squeeze 跟 macro 的相同名 pattern 共享 PnL），per-cycle 单文件路径还能 scale 吗？
- 备选：引入 `cases/<cycle_id>/{tech,chain,news,macro}.md` 子目录（保留扩展性）但增加 inode。值得吗？

**Decision 2：scope 字段 = `shared` | `agent:<id>`**（[FR-004a](spec.md#fr-004a)）

替代原来的硬编码 `SKILL_NAME_BY_AGENT` mapping。middleware 通过 frontmatter 动态发现（[FR-004b](spec.md#fr-004b)）。
- 问题：如果一个 skill 应该注入给"tech + chain"两个但不是 shared，spec 没规则。需要支持 `scope: agent:tech,chain` 列表形式吗？还是说"两个 agent 用 = 提升为 shared"是合理的强约束？
- 用户故事 [US3 Acceptance Scenario 2](spec.md#user-story-3--skills-层5-个高层能力包anthropic-协议priority-p1) 例举了"用户手工创建 momentum-trader/SKILL.md scope: agent:tech 后下个 cycle 自动 pickup"。检查 [tasks.md T024 / T027](tasks.md#user-story-3---skills-层5-skillmd-动态发现--curation--propose-newpriority-p1) 的实现是否真的能做到"无需重启"。

**Decision 3：4 层防过拟合（删 L5 时间衰减）**（[FR-010](spec.md#fr-010)）

用户在 brainstorm 阶段明确 "L5 — 时间衰减不需要"。
- 问题：crypto 行情 regime 切换快，2024-2025 的 funding pattern 在 2026 仍可能完全无效。**没有时间衰减意味着旧 pattern 永远不会自然 demote**——只能靠 maturity FSM 的 `win_rate < 0.40 OR cases ≥ 30 且差距收敛` 触发 deprecated。这是不是太被动？是否需要一个"manual archive after Q"的 CLI 留个口子？
- 反过来说，reflection 默认每 N cycles 重算 win_rate，旧样本会被新样本稀释——某种程度上等价于"有限时间窗滑动"。这个等价是否充分？请看 [tests/test_anti_overfitting_equivalence.py 设计 (T014)](tasks.md#tests-for-user-story-2-) 是否覆盖此论点。

**Decision 4：threading.Lock vs fcntl.flock**（[FR-013](spec.md#fr-013)）

review v3 修复（CRITICAL-4）。原稿写 `fcntl.flock`，与 [research R4](research.md#r4文件锁简化) 的 `threading.Lock` 矛盾。spec 改 threading.Lock 因为本期单进程 single-writer。
- 问题：scheduler 跑在独立 process（参见 docker-compose.yml `scheduler` service）。如果 user 在 scheduler 跑期间手动 `arena skills curate`，会不会有跨进程写竞争？或者 curation CLI 应当强制要求 scheduler 先停？spec 没说。

**Decision 5：`load_skill` rate-limit 10/cycle**（[FR-025](spec.md#fr-025)）

任意硬编码上限。
- 问题：multi-turn debate 流程（[debate/convergence.py](../../src/cryptotrader/debate/convergence.py)）一个 cycle 可能跨 4 + 8 = 12 LLM 轮。如果某 round 每个 agent 都调一次 `load_skill`，会不会撞到 10 上限？spec 默认每 trace_id（cycle）共用一个计数器——这个聚合粒度对吗？

### 不太确定的地方（5 min）

- **[FR-036](spec.md#fr-036)** "仅 drop `decision_commits.experience_json` 一列"。我把 alembic migration 拆到 [T046](tasks.md#user-story-6---旧系统全删priority-p2) 单独一步，但没验证 `experience_json` 列上是否有 index / FK / 业务依赖。请运行 `\d decision_commits` 和 `grep -rn experience_json src/` 确认无遗漏。
- **[research R10](research.md#r10trading-knowledge-skillmd-的内容来源plan-阶段新增)** 说 trading-knowledge SKILL.md "纯手工写"。tasks.md [T010](tasks.md#user-story-1---双层架构解耦priority-p1) 把它当作初始化的一部分要求——但**手写内容质量**是否需要单独 review pass？目前 spec 没要求。
- **SC-001 baseline measurement**（[T004](tasks.md#phase-1-setup共享基础设施)）要求实施前测 5 cycles 并记录到 `baseline-tokens.md`。如果用户在 [T054](tasks.md#phase-9-polish--cross-cutting-concerns) 验证时发现下降只有 25%（< 30% 阈值），spec 没说该 fail merge 还是放宽阈值——这是"硬质量门"还是"目标"？

### Risks and open questions（5 min）

- **跨 agent applied: 解析的歧义**（[FR-026](spec.md#fr-026) 第三点）：bare name 在多 agent 同名时记 warning 跳过该次 PnL 归因。如果某个真实场景下两个 agent 都有 `funding_squeeze` pattern 名（高概率发生——LLM 命名收敛），这个 warning 是否会**累积大量丢失归因**？是否应当在 reflection 检测到首次冲突就直接强制重命名其中一个？
- **middleware mtime 缓存的失效粒度**（[FR-019a](spec.md#fr-019a)）：每次访问对比 mtime 一遍。如果 `agent_skills/` 长期增长到几十个 skill，每个 cycle 4 agent × 几十个 stat() 是否成为热路径？[research R2](research.md#r2性能-sc) 假设 ≤ 50ms 但前提是 1-2 个文件——动态扩展后是否需要 inotify 推模型？
- **propose-new 的"找跨域共性"算法**（[FR-016a](spec.md#fr-016a)）：spec 只说"分析共同 regime/theme"，没说算法（聚类？LLM 提议？相似度？）。tasks.md [T029](tasks.md#user-story-3---skills-层5-skillmd-动态发现--curation--propose-newpriority-p1) 也没说。这是隐含的 follow-up 还是 v1 必做？
- **删除 `learning/reflect.py` 的连锁影响**（[T043](tasks.md#user-story-6---旧系统全删priority-p2)）：grep 一下还有谁 import `learning.reflect`。如果 verbal_reinforcement 之外还有调用方（例如 backtest 路径），删除前需要替代 wiring。

---

*Full context in linked [spec](spec.md), [plan](plan.md), [research](research.md), and [data-model](data-model.md). Per-task spec/research citation see [tasks.md FR/SC links](tasks.md).*
