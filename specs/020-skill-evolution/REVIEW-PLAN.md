# Review Guide: Skill Evolution（spec 019）

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-05-09

---

## What This Spec Does

把 5 个手写 skill（agent role 知识库）从静态全注入升级为按 D-RT-01 算法（regime + IDF + 元数据加权）排序的 top-k 注入。同时把 `propose_new_skill` 创建的新 skill `.draft` 文件交给 LLM 自动推断 metadata（regime_tags / triggers_keywords / importance / confidence），让新 skill 立即可被检索算法正确排序。

简言之：spec 014 的 5 个手写 SKILL.md 进化为含 6 个新字段的"会被排序的 skill 库"；spec 014 的 `propose_new_skill` 写 .draft 时自动加上 LLM 推断的 metadata。

**In scope：**
- Skill schema 加 6 字段（regime_tags / triggers_keywords / importance / access_count / last_accessed_at / confidence），全 default 兼容旧实例
- 5 个现有 SKILL.md 数据迁移（用 brainstorm 阶段 LLM 推断的硬编码 mapping）
- EvolvingSkillProvider 实现 D-RT-01 两层算法（无 sentence-transformers embedding）
- load_skill_tool factory 改造（接受 provider 参数，access_count 走统一 IO 入口）
- propose_new_skill 改造写 .draft 时调 LLM 推断 metadata
- 前端 /memory 页面加第 4 单行 section "Skills Grid" + 4 个新 API endpoints

**Out of scope（关键边界）：**
- **maturity FSM / pnl_track / applied_count / forbidden 等 pattern-only 字段** —— 不加到 Skill；Skill / Pattern 架构分离
- **sentence-transformers embedding（D-RT-01 第 3 组件）** —— 违反"不引入新依赖"约束
- **Skill 高级进化（importance 重计算 / stale 标记）** → spec 020 daemon
- **Anthropic prompt cache + git lineage** → spec 020

## Bigger Picture

trilogy（016 / 017a / 017b / 018-020）的 Skill 子域。spec 016 的 D-DS-01 + D-RT-01 + D-MW-01 决策落地；spec 018 已落地 EvolvingMemoryProvider，本 spec 在同 module-level singleton 中并存 EvolvingSkillProvider；spec 020 加 daemon 触发本 spec 的高级进化。

衔接点：
- spec 017a 的 `SkillProvider` Protocol 不变；EvolvingSkillProvider 实现它
- spec 018 的 `_memory_provider` 不动；本 spec 替换 `_skill_provider`
- spec 014 既有 `discover_skills_for_agent` / `propose_new_skill` / `load_skill_tool` factory 全部 reuse + 改造

值得思考的相邻关系：spec 014 既有 5 SKILL.md 是手写 agent role 知识库（chain-analysis / macro-analysis / news-analysis / tech-analysis 各对应一个 agent 的 system prompt 知识 + 1 个 shared trading-knowledge）。这与 PatternRecord（spec 018 进化的 cycle-derived 策略）是**两种不同概念**——spec 016 D-DS-01 把它们混淆，本 spec 修订只采纳适合 Skill 的 6 字段（不加 maturity FSM / pnl_track）。

外部参考：spec 016 的 MetaClaw（IDF 检索）+ EvoSkill（数值 frontier）+ Hermes（match-score）3 项目的 retrieval 方案在本 spec 综合实施；sentence-transformers embedding（D-RT-01 第 3 组件）违反约束被推迟到 spec 020。

---

## Spec Review Guide (30 分钟)

> 30 分钟 review 建议把时间花在 4 处：trilogy 边界、IDF 在小语料集上的实际价值、LLM 推断 metadata 成本、load_skill_tool factory 改造的破坏面。

### Understanding the approach (8 min)

读 [spec.md Purpose](spec.md#purpose) + [research.md Decision 1-6](research.md) 了解整体路径。带着这些问题阅读：

- 5 个手写 skill 的 maturity 已经是"manually edited 高质量"——为什么还需要 D-RT-01 排序？短期内 5 skill 全注入也能 work，spec 019 的实际收益主要是为未来 skill 集增长（spec 014 既有 propose_new_skill / spec 020 daemon 自动扩展）铺路。是否真的需要在 5 skill 现状就做？
- spec 016 D-DS-01 列了 17 字段；本 spec Q1 决策只采纳 6 字段。reviewer 应该确认"删去的 11 字段"（如 maturity FSM / pnl_track / applied_count）确实不适合 Skill。参考 [research.md Decision 2](research.md#decision-2maturity-不加到-skill)
- D-RT-01 第 3 组件（sentence-transformers embedding）被推迟到 spec 020。在 5 skill 集 + IDF 信息论上区分度低（log(5/1)=1.61）的现状下，本 spec 的检索算法实际效果是否优于 spec 014/15/17b 的全注入？参考 [research.md Decision 3](research.md#decision-3idf-算法实现)

### Key decisions that need your eyes (12 min)

**Maturity 不加到 Skill** ([research.md Decision 2](research.md#decision-2maturity-不加到-skill))

spec 016 D-DS-01 列出 maturity 字段（draft/tested/stable/clean/mature 5 状态），但 spec 014 的 5 SKILL.md 是手写知识库，没有 PnL 进化路径。本 spec 不加 maturity 到 Skill。

- Question for reviewer：未来 spec 020 daemon 加"基于 access pattern 重计算 importance"机制时，是否会有 maturity-equivalent 概念出现？例如"长期高 access_count 的 skill 算 mature；从未被引用的 skill 算 stale"。如果会，是否应该现在就加 maturity_score (float) 字段为 spec 020 铺路？目前不加

**5 skill 集 IDF 数学价值** ([research.md Decision 3](research.md#decision-3idf-算法实现))

5 skill 集上 IDF：`math.log(5/1) = 1.61`，`math.log(5/5) = 0`。区分度低；本 spec 仍实施 IDF 是为未来 skill 集增长铺路。

- Question for reviewer：是否值得把 IDF 推迟到 spec 020？目前实施需要 ~150 行（含单测）+ telemetry attribute，但短期实际效果≈ 0。是否更适合 spec 020 等 skill 集 ≥ 20 时再加？目前 brainstorm Q2 选了 B 含 IDF；reviewer 可质疑这个选择
- 反方观点：IDF 实施是 idempotent；现在做晚做都要做；30 LOC 的 algorithm.py 不算大成本

**LLM 推断 metadata 写入 .draft 的语义** ([research.md Decision 6](research.md#decision-6propose_new_skill-写-draft-含-llm-metadata))

`propose_new_skill` 现在写 `.draft` 文件含 LLM 推断的 metadata。用户 review .draft 后 manual save → 变 SKILL.md（metadata 已就位）。

- Question for reviewer：用户 review .draft 时是否会想自己手动调整 metadata（如 LLM 推断的 regime_tags 不准）？如果会，需要让 .draft 文件容易 manual edit（YAML 是；好）。但用户编辑后 save 会破坏 LLM 自动逻辑——本 spec 假设这是合理的（用户编辑优先），无机制冲突
- 另一面：LLM 推断失败时写默认值（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）；用户必须手动填。是否值得加 CLI 提示"LLM 推断失败，请手动编辑 .draft frontmatter"？目前 spec 没显式

**load_skill_tool factory 改造的破坏面** ([data-model.md load_skill_tool](data-model.md))

spec 014 既有 `load_skill_tool = _make_load_skill_tool()` module-level 实例；本 spec 在 `nodes/agents.py` init 时**替换** `_t.load_skill_tool` instance。

- Question for reviewer：替换 module-level instance 是 monkey-patch 模式，与 Python pythonic 风格不太一致。reviewer 可能更倾向于把 ToolAgent 实例化时显式拿 provider 注入，而不是依赖 module-level monkey-patch。但 spec 017b 的 ToolAgent 已经是 `tools=[..., load_skill_tool]` 模式，要改的话动 spec 017b 既有路径。本 spec 选了"最小破坏"路径：仅 monkey-patch module-level instance
- 影响：测试时 `import ... load_skill_tool` 拿到的可能是改造后或改造前的 instance（取决于 nodes/agents.py 是否已 init）。test_load_skill_tool.py 需要明确 fixture 模式

### Areas where I'm less certain (5 min)

- [tasks.md T024](tasks.md#phase-4-provider--集成--c3-commitatomic-切换) 修改 `_make_load_skill_tool` 加 provider 参数，但保留 spec 014 兜底（provider=None 时走 `load_skill(name)`）。这是为了向后兼容 spec 014 调用，但与 Q1 C 决策"直接删旧不留 fallback"略有冲突。Reviewer 可质疑：是否应该删除 spec 014 兜底分支，强制 provider 必传？目前我选了软性向后兼容
- [tasks.md T040](tasks.md#phase-5-api--前端--e2e--c4-commit) 提到 i18n 文件路径"先 grep 确认"——spec 018 实际是 `web/src/locales/zh-CN/memory.json`，这个 brainstorm 阶段已 spot-check 但 spec 没显式列出最终路径。Implement 时可能撞上 spec 014 的 i18n 模式（`web/src/i18n/zh-CN.ts`）vs spec 018 模式（`web/src/locales/...`）的差异
- [contracts/skill-api-routes.md GET /api/memory/skill-access](contracts/skill-api-routes.md) 实际只返回当前 skill 状态（access_count + last_accessed_at），不是细粒度 access 时间戳事件流。如 reviewer 期望"每次 access 都有独立 event"，需要 spec 020 daemon 加 access log（OOS）。本 spec 的 endpoint 名字"skill-access"可能误导
- [tasks.md T019](tasks.md#phase-4-provider--集成--c3-commitatomic-切换) "从 snapshot 推 current_regime（如 funding_rate > 0.0003 → high_funding）" — 实际是否 5 skill 都用 regime_tags=[]（match all）？如果是，第一层 regime_tags 预过滤就是 noop。spec.md FR-W3 5 skill mapping 都是 regime_tags=[]，所以是的——但本 spec 仍实施 regime_tags 提取逻辑为未来用户手动编辑 / spec 020 LLM 推断填充非空 regime_tags 准备

### Risks and open questions (5 min)

- 实施 subagent 在 C3 atomic commit 期间是否会触及 BLACKLIST 文件（spec 014 既有 learning/curation.py 等）？tasks.md BLACKLIST 已显式列出但 subagent 仍可能"helpfully" 改 spec 014 既有 skill_proposal.py 内部不在 spec 范围的部分。建议 implement subagent prompt 含明确"修改 propose_new_skill 函数仅在 build_draft_content 之后加 LLM 推断步骤；不要改其他部分"
- LLM 推断 metadata 在 `.draft` 写入时跑——如果 propose_new_skill 在 cycle 内被调用（spec 014 当前是手动 CLI），则 cycle 增加 1 LLM 调用延迟（~500-1000ms）。需要确认 spec 014 的 propose_new_skill 调用频率不在 cycle 关键路径
- Migration 脚本对未知 skill name 用空 mapping。如果生产环境 spec 020 daemon 已经创建过若干新 skill（带 LLM 推断），spec 019 落地时这些 skill 已含字段——但 spec 020 还未上线，所以现状只有 5 个 skill，安全
- 5 skill 的 access_count 在第一次落地时全是 0；recency_bonus 全是 1.0（exp(0)=1）；第一次 cycle retrieval 排序退化为 `(idf + importance + 1) × confidence`。是否合理？因为 importance 范围是 [0,1]，confidence [0,1]，"+1" 主导排序——所有 skill 都被算成"重要+最近"，IDF 才是真正的区分。这是预期行为吗？目前 spec FR-W8 算法没有考虑这个 edge case；可能需要 score = `(idf + importance) × confidence × recency_bonus`（recency 作乘子而非加项）。Reviewer 应该 review 这个公式的合理性

---
*完整内容见 [spec.md](spec.md)、[plan.md](plan.md)、[tasks.md](tasks.md)。*
