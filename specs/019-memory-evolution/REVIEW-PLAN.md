# Review Guide: Memory Evolution（spec 018）

**Spec:** [spec.md](spec.md) | **Plan:** [plan.md](plan.md) | **Tasks:** [tasks.md](tasks.md)
**Generated:** 2026-05-09

---

## What This Spec Does

把 spec 014 既有的"记忆机制"从静态变成会进化的：4 个 analysis agent 跑完一次决策后，根据 PnL + LLM 失败诊断自动调整记忆中规则的成熟度状态、按客观信号自动归档无效规则、用 Pareto frontier 排序选取注入到下一次决策 prompt 的 top-k 规则。同时修复 spec 017a 留下的 DefaultMemoryProvider 路径 bug（17a 当时设计时假设 cases 存为 `<agent>/cases.jsonl`，实际 spec 014 是 `cases/<cycle_id>.md` 全局），让 4 agent 真正收到记忆注入。

**In scope：**
- DefaultMemoryProvider 路径 bug 修复 + 数据迁移（~80 case + 任何 patterns）
- 5-signal maturity FSM（沿用 spec 014 既有 4 状态 + 加 archived 终态，spec 016 D-EV-03 的 5 信号映射到状态转换条件）
- Pareto frontier 双目标排序（win_rate × confidence_proxy，D-EV-02）
- IVE 失败分类（每 case 1 次 LLM 调用，D-EV-04）
- "三连续 fundamental → 自动归档"
- cases/<id>.md schema 扩展（Trade Execution + Causal Chain + IVE Classification 3 段）
- 前端 `/memory` 页面（4 sections）+ 4 个 API endpoints

**Out of scope（关键边界）：**
- **SKILL.md schema 升级 / EvolvingSkillProvider** → spec 019（Skill 子域）
- **Anthropic prompt cache / offline reflect daemon / git lineage** → spec 020（Ops 子域）
- **`Maturity` 重新定义为 5 状态** → 沿用 spec 014 既有 4 状态 + archived 终态（不破坏既有数据）
- **GPU 加速** → spec 016 已明确 NO GPU

## Bigger Picture

这是 trilogy（016 研究 → 017a/b 基建 → 018-020 进化算法）的第 3 段。spec 016 完成了 8 项目研究（[research/synthesis.md](../016-research-skill-evolution-prior-art/research/synthesis.md)）确立路径；spec 017a/b 完成了 PromptBuilder + MemoryProvider 协议接入；本 spec 把 17a 的占位 DefaultMemoryProvider 替换为真正会进化的 EvolvingMemoryProvider。

衔接点：spec 019（Skill）会替换 DefaultSkillProvider 为 EvolvingSkillProvider；spec 020（Ops）加 daemon 触发本 spec 的 `evaluate_all_rules()` + `classify_pending_cases()`。本 spec 暴露的 Provider 接口稳定，spec 019/020 不需要改本 spec 任何代码。

值得思考的相邻关系：spec 014 既有 `learning/memory.py` 含 `distill_patterns` 函数（reflect 流程）。本 spec **不替换** distill — 只在 cycle 末段加 evaluate_node 跑 evaluate + classify。distill 仍由 spec 014 既有路径触发（manual / spec 020 的 daemon）。这种"叠加而非替换"是为了 trilogy 边界清晰，但代价是 distill 逻辑可能与本 spec FSM/Pareto 有重复点 — 留给 spec 020 重新审视。

外部参考：spec 016 的 SkillClaw（D-MW-02 因果链）+ Hermes（D-EV-04 失败分类）+ EvoSkill（D-EV-02 Pareto frontier）+ skill-evolution（D-EV-03 FSM）4 项目的核心算法本 spec 都直接采纳。

---

## Spec Review Guide (30 分钟)

> 30 分钟 review 建议把时间花在 4 处：trilogy 切分边界、Maturity 兼容策略、IVE LLM 成本、C3 atomic commit 风险。

### Understanding the approach (8 min)

读 [spec.md Purpose](spec.md#purpose) + [tasks.md Implementation Strategy](tasks.md#implementation-strategy) 了解整体路径。带着这些问题阅读：

- 是否真的应该把 Memory / Skill / Ops 切分为 3 个独立 spec？三者其实都是 "Provider 实现 + 算法 + 持久化"模式，是否单 spec 大包合更高效？参考 [research.md Decision 1](research.md) — 我没把 Q1 三 spec 的具体 alternative 写进研究，brainstorm 阶段已敲定但论据是"实施风险" + "trilogy 习惯"
- 用户偏好"删旧不留 fallback"，但本 spec 选了 Q6 = C "Empty placeholder + warning log"（cycle 不 break）。这是不是隐性 fallback？参考 [spec.md FR-Z9](spec.md#evolvingmemoryprovider) — 我认为这是"故障容错"不是"fallback to 旧路径"，但 reviewer 可质疑这条边界
- spec 016 D-EV-03 字面提到 "draft / tested / stable / clean / mature" 5 状态，本 spec 沿用 spec 014 的 4 状态 + 加 archived。映射是否准确？参考 [research.md Decision 1](research.md#decision-1maturity-沿用-4-状态--加-archived不重新定义-5-状态) — D-EV-03 的"5-signal"实质是"5 个状态转换信号"，不是"5 个状态"，所以映射合理；但 reviewer 应该确认这是符合原意

### Key decisions that need your eyes (12 min)

**Maturity 沿用 4 状态 + archived，不重新定义 5 状态** ([research.md Decision 1](research.md#decision-1maturity-沿用-4-状态--加-archived不重新定义-5-状态))

spec 014 的 `Maturity = Literal["observed","probationary","active","deprecated"]` 已生产运行；spec 016 D-EV-03 提到"draft/tested/stable/clean/mature"5 状态。本 spec 选择映射而非重定义。

- Question for reviewer：映射 `observed → "draft"` / `probationary → "tested"` / `active → "stable+clean+mature"` / `archived → "archived"` 是否会丢失 D-EV-03 中"clean"和"mature"的差异化语义？clean 状态是"frontmatter 全填 + body ≤300 行"，mature 是"无硬编码常量 + 依赖完整"。我合并到 active 一个状态，意味着这两个细分诊断丢了。如果 reviewer 认为有价值，需要展开为 5 状态；目前 spec 没有

**IVE LLM 成本：每 case 必跑 vs 仅亏损 case 跑** ([brainstorm Q5](../../brainstorm/04-spec-018-memory-evolution.md))

每月 ~3600 次 IVE LLM 调用 × ~500 token = 1.8M token / 月。GPT-4o-mini 实际成本 ~$0.27/月。但成本不是唯一因素：

- Question for reviewer：每 case 跑 IVE 意味着盈利 case 也跑（结果默认 noise）。是不是浪费？brainstorm 选了 A 而不是 B（仅亏损 case 跑），理由是"完整失败信号"。但 reviewer 可质疑：盈利 case 跑 IVE 真的有价值吗？是否应该改 B（仅亏损跑）来降成本 70%？目前 spec FR-Z17 是 A

**C3 atomic commit 体积** ([tasks.md Phase 4](tasks.md#phase-4-provider--nodes-集成--c3-commitatomic-切换))

C3 commit 含 18 个 task，diff ~1100 行，触及 prompt_builder.py / nodes/agents.py / nodes/evolution.py / nodes/journal.py / nodes/execution.py / graph.py + 2 测试。spec 017b 经验：~1100 行 atomic commit subagent drift 风险中等；本 spec 同量级。

- Question for reviewer：C3 是否值得拆分？拆分需要 compat shim（在 atomic 外加临时 fallback），违反"无 fallback"偏好。但 17b 的 C2 atomic 也是 ~1100 行成功了，所以风险可控

**前端 vitest 配置 + i18n 文案** ([tasks.md Phase 5](tasks.md#phase-5-前端-memory--api--e2e--c4-commit))

C4 含 web/ 改动 11 文件。`tests/web/test_memory_page.tsx` 用 vitest 跑（spec 014 既有 vitest.config.ts）。

- Question for reviewer：spec 014 的 vitest 是否覆盖了类似规模的页面？本 spec 是首个新加 `/memory` 页面 + 4 子组件。如果 vitest fixture / mocking 模式不一致，T052 的 4 用例可能要花更多时间。我假设 spec 014 既有 fixture 模式可复用（参考 `tests/web/test_*.tsx` 现有文件，但我没 grep 验证），如果不存在 fixture 模式，C4 工作量可能 +1 天

### Areas where I'm less certain (5 min)

- [tasks.md T030](tasks.md#phase-4-provider--nodes-集成--c3-commitatomic-切换) 描述"在 risk_router 之后加 evaluate 节点共享给两条 journal 分支"。我假设 LangGraph 支持多个上游节点指向同一节点（fan-in pattern），spec 014 既有 graph 应该有此模式。如果不支持需要拆为 evaluate_for_trade + evaluate_for_rejection 两个节点，T028 实施时确认
- [contracts/memory-api-routes.md](contracts/memory-api-routes.md) 假设 spec 015 既有 API 鉴权（X-API-Key header）模式，本 spec 4 endpoints 沿用。如果实际不是这种模式（如 cookie auth），T040 实施时需调整
- [spec.md FR-Z16](spec.md#ive-失败分类) 的"诊断答案 5 项 yes/no/uncertain"是 LLM 输出 schema。如果 LLM 给出非合法 schema（如返回 "true"/"false"），代码需要标准化 / 重试。FR-Z18 提到"LLM 输出非合法 JSON → 重试 1 次"，但没有显式覆盖"格式不合"的场景。T017 实施时应处理
- [tasks.md T033](tasks.md#phase-4-provider--nodes-集成--c3-commitatomic-切换) "填充 causal_chain 字段（per-agent tool_calls 摘要 + verbal_reinforcement_input + debate_intermediate）— 从 state 中提取" — state 里是否有 tool_calls？我未确认 LangGraph state 是否含 ToolCall 序列；spec 010 OpenTelemetry trace 才是真实数据来源。如果 state 没有，则 causal_chain 段只能写 verbal_reinforcement_input + debate_intermediate（部分缺失）。T033 实施时验证；可能需要从 OTel trace 后端读取（但那超出 spec 014 的 in-process state 范畴）

### Risks and open questions (5 min)

- 实施 subagent 在 C3 atomic commit 期间是否会触及 OOS 范围（如 17b 实施时误改 memory.py）？spec.md OOS + tasks.md BLACKLIST 已显式列出但 subagent 仍可能 drift；建议 implement subagent prompt 包含明确"以下文件 NOT 在范围"
- 迁移脚本对**生产环境**实施前必须 dry-run 验证。如果迁移失败半途，部分 case 已加新段、部分未加 → 数据不一致。T038 提到 staging dry-run，但生产实施需要在 6 步 Migration Strategy 中加"备份验证"步骤
- IVE 5 诊断问题中"是否同 regime 下其他规则也亏损？"需要查同 regime_tags 的历史 case，但生产 patterns 大部分是空目录（spec 014 数据未积累）。早期 1-2 周 IVE 可能因缺乏对比 case 而效果差。是否值得给 spec 020 加一个"等待 N case 累积才启用 IVE"的开关？目前没有
- 前端 `/memory` 页面 4 sections 可能信息密度过低（rules grid 4 agent × 5 状态 = 20 格，多数格为空；transitions 流可能 1 cycle 0-1 条）。是否值得在 spec 020 之前加一个"Empty State"友好提示？目前 SC-Z13 仅要求 4 用例 PASS，未要求"信息密度合理"

---
*完整内容见 [spec.md](spec.md)、[plan.md](plan.md)、[tasks.md](tasks.md)。*
