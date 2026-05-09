# Feature Specification: Spec 020a — Trilogy Ops（cache 观测 + advisory 收尾 + 部署清单 + monitoring）

**Feature Branch**: `021-trilogy-ops`
**Created**: 2026-05-09
**Status**: Draft
**Input**: User description: "spec-020a-trilogy-ops — Trilogy（spec 017a/b + 018 + 019）已全部合并 main，spec 020 收尾 Ops 子域。原 spec 020 范围（cache + daemon + lineage + advisory + 部署 + monitoring）经 brainstorm Q1 拆分：本 spec（020a）= cache 观测 + 3 项 P2 advisory 修复 + 部署清单 + 2 项核心 monitoring；后续（020b）= reflect daemon + git lineage 自动化。"

## Clarifications

### Session 2026-05-09

- Q: FR-Z18 的 2 个 monitoring 指标聚合源 / dashboard 消费路径？ → A: 复用 spec 015 既有 `/metrics` Prometheus endpoint，新增 2 metric 由 OTel span attr 进程内聚合（不引入新 pipeline）
- Q: SC-Z3 期望 cache attr 出现在 5 处 LLM call（4 agent + verdict）— verdict 可能走 weighted-downgrade 跳过 LLM 调用，如何计数？ → A: SC-Z3 改为 "≥ 4 agent LLM 点必须含 cache 字段（verdict 因 weighted-downgrade 可能跳过 LLM 不计入）"
- Q: `log_llm_usage()` 在 cache_read + cache_creation 同时为 0 时（非 Anthropic provider / cache 关闭）如何写 OTel attr？ → A: 仍写 3 字段全 0（read=0/creation=0/hit_rate=0），保持字段一致性，便于 dashboard query

## Background

Trilogy（spec 017a/b + 018 + 019）已合并 main，trilogy 共交付：

- spec 017a：PromptBuilder 基建（commit cfd3acc）
- spec 017b：4 agent 集成切换（commit 5b65a4a）
- spec 018：Memory Evolution（commit 458a0f2 + 14afc50 + 1c0302d）
- spec 019：Skill Evolution（commit 3fbf941）

trilogy 落地后留下 5 个待办：

1. Anthropic prompt cache 已生产但**未量化** hit rate（5min ephemeral TTL vs 1h cycle frequency 的实际 ROI 未知）
2. spec 018 P2-1 advisory：IVE `classify_case` 同步阻塞 event loop（`llm.invoke()` 调用 3-5s）
3. spec 019 P2-02 advisory：SkillsGrid 缺 `triggers_keywords` badges
4. spec 019 P2-03 advisory：`propose_new_skill` LLM 推断失败时未持久化失败标志
5. trilogy 生产部署缺 staging 验证脚本 + rollback runbook

本 spec 是 trilogy 的运维收尾，不含 daemon / lineage（拆 spec 020b）。直接删旧不留 fallback。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Staging 验证脚本（Priority: P1）

作为运维人员，部署 trilogy 到生产前，我希望用 1 条命令跑完整的 staging smoke check（migrate dry-run + 单 cycle telemetry + retrieval 命中校验），把跨 spec 部署风险压缩到 1 次脚本调用。

**Why this priority**：trilogy 含 3 个 migrate 脚本（spec 017b 配置 / 018 patterns+cases / 019 SKILL.md frontmatter），手动执行容易遗漏顺序；生产部署事故窗口短，需要可重复的 1-key smoke。

**Independent Test**：在 dev 机或 CI 跑 `python scripts/staging_validate.py --dry-run`，输出每 step 的 PASS/FAIL 行，整体 exit code = 0 即视为通过。

**Acceptance Scenarios**：

1. **Given** trilogy 全部已合并 + dev 机干净状态，**When** 运行 `python scripts/staging_validate.py --dry-run`，**Then** stdout 含 ≥ 6 个 `[step N] <name>: PASS` 行，整体 exit 0
2. **Given** 故意破坏 spec 018 migrate 脚本（rename 关键函数），**When** 运行验证脚本，**Then** 在该 step 输出 `FAIL` + 错误原因，整体 exit ≠ 0
3. **Given** mocked LLM 返回缺 `usage_metadata` 字段，**When** 运行 cycle smoke step，**Then** cache 字段 step 输出 FAIL 标志缺失字段名称

---

### User Story 2 - Rollback Runbook（Priority: P1）

作为 SRE，trilogy 部署后若发现 cycle 异常，我希望有清晰的 rollback runbook 含每个 spec 的 git revert SHA + DB 回退命令 + 验证步骤，确保生产事故能在 ≤ 30 min 内回滚。

**Why this priority**：trilogy 涉及 3 个 commit（17b/18/19），每个含数据迁移；事故时凭记忆 revert 易遗漏；runbook 是生产 SLA 前置。

**Independent Test**：检查 `docs/rollback-trilogy.md` 存在，含 spec 017b / 018 / 019 三个 rollback 段，每段 ≥ 3 step（git 命令 / DB 回退 / 验证），且每段含 "rollback 后 known data loss" 章节。

**Acceptance Scenarios**：

1. **Given** runbook 文档存在，**When** SRE 按 spec 018 rollback 段执行所有 step，**Then** repo 状态回到 spec 017b 时点 + agent_memory 旧 schema 可读
2. **Given** runbook 含 known data loss 章节，**When** 阅读 spec 018 段，**Then** 含说明 "archived rules" 数据将丢失
3. **Given** 运维不熟悉 trilogy，**When** 仅按 runbook 操作（无需读代码），**Then** 能完成完整 rollback

---

### User Story 3 - Cache Hit Rate 可视化（Priority: P1）

作为架构师，trilogy 的 Anthropic prompt cache 已通过 `apply_cache_control()` 进入生产，我希望 dashboard 能看到实际的 cache hit rate，作为后续是否升级 head/tail 拆分 + 1h beta cache 的决策依据。

**Why this priority**：当前 `log_llm_usage()` 仅写 structlog 不写 OTel；ephemeral 5 min TTL vs 1h cycle frequency 的 hit rate 未知；不量化就无法判断 ROI。

**Independent Test**：触发 1 个 mocked cycle，检查 OTel trace 中 4 agent + verdict 共 5 处 LLM span 各含 `llm.cache.read_tokens` / `llm.cache.creation_tokens` / `llm.cache.hit_rate` 3 个 attribute；dashboard `/metrics` 页加载后能看到 cache hit rate panel。

**Acceptance Scenarios**：

1. **Given** mocked LLM 返回 `usage_metadata={cache_read_input_tokens=100, cache_creation_input_tokens=400}`，**When** cycle 完成，**Then** OTel span 含 `llm.cache.hit_rate=0.2`
2. **Given** mocked LLM 返回 `usage_metadata` 不含 cache 字段（OpenAI 等非 Anthropic），**When** cycle 完成，**Then** OTel span 含 `llm.cache.read_tokens=0` / `llm.cache.creation_tokens=0` / `llm.cache.hit_rate=0`，不抛异常
3. **Given** dashboard 已部署，**When** 访问 `/metrics` 页，**Then** 看到 24h 平均 cache hit rate panel

---

### User Story 4 - IVE Async 化（Priority: P1）

作为维护者，spec 018 的 IVE `classify_case` 当前用 sync `llm.invoke()` 阻塞 event loop 3-5s，我希望改为 `await llm.ainvoke()` 让 cycle 不再阻塞。

**Why this priority**：阻塞 event loop 影响所有并发 IO（market data fetch / API endpoints），cycle 内可能有多次 IVE 调用导致累积阻塞；spec 018 P2 review-code 已标注。

**Independent Test**：`grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py` 返回空；`pytest tests/test_ive.py -v` PASS；`evaluate_node` 调用方改 await 后单测仍 PASS。

**Acceptance Scenarios**：

1. **Given** IVE 单测 fixture，**When** 调用 `await classify_case(...)`，**Then** 返回正确分类结果
2. **Given** spec 018 evaluate_node async 上下文，**When** 调用 IVE，**Then** event loop 不阻塞（其他 task 可并发）
3. **Given** mocked LLM 超时（30s），**When** IVE 调用，**Then** 走现有 fallback 不阻塞 cycle

---

### User Story 5 - SkillsGrid Triggers + Failure Flag（Priority: P2）

作为前端用户和 skill 维护者，我希望：(a) SkillsGrid 卡片显示 `triggers_keywords` badges，让人看到 skill 触发条件；(b) `propose_new_skill` 的 LLM 推断失败时，`.draft` frontmatter 写入 `inference_failed: true` 标志，便于人工补救识别。

**Why this priority**：spec 019 P2-02 / P2-03 review-code advisory 已标注；不影响核心功能但影响诊断可见性。

**Independent Test**：(a) `grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 1 hit；Vitest 单测验证 5 个 badges + `+N more` 折叠正确；(b) `pytest tests/test_skill_proposal_metadata_inference.py::test_llm_failure_writes_flag` PASS。

**Acceptance Scenarios**：

1. **Given** skill 含 8 个 triggers_keywords，**When** SkillsGrid 渲染该 skill，**Then** 显示 5 个 badges + "+3 more"
2. **Given** skill `triggers_keywords=[]`，**When** SkillsGrid 渲染，**Then** triggers row 整体不渲染（不显示空 row）
3. **Given** propose_new_skill LLM 推断抛异常，**When** 流程完成，**Then** `.draft` frontmatter 含 `inference_failed: true` + 默认 metadata（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）

---

### Edge Cases

- staging_validate 在 trilogy 任一 migrate 失败时 MUST 退出 ≠ 0 + 输出失败 step 详情
- `cache_creation_input_tokens` 字段 LLM provider 未返回时（OpenAI 等非 Anthropic）写 0，不抛异常
- IVE `await llm.ainvoke()` 超时（默认 30s）走现有 fallback 不阻塞 cycle
- SkillsGrid `triggers_keywords` 为空 list 时不渲染 badge row
- `propose_new_skill` LLM 失败时 frontmatter 默认 metadata：`regime_tags=[]` / `triggers_keywords=[]` / `importance=0.5` / `confidence=0.5` / `inference_failed=true`
- OTel SDK 未初始化时（test 环境），cache attr 不抛异常但记入 structlog
- staging_validate `--dry-run` flag 未指定时默认走 dry-run（避免误触发实跑 migrate）

## Requirements *(mandatory)*

### Functional Requirements

#### Staging 验证脚本

- **FR-Z1**：`scripts/staging_validate.py` MUST 存在，含 main 函数支持 `--dry-run` flag（默认 True）
- **FR-Z2**：脚本 MUST 顺序执行：(a) `migrate_017_to_018.py --dry-run`；(b) `migrate_018_to_019.py --dry-run`；(c) 启动单 cycle smoke（mocked LLM + APScheduler 单次触发）；(d) 校验 OTel telemetry 含 spec 017a FR-X18 列出 8 字段 + 本 spec 3 cache 字段；(e) 校验 EvolvingSkillProvider retrieval 命中 ≥ 1 skill；(f) 任一 step 失败 exit ≠ 0
- **FR-Z3**：脚本 stdout 格式 `[step N] <name>: PASS|FAIL <duration>ms`，便于 CI 解析

#### Rollback Runbook

- **FR-Z4**：`docs/rollback-trilogy.md` MUST 存在，含 trilogy 3 spec 的 rollback 段（spec 017b / 018 / 019）
- **FR-Z5**：每段 MUST 含：(a) git revert 命令（带具体 commit SHA）；(b) DB / 数据回退命令（如有 schema / 文件变更）；(c) 验证 step（grep / curl / pytest 之一）
- **FR-Z6**：每段 MUST 含 "rollback 后 known data loss" 章节描述哪些后续 cycle 数据会丢失（如 spec 018 archived rules / spec 019 .draft 文件）

#### Cache 观测

- **FR-Z7**：`src/cryptotrader/agents/base.py:log_llm_usage()` MUST 加 `cache_creation_input_tokens` 字段提取（与既有 `cache_read_input_tokens` 同等位置）
- **FR-Z8**：`log_llm_usage()` MUST 写 OTel span attr：`llm.cache.read_tokens` / `llm.cache.creation_tokens` / `llm.cache.hit_rate`（hit_rate = read / (read + creation)，分母 0 时为 0）；read + creation 同时为 0 时仍 MUST 写 3 字段全 0（保持字段一致性，便于 dashboard query 不需 null 处理）
- **FR-Z9**：OTel span attr MUST 在 4 agent + verdict 调用 LLM 时全覆盖，由 `log_llm_usage()` 统一注入（即所有走 `acompletion_with_fallback` 的调用点）

#### IVE Async 化

- **FR-Z10**：`src/cryptotrader/learning/evolution/ive.py:classify_case()` MUST 改为 `async def`；`llm.invoke(messages)` MUST 改为 `await llm.ainvoke(messages)`
- **FR-Z11**：所有调用 `classify_case()` 的位置 MUST 改 await（grep 验证全 repo 无遗漏调用点）
- **FR-Z12**：IVE 单测 MUST 改用 `pytest.mark.asyncio` fixture（pytest-asyncio 已在 dev deps）

#### SkillsGrid Triggers + Inference Failure

- **FR-Z13**：`web/src/pages/memory/components/SkillsGrid.tsx` MUST 加 `triggers_keywords` badge row（最多显示 5 个，多余以 `+N more` 标识）
- **FR-Z14**：badge row MUST 在 `regime_tags` 行下方，使用 muted 颜色（区别于 regime_tags 主色）
- **FR-Z15**：`triggers_keywords` 为空 list 时整个 row MUST 不渲染（不显示空 row）
- **FR-Z16**：`src/cryptotrader/learning/evolution/skill_metadata_inference.py` MUST 在 LLM 调用 except 路径写 `inference_failed: True` 字段到返回 metadata
- **FR-Z17**：`src/cryptotrader/learning/skill_proposal.py:propose_new_skill()` MUST 把 `inference_failed` 字段（如存在）写入 `.draft` frontmatter

#### Monitoring（仅 dashboard，不告警）

- **FR-Z18**：复用 spec 015 既有 `/metrics` Prometheus endpoint，新增 2 个 metric：(a) `llm_cache_hit_rate_24h_avg`（来自 OTel span attr 进程内聚合，24h sliding window）；(b) `ive_classify_failure_rate_1h_avg`（1h sliding window）。**不**引入新数据 pipeline / Prometheus exporter / DB 表。
- **FR-Z19**：仅 web `/metrics` 页加 2 个 panel，不写 alertmanager 规则（避免 alert fatigue）

#### Migration

- **FR-Z20**：本 spec 不含 schema 变更；不需要新 migrate 脚本

### Key Entities

- **Cache Telemetry Record**：每次 LLM 调用产出的 OTel span attribute 集合，含 read_tokens / creation_tokens / hit_rate 三字段；嵌入 LLM 调用 span 上下文（无独立存储）
- **Rollback Step**：runbook 中单个 step，含 type（git / db / verify）/ command（具体 shell 指令）/ expected output（验证依据）
- **Inference Failure Flag**：skill metadata 字典中的 boolean 字段，frontmatter 中以 `inference_failed: true/false` 表达，触发条件 = LLM 推断 except 兜底
- **Validation Step**：staging_validate 脚本中单个 step，含 step_index / name / status (PASS|FAIL) / duration_ms / error_detail (可选)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-Z1**：`python scripts/staging_validate.py --dry-run` 在 dev 机一次成功 PASS（exit 0），运行时间 ≤ 60s
- **SC-Z2**：`docs/rollback-trilogy.md` 存在含 3 spec 回退路径，每段 ≥ 3 个可执行 step + 1 个 known data loss 段落
- **SC-Z3**：1 mocked cycle 后 OTel trace 中 ≥ 4 agent LLM 点各含 3 个 cache 字段（共 ≥ 12 个 attribute 点）；verdict 因可能走 weighted-downgrade 跳过 LLM 不计入 SC-Z3 但若调用 LLM 则也必须含 cache 字段
- **SC-Z4**：`grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py` 返回空（仅 `llm.ainvoke` 存在）
- **SC-Z5**：`grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 1 hit + Vitest 测试 PASS（5 badges + +N more 折叠）
- **SC-Z6**：`pytest tests/test_skill_proposal_metadata_inference.py::test_llm_failure_writes_flag` PASS
- **SC-Z7**：dashboard `/metrics` 页加载后含 2 个新 panel（cache hit rate / IVE failure rate），manual smoke
- **SC-Z8**：spec 014 / 015 / 17a / 17b / 18 / 19 既有测试不回归（基线 2339 test pass / 0 fail，本 spec 落地后 ≥ 2339 pass）
- **SC-Z9**：`/spex:review-spec` 无 P0 / P1 issues
- **SC-Z10**：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-Z11**：本 spec 单 PR 不超过 4 个 commit（C1 文档 + C2 后端 + C3 前端 + C4 E2E gate）

## Assumptions

- spec 010 OTel SDK 已在生产环境配置 + dashboard 可消费 span attr（此前已验证）
- spec 015 sanitize_input 不影响 OTel attr 写入路径
- LLM provider 返回的 `usage_metadata` 字典在 LangChain `AIMessage` 上字段命名稳定（langchain-openai >= 1.1.10 / langchain-core >= 1.2.17）
- IVE `classify_case` 调用方均处于 async context（spec 018 已确认 `evaluate_node` 是 async coroutine）
- staging_validate 脚本在 dev 机 + CI 均可跑（不依赖生产 DB；mocked LLM + 临时目录）
- pytest-asyncio 已在 dev dependencies（spec 018 已引入）
- Anthropic prompt cache 5 min ephemeral TTL 短期不变；如未来升级 1h beta cache 需重写本 spec hit_rate 公式

## Dependencies

**Upstream**：
- spec 010（OpenTelemetry tracing 基建）
- spec 015（metrics endpoint + sanitize_input 防注入）
- spec 017a（PromptBuilder 基建）
- spec 017b（4 agent 集成切换）
- spec 018（Memory Evolution + IVE）
- spec 019（Skill Evolution + EvolvingSkillProvider + propose_new_skill + SkillsGrid）

**Downstream**：
- spec 020b（reflect daemon + git lineage 自动化，复用本 spec rollback runbook 模板 + monitoring 指标扩展到 5 个）

**External tooling**：无新依赖（OTel 已通过 spec 010 接入；pytest-asyncio 已存在）

## Out of Scope

- ❌ Reflect daemon（spec 016 D-ENG-01 → spec 020b）
- ❌ Git lineage 自动化（spec 016 D-ENG-02 → spec 020b）
- ❌ 强 cache 拆分 head/tail + 1h beta cache 落地（待本 spec B 观测后定，可能升级到 020b 或独立 spec）
- ❌ 5 个 metrics 全套告警 + alertmanager 规则 + Slack webhook（仅 2 个 dashboard panel，不告警）
- ❌ 新 prompt 内容优化 / 新 agent / 新 skill / 新 retrieval 算法
- ❌ schema 变更 / 新 migrate 脚本（本 spec 无数据迁移）
- ❌ Anthropic 之外的 LLM provider 的 cache（OpenAI / Gemini 当前不支持 prompt cache）
- ❌ 配置热重载

## Reversibility

本 spec 落地后可通过 git revert 单 PR 回退（无 schema 变更，无数据迁移）。回退后：
- `scripts/staging_validate.py` 删除（不影响生产 cycle）
- `docs/rollback-trilogy.md` 删除（不影响生产 cycle）
- `log_llm_usage()` cache 字段恢复仅 `cache_read_input_tokens`（不影响 cycle 运行，仅丢失 cache hit rate 观测）
- IVE `classify_case` 恢复 sync（不影响功能，仅恢复 event loop 阻塞）
- SkillsGrid 恢复无 triggers_keywords badges（不影响数据完整性）
- `propose_new_skill` `inference_failed` 字段恢复无（不影响 .draft 写入流程）

## Implementation Outline

### 单 PR 切 4 commit（与 spec 019 同 pattern）

**C1 — 基础设施 + 文档**：
- `scripts/staging_validate.py`（新）
- `docs/rollback-trilogy.md`（新）

**C2 — 后端 algorithm + telemetry**：
- `src/cryptotrader/agents/base.py:log_llm_usage()` 加 `cache_creation_input_tokens` 字段 + OTel span attr
- `src/cryptotrader/learning/evolution/ive.py:classify_case()` async 化
- `src/cryptotrader/learning/evolution/skill_metadata_inference.py` failure flag 持久化
- `src/cryptotrader/learning/skill_proposal.py:propose_new_skill()` frontmatter 写 `inference_failed`
- 调用方 await 修正（grep 全 repo）
- 单测更新（pytest-asyncio）

**C3 — 前端 + dashboard**：
- `web/src/pages/memory/components/SkillsGrid.tsx` triggers_keywords badges
- `web/src/pages/metrics/` 加 2 panel（cache hit rate / IVE failure rate）
- Vitest 测试

**C4 — E2E + 最终 Gate**：
- `tests/test_e2e_trilogy_ops.py` 单测：mocked cycle 验证 OTel cache attr 全覆盖
- `pyproject.toml` per-file-ignores（如需）
- grep / wc / pytest / ruff 全部 gate
