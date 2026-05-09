# Spec 020a — Trilogy Ops（cache 观测 + advisory 收尾 + 部署清单 + monitoring）

**关联 spec**：[研究 016](../specs/016-research-skill-evolution-prior-art/) / [017a](../specs/017-agent-prompt-externalization/) / [017b](../specs/018-agent-prompt-builder-integration/) / [018](../specs/019-memory-evolution/) / [019](../specs/020-skill-evolution/)
**Date**: 2026-05-09
**Status**: brainstorm 完成，待 ship

## 目标

trilogy（spec 017a/b + 018 + 019）已全部合并 main。spec 020 收尾 Ops 子域，按 Q1 拆分：

- **020a（本 spec）**：cache 观测 + 3 项 P2 advisory + 部署清单 + 2 项 monitoring
- **020b（后续）**：reflect daemon + git lineage 自动化

## 5 项关键决策

### Q1：spec 拆分

**Decision**：B — 拆 020a + 020b 两段。

**Rationale**：
- 原 spec 020 范围（cache + daemon + lineage + advisory + 部署 + monitoring）混合"立即可做"和"需长期观察"两类工作
- 020a 集中"已知改动小、可测可验"项；020b 跑 daemon 周期评估再定
- 拆分降低单 spec review 复杂度，与 trilogy 单 spec 单议题风格一致

### Q2：Anthropic prompt cache 策略

**Decision**：B — 弱 cache（仅观测）。

**Rationale**：
- spot-check 发现 `apply_cache_control()` 已生产，对整段 SystemMessage 标 ephemeral
- 但 5min TTL vs 1h cycle frequency → 跨 cycle 命中率 ≈ 0
- 整段 SystemMessage 含 dynamic snapshot/portfolio → 即便 TTL 内也只命中前缀共有部分（spec 016 P5 autogen 教训）
- **不**贸然重构 head/tail 拆分 + 1h beta cache，先量化现状（cache_read_input_tokens / cache_creation_input_tokens）作为决策依据
- 升级到强 cache 留待 020b 或独立 spec

### Q3：跨 spec advisory 收尾

**Decision**：A — 全 3 个打包。

**Rationale**：
- 3 个全是单文件 surgical fix，打包不增复杂度
- spec 020a 收尾性质，advisory 收尾天然契合
- 拆独立 PR 反而增加 review 开销

包含：
- spec 018 P2-1：IVE `llm.invoke()` → `await llm.ainvoke()`（消除 event loop 阻塞 3-5s）
- spec 019 P2-02：SkillsGrid 加 `triggers_keywords` badges
- spec 019 P2-03：`propose_new_skill` LLM 失败时 `.draft` frontmatter 写 `inference_failed: true`

### Q4：部署清单 + 数据迁移顺序

**Decision**：A — staging 验证脚本 + rollback runbook 都做。

**Rationale**：
- trilogy 涉及 3 个 migration 脚本（017b 配置文件 / 018 patterns+cases / 019 skill frontmatter），手敲容易遗漏顺序
- rollback runbook 是生产部署前置 SLA 要求
- 一次性投入换长期可重复

### Q5：Monitoring + Alerting

**Decision**：C — 仅 2 个核心指标（LLM cache hit rate + IVE 失败率），加 dashboard 不加告警。

**Rationale**：
- 2 个指标分别锚定 Q2 和 Q3 advisory，与本 spec 决策直接相关
- 加告警容易 alert fatigue（trilogy 还在 stabilize 期）
- 5 个全套指标待 020b daemon 落地后一并加

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | IVE `classify_case` sync `llm.invoke` | ✓ src/cryptotrader/learning/evolution/ive.py:247 确认 sync 调用，FR-Z10 改为 await ainvoke |
| 2 | SkillsGrid 缺 triggers_keywords | ✓ web/src/pages/memory/components/SkillsGrid.tsx 仅渲染 regime_tags，FR-Z11 加 triggers_keywords badge row |
| 3 | skill_metadata_inference 缺 failure flag | ✓ src/cryptotrader/learning/evolution/skill_metadata_inference.py except 兜底但未持久化 flag，FR-Z13 加 inference_failed 字段 |
| 4 | LLM cache usage 提取位置 | ⚠️ src/cryptotrader/agents/base.py:299 `log_llm_usage()` 已读 cache_read_input_tokens 写 structlog 但**(a)** 缺 cache_creation_input_tokens；**(b)** 未写 OTel span。FR-Z7~9 修订：补提取 + 写 OTel span attr 与 spec 010 trace 对齐 |

## 6 节速览

### 1. Purpose

Trilogy 收尾 — Anthropic prompt cache 观测 + 3 项 P2 advisory 修复 + 部署清单 + 2 项核心 monitoring。**不**含 daemon / lineage（→ 020b）。

### 2. User Stories

- **US-Z1（P1）Operator**：staging 验证脚本 1 命令跑全套 smoke check
- **US-Z2（P1）SRE**：rollback runbook 含 trilogy 每 spec 的 revert + DB 回退步骤
- **US-Z3（P1）Architect**：cache hit rate 写 OTel span，dashboard 可视
- **US-Z4（P1）Maintainer**：IVE 改 await，event loop 不再阻塞
- **US-Z5（P2）UI**：SkillsGrid `triggers_keywords` badges + frontmatter `inference_failed` 持久化

### 3. Functional Requirements（~18 条）

- **FR-Z1~3**：staging_validate 脚本（migrate dry-run + smoke + diff + telemetry 校验）
- **FR-Z4~6**：rollback runbook（git revert SHA + DB 回退 + 验证）
- **FR-Z7~9**：cache 观测（log_llm_usage 加 cache_creation_input_tokens 提取 + 写 OTel span attr `llm.cache.read_tokens` / `llm.cache.creation_tokens` / `llm.cache.hit_rate`）
- **FR-Z10**：IVE classify_case 改 async（await llm.ainvoke）
- **FR-Z11~12**：SkillsGrid triggers_keywords badges（最多 5 + `+N more`）
- **FR-Z13~14**：propose_new_skill LLM 失败时 frontmatter 写 inference_failed: true + 默认 metadata
- **FR-Z15~17**：dashboard 加 2 panel（cache hit rate / IVE 失败率），仅可视不告警

### 4. Success Criteria（~10 条）

- SC-Z1：scripts/staging_validate.py 一次成功 PASS
- SC-Z2：docs/rollback-trilogy.md 存在含 3 spec 回退路径
- SC-Z3：1 cycle 后 OTel trace 含 3 cache 字段
- SC-Z4：grep "llm.invoke" src/cryptotrader/learning/evolution/ive.py 返回空
- SC-Z5：grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx ≥ 1 hit
- SC-Z6：tests/test_skill_proposal_metadata_inference.py::test_llm_failure_writes_flag PASS
- SC-Z7：dashboard 2 panel manual smoke
- SC-Z8：spec 014/15/17b/18/19 既有测试不回归
- SC-Z9：/spex:review-spec 无 P0/P1
- SC-Z10：/spex:review-plan 任务覆盖完整

### 5. Dependencies

- **Upstream**：spec 010（OTel）/ 015（metrics endpoint）/ 017b / 018 / 019
- **Downstream**：spec 020b（daemon + lineage 复用本 spec rollback runbook 模板）

### 6. Out of Scope

- ❌ Reflect daemon（D-ENG-01 → 020b）
- ❌ Git lineage 自动化（D-ENG-02 → 020b）
- ❌ 强 cache 拆分 head/tail + 1h beta cache（待 B 观测后定）
- ❌ 5 个 metrics 全套告警（仅 2 个，不 alert）
- ❌ 新 prompt 内容优化

## 落地约束

- 不破坏 spec 014/15/17a/17b/018/019 公开 API
- 不引入新 runtime 依赖
- 直接删旧不留 fallback（用户偏好延续）
- Markdown 简体中文
- 范围预估 3-5 天

## 衔接 020b

020a 落地后 020b 启动条件：
- 1 周 cache hit rate 数据可用 → 决定是否升级强 cache
- 2 周 IVE 失败率数据可用 → 决定是否加告警
- daemon 设计需基于 trilogy 1 月运行数据
