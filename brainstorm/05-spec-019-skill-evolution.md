# Brainstorm: Spec 019 — Skill Evolution

**Date:** 2026-05-09
**Status:** active
**Spec:** （待 `/speckit-specify` 创建，目录 `specs/020-...`，文档内引用为 "spec 019"）

## Problem Framing

trilogy 第 3 段 Skill 子域。承接：
- spec 016（8 项目研究，已合并 main）
- spec 017a（PromptBuilder 基建，已合并 main）
- spec 017b（4 agent 集成切换，已合并 main）
- spec 018（Memory Evolution，已合并 main）

基于 spec 016 D-DS-01（schema 升级）+ D-RT-01（检索算法）+ D-MW-01（元数据），把 spec 014 / 017a/b 的 `DefaultSkillProvider` 升级为 `EvolvingSkillProvider`。

3 块整合：
1. **Schema 升级 + 数据迁移** — 5 个现有 SKILL.md frontmatter 加 6 字段，硬编码 mapping（基于 brainstorm 阶段 LLM 分析）
2. **EvolvingSkillProvider + D-RT-01 检索算法** — regime_tags 预过滤 + IDF 加权（不含 sentence-transformers embedding，违反约束）
3. **load_skill_tool 改造 + skill_proposal LLM 推断 + 前端 /memory Skills section**

## 7 项关键设计决策

### Q1 — D-DS-01 schema 字段范围（spot-check 修订）

**选项**：A 全采纳 17 字段 / B 子集采纳 + Skill 适配 / C 重新设计
**决策**：**B 子集采纳**

**关键发现**：spec 016 D-DS-01 列的 17 字段大多适合 **PatternRecord**（spec 018 territory），不全适合 **Skill**：
- ✅ 适合 skill：regime_tags / triggers_keywords / importance / access_count / last_accessed_at / confidence
- ❌ 不适合 skill：maturity FSM / pnl_track / applied_count / forbidden / allowed_actions（这些是 pattern-evolution 概念）

5 个现有 SKILL.md 是手写 agent role 知识库，不是 cycle-evolved patterns；不应有 PnL 跟踪等字段。

### Q2 — D-RT-01 检索算法范围

**选项**：A 完整 / B 子集（无 embedding）/ C MVP（仅 regime + 元数据）
**决策**：**B D-RT-01 子集（无 embedding）**

**理由**：embedding 需要 sentence-transformers 依赖，违反"不引入新依赖"约束；IDF 算法纯 Python 可实施，且为 skill 集增长（spec 014 curation / spec 020 daemon 自动扩展）铺路。

**算法**：
- 第一层 scope + regime_tags 预过滤（空 list 视为 match all）
- 第二层 score = `(idf + importance + recency_bonus) × confidence`，取 top-k

### Q3 — load_skill_tool 决策

**选项**：A 保留 / B 改造（走 Provider）/ C 删除
**决策**：**B 改造**

**理由**：保持 access_count 元数据一致性；skill 集大时 LLM 按需 load 仍有价值。

### Q4 — Skill 进化触发器

**选项**：A 共用 spec 018 evaluate_node / B 独立节点 / C 仅 retrieval
**决策**：**C 仅 retrieval**

**理由**：access_count 在 retrieval 时自动累计；高级进化（importance 重计算 / stale 标记 / curation）属 Ops 层，推迟 spec 020 daemon。

### Q5 — 数据迁移

**选项**：A 默认值 / B LLM 辅助 / C 手动
**决策**：**B + Claude 直接做 LLM 工作**

我直接基于 5 个 SKILL.md 全文分析得出硬编码 mapping（无需迁移脚本调 LLM）：

```python
SKILL_MIGRATION_DEFAULTS = {
    "chain-analysis": {
        "regime_tags": [],  # match all
        "triggers_keywords": ["funding rate", "exchange flow", "netflow", "whale",
                              "open interest", "OI", "liquidation", "on-chain",
                              "accumulation", "distribution", "blockchain"],
        "importance": 0.7, "confidence": 0.7,
    },
    "macro-analysis": {
        "regime_tags": [],
        "triggers_keywords": ["fed", "dxy", "dollar index", "fear greed", "etf",
                              "vix", "s&p", "macro", "rate cut", "rate hike",
                              "cpi", "risk-on", "risk-off", "sentiment"],
        "importance": 0.7, "confidence": 0.7,
    },
    "news-analysis": {
        "regime_tags": [],
        "triggers_keywords": ["news", "headline", "regulatory", "etf approval",
                              "ban", "hack", "exploit", "social", "sentiment",
                              "twitter", "catalyst"],
        "importance": 0.6, "confidence": 0.6,
    },
    "tech-analysis": {
        "regime_tags": [],
        "triggers_keywords": ["rsi", "macd", "sma", "moving average", "bollinger",
                              "atr", "chart", "trend", "momentum", "breakout",
                              "support", "resistance", "indicator"],
        "importance": 0.7, "confidence": 0.7,
    },
    "trading-knowledge": {  # shared scope
        "regime_tags": [],
        "triggers_keywords": ["funding", "regime", "spot", "perp", "perpetual",
                              "basis", "confidence", "calibration", "attribution",
                              "data sufficiency", "microstructure"],
        "importance": 0.8, "confidence": 0.8,  # foundational
    },
}
```

### Q6 — spec 014 既有 skill_proposal 改造

**选项**：A 完整改造（LLM 推断）/ B 最小改造（默认值）/ C 不改
**决策**：**A 完整改造（LLM 推断 metadata）**

**理由**：避免新 skill 创建时 schema 不一致；LLM 推断成本可忽略（~$0.005/月）。

### Q7 — 前端可视化策略

**选项**：A 扩展 /memory + Skills section / B 新建 /skills 页面 / C 仅 API
**决策**：**A 扩展 /memory**

**理由**：5 skill 信息密度独立页面太空；与 spec 018 一致性高。

## 6 段 Spec Outline

### Section 1 — Purpose
EvolvingSkillProvider 替换 spec 017a/b DefaultSkillProvider；D-RT-01 算法落地；skill_proposal LLM 推断；前端 /memory 加 Skills section。

### Section 2 — User Stories
- **US-W1（Architect）— EvolvingSkillProvider 真正接入 4 agent（P1，MVP）**
- **US-W2（Architect）— Schema 升级 + 数据迁移零中断（P1）**
- **US-W3（Maintainer）— D-RT-01 检索算法可观测（P2）**
- **US-W4（Maintainer）— load_skill_tool 走 Provider 一致性（P2）**
- **US-W5（Maintainer）— skill_proposal LLM 推断元数据（P1）**
- **US-W6（Operator/Reviewer）— /memory 页面 Skills section（P1）**

### Section 3 — Functional Requirements
32 条 FR-W1..W32，分 8 子模块：
- Schema & Migration（FR-W1..W6）
- EvolvingSkillProvider（FR-W7..W12）
- load_skill_tool 改造（FR-W13..W15）
- skill_proposal LLM 推断（FR-W16..W19）
- API 4 endpoints（FR-W20..W22）
- 前端 /memory Skills section（FR-W23..W27）
- Telemetry（FR-W28..W29）
- Migration Tooling（FR-W30..W32）

### Section 4 — Success Criteria
19 条 SC-W1..W19，覆盖 schema migration / EvolvingSkillProvider / load_skill_tool / skill_proposal / API + frontend / E2E + 回归 / quality gate。

### Section 5 — Dependencies & Out of Scope
**Upstream**：spec 017a/b/14/15/10/16/18
**Downstream**：spec 020（Ops 子域）
**移至 spec 020**：skill 进化触发器 / Anthropic prompt cache / git lineage / sentence-transformers embedding

### Section 6 — Implementation Outline
4 commit 单 PR，~50 task，~6 天工作量：
- C1：迁移工具 + schema 字段（~600 行）
- C2：IDF + LLM 推断模块（~600 行）
- C3：Provider + 集成 atomic（~900 行）
- C4：API + 前端 + E2E（~700 行）

## Approaches Considered

每个 Q 都列了 3-5 个 alternative。整体架构哲学：
- "explicit > magic"：硬编码 5 skill mapping 而非全自动 LLM 迁移
- "isolation > coupling"：EvolvingSkillProvider 与 EvolvingMemoryProvider 并存于同一 singleton
- "no fallback"：DefaultSkillProvider 退役（不留 spec 017b 路径）
- "scope-driven evolution"：本 spec 仅 retrieval；spec 020 加 daemon 触发高级进化

## Decision

按 7 项决策落地：B / B / B / C / B + Claude / A / A。整合范围：~15 文件，~2800 行 diff，4 commit 单 PR。

## Open Threads（已 spot-check 解决，2026-05-09）

### ✅ Thread 1：discover_skills_for_agent 签名

`(agent_id: str, skill_dir: Path | None = None) -> list[Skill]`（spec 014 既有）。EvolvingSkillProvider 第一层 reuse 安全。

### 🔴 Thread 2：load_skill_tool 是 factory 模式

实际：
```python
def _make_load_skill_tool(skill_dir=None):
    @tool
    def load_skill_tool(name: str) -> str:
        result = load_skill(name, skill_dir=skill_dir)
        ...
load_skill_tool = _make_load_skill_tool()  # module-level instance
```

**spec 修订**：FR-W13 改造方式 = 让 factory `_make_load_skill_tool()` 接受 `provider` 参数；tool 内部调 `provider.get_skill_by_name(name)` 替代 `load_skill(name)`。Module-level instance 在 nodes/agents.py wire 时传入 EvolvingSkillProvider singleton。

### 🔴 Thread 3：skill_proposal 实际函数名 + 工作流

**spec 014 实际**：
- 函数名是 `propose_new_skill`（不是 `propose_skill`）
- 输出到 stdout 或 `<name>/SKILL.md.draft`，**不自动创建 skill 文件**
- 需用户 review + manual save 才变正式 SKILL.md

**spec 修订**：
- FR-W16 改：`propose_new_skill()` 创建 `.draft` 时调 LLM 推断 metadata，写入 .draft frontmatter
- 用户 manual save 后 metadata 已就位（无需再次调 LLM）
- 新增 telemetry attribute `skill.proposal.draft_path` 表示 draft 写入位置

### 🔴 Thread 4：MemoryPage 实际 layout

**spec 018 实际**：
```
Section 1: RulesGrid (单行)
Section 2: 2-column grid {CasesTimeline | RecentTransitions}
Section 3: ArchivedRules (单行)
```

**spec 修订**：FR-W23 改为"在 ArchivedRules 之后加 SkillsGrid 单行 section（变 4 sections 总）"，保持 spec 018 既有 2-column layout 不破坏。

### ✅ Bonus — SkillProvider Protocol 签名一致

`get_available_skills(agent_id, snapshot, k=5)` 与 FR-W7 一致；spec 017a 注释 "spec 018 提供进化版实现" 是错位（实际 spec 018 是 MemoryProvider，spec 019 才是 SkillProvider 进化版）。落地时顺便修注释。

## Decision Updates（基于 spot-check 结果，2026-05-09）

| 字段 | 旧定义 | 新定义 |
|---|---|---|
| FR-W13 | load_skill_tool 内部直接调 provider | `_make_load_skill_tool(provider)` factory 注入 provider；tool 内部调 `provider.get_skill_by_name(name)` |
| FR-W16 | `propose_skill()` 创建新 skill 时调 LLM | `propose_new_skill()` 创建 .draft 时调 LLM；用户 manual save 后 metadata 就位 |
| FR-W23 | 加第 5 个 Section "Skills Grid" | 在现有 3 sections（含 2-col grid）之后加 SkillsGrid 单行 section（变 4 sections 总） |
| FR-W29 | telemetry 6 字段 | telemetry 7 字段（加 `skill.proposal.draft_path`） |

## 实施 hints（给 implement subagent）

- ⚠️ 必须先读 `src/cryptotrader/agents/skills/{schema,loader,tool}.py` + `src/cryptotrader/learning/skill_proposal.py` 获取**真实函数名 + 签名**，不要凭脑补
- `discover_skills_for_agent` 沿用作 scope 第一层过滤入口
- `load_skill_tool` 是 factory 模式（不是简单函数）
- `propose_new_skill` 写 `.draft` 文件（不是 `SKILL.md`）
- MemoryPage 既有结构是 1 + 2-col + 1 = 3 sections（含 grid layout），spec 019 加 4th
