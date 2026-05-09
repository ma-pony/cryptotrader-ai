# Feature Specification: Skill Evolution（spec 019）

**Feature Branch**: `020-skill-evolution`
**Created**: 2026-05-09
**Status**: Draft
**Input**: User description: "skill-evolution — schema 升级 + EvolvingSkillProvider + D-RT-01 检索算法 + skill_proposal LLM 推断 + 前端 /memory Skills section"

> **目录命名说明**：本 spec 是 trilogy 切分后的 Skill 子域，逻辑名 "spec 019"。spec-kit 按递增分配到 020。文档内引用一律称 "spec 019"。

## Purpose

完成 trilogy 第 3 段的 Skill 子域：基于 spec 016 D-DS-01（schema 升级）+ D-RT-01（检索算法）+ D-MW-01（元数据），把 spec 014 / 017a/b 的 `DefaultSkillProvider` 升级为可进化的 `EvolvingSkillProvider`。

3 块整合：

1. **Schema 升级 + 数据迁移**：
   - `agent_skills/<name>/SKILL.md` frontmatter 加 6 字段（regime_tags / triggers_keywords / importance / access_count / last_accessed_at / confidence）
   - `Skill` dataclass 同步加字段（全 default 兼容旧实例）
   - 一次性迁移脚本 `scripts/migrate_018_to_019.py`：5 个现有 skill 用硬编码 mapping（基于 brainstorm 阶段 LLM 分析得出 — 见下文 Functional Requirements FR-W3）
   - spec 014 既有 `learning/skill_proposal.py:propose_new_skill()` 改造：创建 `.draft` 时调 LLM 推断 metadata 写入 frontmatter（用户 manual save 后 metadata 已就位）

2. **EvolvingSkillProvider + 检索算法升级**：
   - 替换 spec 017a 的 `DefaultSkillProvider`（其简单 scope filter 退役）
   - 实现 spec 017a 既有 `SkillProvider` Protocol（鸭子类型）
   - **第一层过滤**：scope 匹配（沿用 spec 014 既有 `discover_skills_for_agent`）+ regime_tags 预过滤（空 list 视为 match all 向后兼容）
   - **第二层排序**：`score = (idf_score + importance + recency_bonus) × confidence`
     - IDF：基于 triggers_keywords 倒频率（在当前 skill 集 + snapshot 关键词集合中计算）；triggers_keywords=[] 时 idf_score=0
     - recency_bonus = `exp(-(now - last_accessed_at).total_seconds() / (7 × 86400))`
     - 取 top-k（默认 5）
   - access_count / last_accessed_at 自动累计回写
   - 同 spec 018 的 nodes/agents.py module-level singleton 集成（与 EvolvingMemoryProvider 并存）

3. **load_skill_tool 改造 + 前端集成**：
   - `_make_load_skill_tool(provider)` factory 接受 `provider` 参数；tool 内部调 `provider.get_skill_by_name(name)` 替代 `load_skill(name)`
   - access_count 走统一 IO 入口（与 retrieval 路径一致）
   - 前端 `/memory` 页面（spec 018 已建 3 sections：RulesGrid + 2-col grid {CasesTimeline | RecentTransitions} + ArchivedRules）**在最后加 SkillsGrid 单行 section（变 4 sections）**
   - 4 个新 API endpoints：
     - `GET /api/memory/skills?agent={id}` 返回 skill 列表 + 元数据
     - `GET /api/memory/skills/{name}` 返回 skill 详情 + body
     - `GET /api/memory/skill-access?since={iso}&agent={id}` 返回近期 access 事件
     - `GET /api/memory/skill-proposals?since={iso}` 返回 spec 014 propose_new_skill 创建 .draft 的历史

落地后：
- 4 agent 在 cycle prompt 中收到经 D-RT-01 算法排序的 top-k skill
- 新 skill `.draft` 创建时 LLM 自动填充 regime_tags / triggers_keywords / importance / confidence
- spec 020 daemon 接入点就绪（access_count 进化数据已积累）

本 spec **直接删旧不留 fallback**：DefaultSkillProvider 退役；任何步骤失败时返回空 list + warning log（cycle 不 break）。回滚走 git revert。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - EvolvingSkillProvider 真正接入 4 agent (Priority: P1) 🎯 MVP

作为架构师，spec 017a/b/18 落地的 PromptBuilder + Provider 协议真正用上"会进化的 skill"——4 agent 在每个 cycle 的 prompt 中收到的 `available_skills` section 是经 D-RT-01 算法排序的 top-k skill。

**Why P1**：spec 017b 的 DefaultSkillProvider 只做 scope filter，不排序也不进化。本 spec 让 spec 016 D-RT-01 真落地。

**Independent Test**：跑 1 mocked cycle → 4 agent 各自的 SystemMessage `available_skills` section 含按 IDF + importance + recency 排序的 top-k skill body；access_count 在 retrieval 后回写文件。

**Acceptance Scenarios**：
1. **Given** `agent_skills/` 含 5 skill（4 agent-specific + 1 shared），**When** EvolvingSkillProvider.get_available_skills(agent_id="tech")，**Then** 返回 list 含 tech-analysis + trading-knowledge 共 2 skill，按算法排序
2. **Given** EvolvingSkillProvider.get_available_skills 调用，**When** 返回 list 后，**Then** 对应 SKILL.md 文件 frontmatter `access_count` +1 / `last_accessed_at = now()`
3. **Given** Provider 内部任一步骤失败（IDF / IO / 排序），**When** 调用方调用，**Then** 返回空 list + warning log；cycle 不 break

---

### User Story 2 - Schema 升级 + 数据迁移零中断 (Priority: P1)

作为架构师，5 个现有 SKILL.md frontmatter 加 6 字段后向后兼容；spec 014 既有 `propose_new_skill` 创建 `.draft` 时也写新字段（不再 schema 不一致）。

**Why P1**：避免 spec 014 既有 propose_new_skill 触发创建新 skill `.draft` 时缺字段，破坏 EvolvingSkillProvider 排序逻辑。

**Independent Test**：跑 `scripts/migrate_018_to_019.py` 后 5 个 SKILL.md 全含 6 新字段；模拟 propose_new_skill → `.draft` 文件含 LLM 推断的 regime_tags + triggers_keywords。

**Acceptance Scenarios**：
1. **Given** `agent_skills/chain-analysis/SKILL.md` 不含新字段，**When** 迁移脚本跑，**Then** frontmatter 加上预先定义的 mapping（regime_tags=[] / triggers_keywords=funding rate/exchange flow/whale/.../ importance=0.7 / confidence=0.7 / access_count=0 / last_accessed_at=mtime）
2. **Given** spec 014 既有 `propose_new_skill()` 触发，**When** 创建 `.draft` 文件，**Then** `.draft` frontmatter 含 LLM 推断的 regime_tags + triggers_keywords（mock LLM 返回 fixture 值）+ 默认 importance=0.5 / confidence=0.5
3. **Given** 迁移脚本是幂等的，**When** 重复跑 2 次，**Then** 文件内容一致
4. **Given** 迁移脚本支持 `--dry-run`，**When** 加 flag 跑，**Then** 输出预览不修改文件
5. **Given** 未知 skill name（不在 5 个硬编码 mapping 中），**When** 迁移脚本处理，**Then** 加默认空字段（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）

---

### User Story 3 - D-RT-01 检索算法可观测 (Priority: P2)

作为后续维护者，每次 cycle 的 EvolvingSkillProvider.get_available_skills 调用要在 OpenTelemetry trace 写下排序明细。

**Why P2**：D-RT-01 算法多组件（IDF / importance / recency / confidence），故障调试需要看到中间结果。

**Independent Test**：跑 1 mocked cycle → trace 含 `skill.retrieval.candidates_after_regime_filter` / `skill.retrieval.top_k_with_scores` 等 attribute。

**Acceptance Scenarios**：
1. **Given** 1 cycle 完成，**When** trace 后端查询，**Then** 含 `skill.retrieval.candidates_after_regime_filter`（list[skill_name]）
2. **Given** 第二层排序完成，**When** trace 查询，**Then** 含 `skill.retrieval.top_k_with_scores`（list[{name, score, idf_component, importance_component, recency_component}]）
3. **Given** 1 skill 因 regime_tags 不匹配被过滤，**When** trace 查询，**Then** `skill.retrieval.filtered_out` 含 (skill_name, reason="regime_tags mismatch")

---

### User Story 4 - load_skill_tool 走 Provider 一致性 (Priority: P2)

作为后续维护者，LangChain `load_skill_tool` 调用要走 `EvolvingSkillProvider.get_skill_by_name(name)` 替代直接 `load_skill(name)`，让 access_count 等元数据在 tool 调用和 retrieval 调用中累计一致。

**Why P2**：D-MW-01 的 access_count 是 D-RT-01 recency_bonus 的关键输入；双 IO 路径会让进化数据失真。

**Independent Test**：mock LLM 调 `load_skill_tool(name="chain-analysis")` → access_count +1；调 `EvolvingSkillProvider.get_available_skills` 也 +1；两条路径的 access_count 计数一致。

**Acceptance Scenarios**：
1. **Given** `EvolvingSkillProvider.get_skill_by_name("tech-analysis")` 调用，**When** 返回 Skill 对象，**Then** access_count +1 + last_accessed_at = now() 写回文件
2. **Given** `_make_load_skill_tool(provider)` 注入 EvolvingSkillProvider，**When** load_skill_tool 被 LLM 调用，**Then** 内部走 `provider.get_skill_by_name(name)`，access_count 累计一致

---

### User Story 5 - skill_proposal LLM 推断元数据 (Priority: P1)

作为后续维护者，spec 014 既有 `propose_new_skill()` 创建 `.draft` 文件时调一次 LLM 推断 regime_tags + triggers_keywords + 初始 importance + confidence，写入 `.draft` frontmatter。用户 manual save 后 metadata 已就位。

**Why P1**：避免 schema 不一致；让新 skill 立刻可被 D-RT-01 算法正确排序。

**Independent Test**：mock LLM 返回 fixture metadata；跑 propose_new_skill → 新 `.draft` frontmatter 含 LLM 输出的字段值。

**Acceptance Scenarios**：
1. **Given** spec 014 既有 propose_new_skill 流程触发创建 .draft `dydx-funding-arbitrage`，**When** 调 LLM 推断 metadata，**Then** `.draft` frontmatter 含 LLM 返回的 regime_tags=[high_funding] / triggers_keywords=[funding, dydx, arbitrage] / importance=0.6 / confidence=0.6
2. **Given** LLM 推断失败（mock 异常），**When** propose_new_skill 完成，**Then** `.draft` 写入默认值（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）+ warning log
3. **Given** LLM 输出非合法 JSON，**When** parse 失败，**Then** 重试 1 次后回退默认值
4. **Given** 用户 review .draft 后 manual save → 变 SKILL.md，**Then** SKILL.md 已含 LLM 推断的 metadata（无需再次调 LLM）

---

### User Story 6 - /memory 页面 Skills section (Priority: P1)

作为运维 / reviewer，spec 018 已建 `/memory` 页面（3 sections：RulesGrid 单行 + 2-col grid {CasesTimeline | RecentTransitions} + ArchivedRules 单行）；本 spec **在最后加 SkillsGrid 单行 section**（变 4 sections 总）。

**Why P1**：用户明确"前端可视"诉求；本 spec 集成到现有 /memory 而非新建独立页面。

**Independent Test**：跑 1 mocked cycle 后 web UI 访问 `/memory` → 末端 Skills section 显示 5 skill 列表 + access counts + 最近 transitions。

**Acceptance Scenarios**：
1. **Given** `/memory` 页面加载，**When** 用户访问，**Then** 在 ArchivedRules 之后显示 "Skills Grid" section，含 4 agent 子分区 + 1 shared 子分区，每格显示 skill name / scope / importance / access_count / last_accessed_at
2. **Given** 调 `GET /api/memory/skills?agent=tech`，**Then** 返回 200 + JSON list
3. **Given** spec 014 既有 propose_new_skill 创建 .draft，**When** 访问 Skill Proposals 区，**Then** 显示 proposal 历史含 LLM 推断的 metadata + draft_path
4. **Given** Sidebar 含 8 路由（spec 018 落地后），**When** spec 019 落地，**Then** Sidebar 不变（不加新路由，仅扩展 /memory 页面）

---

### Edge Cases

- `agent_skills/` 为空 → EvolvingSkillProvider 返回 empty list；available_skills section 占位"暂无可用技能"
- skill 含 `regime_tags=[]` → 视为 match all（向后兼容 spec 014/15/17b 全注入语义）
- skill 含 `triggers_keywords=[]` → IDF 评分 = 0（不参与第二层排序，仅 importance × recency × confidence）
- LLM 推断 metadata 失败 → 默认值 + warning log
- IDF 算法失败（如关键词分词错误）→ 跳过该 skill 的 IDF 分数（视为 0）
- ToolAgent.tools 含 load_skill_tool，但 LLM 用 wrong skill_name 调用 → 返回 None + log warning
- 同 cycle 4 agent 各自调 get_available_skills → access_count 各自 +1（同一 skill 可能在 cycle 内 +1 多次）
- 进程内未在 OpenTelemetry tracing 上下文 → telemetry 字段降级到 structured log
- 用户编辑 SKILL.md 后未跑迁移 → spec 014 IO 沿用既有兜底（dataclass default 兼容）
- propose_new_skill 创建 `.draft` 时 LLM 推断已成功 → 用户 manual save 时 metadata 已在 .draft，无需 spec 014 既有 logic 再次填默认值

## Requirements *(mandatory)*

### Functional Requirements

#### Schema & Migration

- **FR-W1**：`agent_skills/<name>/SKILL.md` frontmatter MUST 含 spec 014 既有字段（`name / description / scope / version / manually_edited`）+ 本 spec 新增 6 字段：
  - `regime_tags: list[str] = []`（空 list 视为 match all regime）
  - `triggers_keywords: list[str] = []`（IDF 输入；空 list 时 IDF 评分=0）
  - `importance: float = 0.5`（0.0-1.0）
  - `access_count: int = 0`
  - `last_accessed_at: ISO8601`（默认 file mtime）
  - `confidence: float = 0.5`（0.0-1.0）
- **FR-W2**：`src/cryptotrader/agents/skills/schema.py:Skill` dataclass MUST 同步加 6 字段，全部带 default 兼容旧实例
- **FR-W3**：`scripts/migrate_018_to_019.py`（NEW）MUST 实现一次性迁移，使用以下硬编码 mapping（基于 brainstorm 阶段 LLM 分析）：
  ```python
  SKILL_MIGRATION_DEFAULTS = {
      "chain-analysis": {
          "regime_tags": [],
          "triggers_keywords": ["funding rate", "exchange flow", "netflow",
                                "whale", "open interest", "OI", "liquidation",
                                "on-chain", "accumulation", "distribution",
                                "blockchain"],
          "importance": 0.7, "confidence": 0.7,
      },
      "macro-analysis": {
          "regime_tags": [],
          "triggers_keywords": ["fed", "dxy", "dollar index", "fear greed",
                                "etf", "vix", "s&p", "macro", "rate cut",
                                "rate hike", "cpi", "risk-on", "risk-off",
                                "sentiment"],
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
          "triggers_keywords": ["rsi", "macd", "sma", "moving average",
                                "bollinger", "atr", "chart", "trend", "momentum",
                                "breakout", "support", "resistance", "indicator"],
          "importance": 0.7, "confidence": 0.7,
      },
      "trading-knowledge": {  # shared scope
          "regime_tags": [],
          "triggers_keywords": ["funding", "regime", "spot", "perp", "perpetual",
                                "basis", "confidence", "calibration",
                                "attribution", "data sufficiency",
                                "microstructure"],
          "importance": 0.8, "confidence": 0.8,  # foundational
      },
  }
  ```
  未知 skill name → 默认空字段
- **FR-W4**：迁移脚本 MUST 是幂等的（重复跑不损坏；已有字段不覆盖）
- **FR-W5**：迁移脚本 MUST 支持 `--dry-run` 模式
- **FR-W6**：迁移脚本启动期 MUST print 备份建议

#### EvolvingSkillProvider

- **FR-W7**：`src/cryptotrader/learning/evolution/skill_provider.py:EvolvingSkillProvider`（NEW）MUST 实现 spec 017a 的 `SkillProvider` Protocol：
  - `get_available_skills(agent_id, snapshot, k=5) -> list[Skill]`
  - `get_skill_by_name(name) -> Skill | None`（FR-W12 / load_skill_tool 用）
- **FR-W8**：`get_available_skills()` 内部 MUST 实现 D-RT-01 两层算法：
  - **第一层**：scope 匹配（沿用 `discover_skills_for_agent`）+ regime_tags 预过滤（空 list = match all；非空时 `current_regime in skill.regime_tags`）
  - **第二层**：每 candidate score = `(idf_score + importance + recency_bonus) × confidence`
    - `idf_score`：基于 triggers_keywords 倒频率；triggers_keywords=[] 时 idf_score=0
    - `recency_bonus = exp(-(now - last_accessed_at).total_seconds() / (7 × 86400))`
    - 取 top-k
  - 写回 `access_count += 1` + `last_accessed_at = now()`
- **FR-W9**：`get_available_skills()` MUST 在内部任一步骤异常时 catch + log warning + 返回空 list（不抛异常）
- **FR-W10**：`get_skill_by_name(name)` MUST：
  - 扫所有 SKILL.md 找 name 匹配
  - access_count += 1 + last_accessed_at = now() 回写
  - 返回 Skill 或 None
- **FR-W11**：`src/cryptotrader/agents/prompt_builder.py:DefaultSkillProvider` class MUST 删除
- **FR-W12**：`src/cryptotrader/nodes/agents.py:_get_or_build_pb` MUST 把 `_skill_provider` 替换为 `EvolvingSkillProvider`（spec 018 的 `_memory_provider` 仍 EvolvingMemoryProvider 不动）

#### load_skill_tool 改造（factory 模式）

- **FR-W13**：`src/cryptotrader/agents/skills/tool.py:_make_load_skill_tool` MUST 接受 `provider: EvolvingSkillProvider | None = None` 参数；tool 内部调 `provider.get_skill_by_name(name)`（如 provider 非空），不再调 `load_skill(name)`
- **FR-W14**：`load_skill_tool` module-level 实例 MUST 在 `nodes/agents.py` wire 时通过 `_make_load_skill_tool(_skill_provider)` 重新创建（注入 EvolvingSkillProvider singleton）；保持 ToolAgent.tools 注入路径不变
- **FR-W15**：`load_skill_tool` 失败时（provider 异常 / skill 找不到）返回 None + log warning

#### skill_proposal LLM 推断（写入 .draft）

- **FR-W16**：`src/cryptotrader/learning/skill_proposal.py:propose_new_skill()` 创建 `.draft` 文件时 MUST：
  - 调 LLM 一次推断 metadata（regime_tags + triggers_keywords + 初始 importance + 初始 confidence）
  - LLM 输出 JSON：`{"regime_tags": [...], "triggers_keywords": [...], "importance": 0.0-1.0, "confidence": 0.0-1.0}`
  - 写入 `.draft` frontmatter
  - 用户 manual save 后 metadata 已就位（无需再次推断）
- **FR-W17**：LLM 推断 prompt 含：
  - 新 skill 的 name + description + body 摘要
  - spec 014 既有 regime taxonomy（high_funding / negative_funding / high_vol / low_vol / trending_up / trending_down / extreme_fear / extreme_greed）
  - 现有 5 skill 的 mapping 作为示例（FR-W3 字典内容）
- **FR-W18**：LLM 调用失败 / 输出非合法 JSON / 重试 1 次后仍失败 → 写默认值（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）+ warning log
- **FR-W19**：LLM 推断 prompt 模板 + parse 逻辑放在 `src/cryptotrader/learning/evolution/skill_metadata_inference.py`（NEW，~150 行）

#### API（4 个新 endpoints）

- **FR-W20**：`src/api/routes/memory.py` 已含 spec 018 的 4 endpoints；本 spec 在同文件**扩展**加 4 endpoints：
  - `GET /api/memory/skills?agent={id}` 返回 list[Skill summary]（含 6 新字段）
  - `GET /api/memory/skills/{name}` 返回 skill 详情 + body
  - `GET /api/memory/skill-access?since={iso}&agent={id}` 返回近期 access 事件
  - `GET /api/memory/skill-proposals?since={iso}` 返回 propose_new_skill 创建的 .draft 历史
- **FR-W21**：API 错误返回沿用 spec 018 模式（400 / 404 / 500 + structured error JSON）
- **FR-W22**：API 鉴权沿用 spec 015 既有 X-API-Key

#### 前端 /memory 页面 Skills section

- **FR-W23**：`web/src/pages/memory/MemoryPage.tsx` MUST 在现有 3 sections（RulesGrid 单行 + 2-col grid {CasesTimeline | RecentTransitions} + ArchivedRules 单行）**之后**加第 4 单行 section "Skills Grid"
- **FR-W24**：新建 `web/src/pages/memory/components/SkillsGrid.tsx`（NEW，~200 行），渲染 4 agent + 1 shared 共 5 skill 子分区，每格显示：
  - skill name + scope
  - importance / access_count / last_accessed_at
  - regime_tags badges
  - triggers_keywords badges（前 3 个）
  - 点击展开 body
- **FR-W25**：`web/src/pages/memory/queries.ts` MUST 加 4 个新 React Query hooks（useSkills / useSkillByName / useSkillAccess / useSkillProposals）
- **FR-W26**：`web/src/i18n/{zh-CN,en-US}.ts` 或对应 i18n 文件（spec 018 实际可能放 `web/src/locales/zh-CN/memory.json`）MUST 加 Skills section 文案
- **FR-W27**：`web/src/components/layout/sidebar.tsx` 不变（不加新路由）

#### Telemetry & Observability

- **FR-W28**：每次 `EvolvingSkillProvider.get_available_skills` 调用 MUST 写以下 OpenTelemetry span attributes：
  - `skill.retrieval.candidates_after_regime_filter` (list[str])
  - `skill.retrieval.top_k_with_scores` (list[dict]，每项 `{name, score, idf_component, importance_component, recency_component}`)
  - `skill.retrieval.filtered_out` (list[dict]，每项 `{name, reason}`)
  - `skill.retrieval.duration_ms` (float)
- **FR-W29**：每次 `propose_new_skill` 调用 MUST 写：
  - `skill.proposal.name` (str)
  - `skill.proposal.draft_path` (str — `.draft` 文件路径)
  - `skill.proposal.llm_inferred_regime_tags` (list[str])
  - `skill.proposal.llm_inferred_triggers_keywords` (list[str])
  - `skill.proposal.llm_inferred_importance` (float)
  - `skill.proposal.llm_inferred_confidence` (float)
  - `skill.proposal.llm_call_failed` (bool)

#### Migration Tooling

- **FR-W30**：`scripts/migrate_018_to_019.py` MUST 在 spec 019 落地前手动运行
- **FR-W31**：迁移脚本 MUST 单测覆盖（`tests/test_migrate_018_to_019.py`）
- **FR-W32**：迁移脚本 MUST 输出迁移日志 + 失败行的 audit trail

### Key Entities

- **Skill (sec 014 既有 + 本 spec 扩展)**：spec 014 在 `src/cryptotrader/agents/skills/schema.py:46`；本 spec 加 6 字段 default
- **SkillProvider Protocol**：spec 017a 既有；本 spec 不修改 Protocol；EvolvingSkillProvider 实现它
- **EvolvingSkillProvider**：本 spec 新增 class，含 D-RT-01 两层检索算法
- **load_skill_tool**：spec 014 既有 LangChain tool（factory `_make_load_skill_tool` 模式）；本 spec 加 provider 参数
- **propose_new_skill**：spec 014 既有函数（`learning/skill_proposal.py`），写 `.draft` 文件；本 spec 加 LLM metadata 推断
- **PromptBuilderSingleton**：spec 017b/18 既有 module-level dict in `nodes/agents.py`；本 spec 把 `_skill_provider` 替换为 EvolvingSkillProvider

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-W1**：`scripts/migrate_018_to_019.py` 跑完后 5 个 SKILL.md frontmatter 全部含 6 新字段
- **SC-W2**：5 个已知 skill 含预期 mapping（chain/macro/news/tech-analysis 的 importance=0.7 / confidence=0.7；trading-knowledge 的 importance=0.8 / confidence=0.8；triggers_keywords 含 FR-W3 列出的关键词）
- **SC-W3**：`tests/test_migrate_018_to_019.py` ≥ 8 用例 PASS
- **SC-W4**：`tests/test_evolving_skill_provider.py` ≥ 12 用例 PASS（含 D-RT-01 两层算法 / IDF / recency / 排序 / access_count 回写 / 异常容错）
- **SC-W5**：`grep -rn "class DefaultSkillProvider" src/cryptotrader/` 返回空（spec 017a/b 实现退役）
- **SC-W6**：`nodes/agents.py:_get_or_build_pb` 返回的 PromptBuilder 的 `_skill_provider` MUST 是 `EvolvingSkillProvider`；`_memory_provider` 仍 `EvolvingMemoryProvider`（spec 018 不动）
- **SC-W7**：`tests/test_load_skill_tool.py` ≥ 4 用例 PASS（factory 注入 + access_count + 异常）
- **SC-W8**：`grep -rn "open(.*SKILL.md" src/cryptotrader/agents/skills/tool.py` 返回空（不再直接读文件，全走 Provider）
- **SC-W9**：`tests/test_skill_proposal_metadata_inference.py` ≥ 6 用例 PASS（LLM 成功 / 失败 / 重试 / 默认值兜底 / .draft 写入）
- **SC-W10**：spec 014 既有 `tests/test_skill_proposal.py` 不回归
- **SC-W11**：`tests/test_api_memory_skills.py` ≥ 8 用例 PASS（4 endpoints 各覆盖正常 + 错误情况）
- **SC-W12**：`tests/web/test_memory_page.tsx`（spec 018 既有）扩展 ≥ 4 新用例 PASS（SkillsGrid 渲染 / 展开 body / regime_tags badges / Skill Proposals 区）
- **SC-W13**：`web/src/components/layout/sidebar.tsx` 不变（不加新路由）
- **SC-W14**：`tests/test_e2e_skill_evolution.py` 跑 1 mocked cycle 全链路 PASS
- **SC-W15**：现有 spec 014 / 15 / 17a / 17b / 18 测试不回归（基线 ≥ 2254 pass）
- **SC-W16**：通过 `/spex:review-spec` 无 P0 / P1 issues
- **SC-W17**：通过 `/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-W18**：通过 `/spex:review-code` 合规评分 ≥ 95% + Deep Review Report
- **SC-W19**：通过 `/spex:verification-before-completion` stamp gate（全套测试 ≥ 2300 pass / 0 fail）

## Assumptions

- spec 017a/b 公开 API（PromptBuilder.build / SkillProvider Protocol）签名稳定
- spec 018 EvolvingMemoryProvider 接入路径稳定，本 spec 在同 singleton 中并存
- 现有 5 SKILL.md 数据格式一致（spec 014 既有约定）
- `discover_skills_for_agent` 函数（spec 014 既有，spec 017b 修复 scope filter 后）可直接 reuse
- IDF 算法在 5 skill 集 + snapshot ~50 字段量级下耗时 < 50ms / 调用
- LLM 推断 metadata 调用 GPT-4o-mini ~500 token；spec 014 propose_new_skill 频率低（~1/day），月成本 ~$0.005
- spec 020 落地前 skill 集大概率维持 5-10 个

## Dependencies

**Upstream**：
- spec 017a / 017b / 018（已合并 main）
- spec 014 / 015 / 010 / 016

**Downstream**：
- spec 020（待立项，trilogy Ops 子域）—— 加 daemon 触发 skill 进化（importance 重计算 / stale 标记 / Anthropic cache / git lineage）

**External tooling**：无新依赖（IDF 用 pure Python，无 sklearn/nltk）

## Out of Scope

**移至 spec 020（Ops 子域）**：
- Skill 进化触发器（importance 重计算 / stale 标记）
- Anthropic prompt cache 配置
- Offline reflect daemon
- Git lineage 自动化
- Sentence-transformers embedding 接入（D-RT-01 第 3 组件，违反"不引入新依赖"）

**本 spec 显式不动**：
- spec 014 既有 `learning/curation.py` / `learning/memory.py` / `learning/regime.py`
- spec 018 既有 EvolvingMemoryProvider（同 singleton 并存）
- 4 agent 类（spec 017b 稳定）
- LangGraph 主图（access_count 在 retrieval 内自动累计）
- Maturity Literal（不加到 Skill）

## Reversibility

本 spec 落地后**部分可逆**：
- **可逆**：4 agent runtime 切换、API endpoints、前端 SkillsGrid section — git revert
- **半可逆**：SKILL.md frontmatter 6 新字段 — git revert 代码后字段仍在文件里，但 EvolvingSkillProvider 不再读
- **不可逆**：迁移脚本一次性写入 — 需反向脚本回滚

降低风险措施：
- 迁移脚本含 `--dry-run` + 幂等
- C3 atomic commit 单 commit 包含全部 EvolvingSkillProvider 切换；revert 该 commit 即返 spec 018 状态

## Implementation Outline

### 4 commit 单 PR

#### C1 — 数据迁移 + schema（无 behavior 变化）
- `scripts/migrate_018_to_019.py`（NEW，~300 行）
- `tests/test_migrate_018_to_019.py`（NEW，≥ 8 用例）
- `tests/fixtures/skills_old_format/`（NEW）
- `src/cryptotrader/agents/skills/schema.py` MODIFY — Skill 加 6 字段
预估 diff：~600 行

#### C2 — 算法层 + LLM 推断模块
- `src/cryptotrader/learning/evolution/idf.py`（NEW，~120 行）
- `src/cryptotrader/learning/evolution/skill_metadata_inference.py`（NEW，~150 行）
- `tests/test_idf.py` ≥ 6 用例
- `tests/test_skill_metadata_inference.py` ≥ 6 用例
预估 diff：~600 行

#### C3 — Provider + 集成（atomic 切换）
- `src/cryptotrader/learning/evolution/skill_provider.py`（NEW，~300 行）
- `src/cryptotrader/agents/prompt_builder.py` MODIFY — 删 DefaultSkillProvider class
- `src/cryptotrader/nodes/agents.py` MODIFY — 切到 EvolvingSkillProvider
- `src/cryptotrader/agents/skills/tool.py` MODIFY — `_make_load_skill_tool` 加 provider 参数
- `src/cryptotrader/learning/skill_proposal.py` MODIFY — propose_new_skill 加 LLM 推断
- `tests/test_evolving_skill_provider.py` ≥ 12 用例
- `tests/test_load_skill_tool.py` ≥ 4 用例
- `tests/test_skill_proposal_metadata_inference.py` ≥ 6 用例

⚠️ Atomic：必须含全部 Provider 切换 + tool 改造 + propose_new_skill 改造
预估 diff：~900 行

#### C4 — API + 前端 + E2E
- `src/api/routes/memory.py` MODIFY — 加 4 endpoints
- `tests/test_api_memory_skills.py` ≥ 8 用例
- `web/src/pages/memory/components/SkillsGrid.tsx`（NEW）
- `web/src/pages/memory/MemoryPage.tsx` MODIFY
- `web/src/pages/memory/queries.ts` MODIFY
- `web/src/locales/{zh-CN,en-US}/memory.json` MODIFY
- `tests/web/test_memory_page.tsx` MODIFY ≥ 4 新用例
- `tests/test_e2e_skill_evolution.py`（NEW）

预估 diff：~700 行

### 任务总数
约 50 task。具体由 `/speckit-tasks` 生成。

### 估时
| 阶段 | 工作量 |
|---|---|
| C1（迁移 + schema） | 0.5 天 |
| C2（IDF + LLM 推断） | 1.5 天 |
| C3（Provider 集成 atomic） | 1.5 天 |
| C4（API + 前端 + E2E） | 1.5 天 |
| Code review 修复 + dry-run 验证 | 1 天 |
| **合计** | **6 天** |

### Migration Strategy（生产环境）
1. 部署前在 staging 跑 `python scripts/migrate_018_to_019.py --dry-run`
2. 备份 `agent_skills/` 整目录
3. 实跑迁移脚本
4. 部署 spec 019 代码（git pull + scheduler restart）
5. 监控第 1 cycle：`/api/memory/skills` 返回正确数据；`skill.retrieval.*` telemetry 字段
6. 监控第 24h：access_count 累计；如有 propose_new_skill 触发，验证 LLM 推断 metadata 写入 .draft
7. 若发生回退：`git revert <C3 commit>` + 把备份的 `agent_skills/` 复原
