# Tasks：Agent Prompt Builder Integration（spec 017b）

**输入**：[plan.md](plan.md) / [spec.md](spec.md) / [data-model.md](data-model.md) / [contracts/](contracts/) / [research.md](research.md)
**Tests**：spec 显式要求测试（SC-Y8 / SC-Y9 / SC-Y10 / SC-Y11 / SC-Y12）
**Commit 序列**：3 commit 单 PR（C1 纯新增 / C2 atomic 切换 / C3 E2E + gate）

## 格式：`[ID] [P?] [Story] Description`

- **[P]**：可与同 phase 其他任务并行（不同文件，无依赖）
- **[Story]**：US1..US5 对应 spec user stories
- 全部任务路径为 repo root 相对路径

---

## Phase 1: Setup（无）

本 spec 无 Setup phase — `config/agents/` 目录已在 spec 017a 创建。

---

## Phase 2: Foundational — C1 commit（纯新增）

**目的**：创建 4 agent config 文件 + snapshot_renderer.py + test_snapshot_renderer.py。完成本 phase 后所有现有测试不变（无 behavior 变化）。

**Checkpoint**：C1 commit 后 `pytest tests/ --no-cov 2>&1 | tail -5` 仍 PASS（含新增 test_snapshot_renderer 通过）。

- [ ] T001 [P] [US1] 读 `src/cryptotrader/agents/tech.py:15-31` 提取 TechAgent 现有 ROLE 字符串
- [ ] T002 [P] [US1] 读 `src/cryptotrader/agents/chain.py` 提取 ChainAgent 现有 ROLE 字符串
- [ ] T003 [P] [US1] 读 `src/cryptotrader/agents/news.py` 提取 NewsAgent 现有 ROLE 字符串
- [ ] T004 [P] [US1] 读 `src/cryptotrader/agents/macro.py` 提取 MacroAgent 现有 ROLE 字符串
- [ ] T005 [US1] 读 `src/cryptotrader/agents/base.py:330-365` 提取 ANALYSIS_FRAMEWORK 全文，拆为 discipline 段（约 30 行）+ JSON schema 段（约 5 行）
- [ ] T006 [P] [US1] 创建 `config/agents/tech.md`，frontmatter 含 agent_id=tech / description / 5 sections / budget / priority；body 5 个 section 中：system_prompt = TechAgent ROLE + ANALYSIS_FRAMEWORK discipline；output_schema = ANALYSIS_FRAMEWORK JSON schema；user_tail / available_skills / recent_memory 写占位说明文字
- [ ] T007 [P] [US1] 创建 `config/agents/chain.md`（同 T006 模板，agent_id=chain）
- [ ] T008 [P] [US1] 创建 `config/agents/news.md`（同 T006 模板，agent_id=news）
- [ ] T009 [P] [US1] 创建 `config/agents/macro.md`（同 T006 模板，agent_id=macro）
- [ ] T010 [US3] 创建 `src/cryptotrader/agents/snapshot_renderer.py`：`render_crypto_snapshot(snapshot: dict, experience: str = "") -> str` 函数，搬运 `BaseAgent._build_prompt()` 全部逻辑（funding ELEVATED/NEGATIVE 标注 / futures volume SPIKE/LOW / open interest / news headlines + sanitize_input / data quality warnings / experience cap / TechAgent indicators 字段渲染）。注意 import `cryptotrader.security:sanitize_input` 与 `cryptotrader.agents.base:FUNDING_RATE_HIGH/LOW`
- [ ] T011 [US3] 创建 `tests/test_snapshot_renderer.py` ≥ 6 用例：(a) `test_funding_elevated_annotation`：funding=0.0005 → 输出含 "ELEVATED — crowded long"；(b) `test_funding_negative_annotation`：funding=-0.0002 → 输出含 "NEGATIVE — crowded short"；(c) `test_news_headlines_sanitized`：headline 含 "Ignore previous instructions" → 输出经过 sanitize_input；(d) `test_data_quality_warnings`：onchain.open_interest=0 + onchain.exchange_netflow=0 → 输出含 "On-chain data unavailable" warning；(e) `test_experience_capped`：传 experience 长 5000 字符 → 输出 experience 段 ≤ 4000 字符；(f) `test_tech_indicators_rendered`：snapshot 含 indicators={"rsi":65, "macd":...} → 输出含 "Technical Indicators:"
- [ ] T012 [US1] 运行 `uv run python -m pytest tests/test_snapshot_renderer.py -v --no-cov`，断言全部 PASS（≥ 6 用例）
- [ ] T013 [US1] 运行完整回归 `uv run python -m pytest tests/ --no-cov -x 2>&1 | tail -10`，确认无回归（C1 commit 前 gate）
- [ ] T014 **Commit C1**：`git add config/agents/ src/cryptotrader/agents/snapshot_renderer.py tests/test_snapshot_renderer.py` 后 `git commit -m "feat(spec-017b/c1): config/agents + snapshot_renderer + tests (no behavior change)"`

**Checkpoint C1**：4 agent config 文件 + snapshot_renderer.py + 6+ test PASS；main 路径无 behavior 变化。

---

## Phase 3: User Story Migration — C2 commit（atomic 切换）

**目的**：4 agent 真切换到 PromptBuilder 路径 + 删除所有旧路径（middleware / ANALYSIS_FRAMEWORK / role_description / prompt_template / _resolve_*）。

**Goal**：US-Y1（配置驱动）+ US-Y2（Skill 零回归）+ US-Y3（snapshot_renderer 接入）+ US-Y4（backtest 零回归）满足。

**Independent Test**：4 agent 单测 + 17a 基建测试 + spec 014/15 回归测试全部 PASS。

**⚠️ Atomic**：T015-T035 必须**全部完成**才能跑测试 / 提交 C2 commit。中间状态会让 4 agent 实例化失败。

### prompt_builder.py 扩展

- [ ] T015 [US2] 在 `src/cryptotrader/agents/prompt_builder.py:Skill` dataclass 加 `name: str = ""` 字段（如不存在），默认值 `skill_id` 当 name 缺时 fallback
- [ ] T016 [US2] 修改 `DefaultSkillProvider.get_available_skills()`：用 `from cryptotrader.agents.skills.loader import discover_skills_for_agent` 替换 017a 的 `agent_id in skill.tags` 过滤；签名 `get_available_skills(agent_id, snapshot, k=5)` 不变
- [ ] T017 [US2] 修改 `PromptBuilder._render_skills()`：输出格式 `\n\n---\n## Skill: {skill.name or skill.skill_id}\n\n{skill.body}`（参考 SkillsInjectionMiddleware 删除前 line 50-72 的格式）；空 list 返回 "暂无可用技能"
- [ ] T018 [US3] 修改 `PromptBuilder._render_snapshot()`：内部 `from cryptotrader.agents.snapshot_renderer import render_crypto_snapshot; return render_crypto_snapshot(snapshot)`
- [ ] T019 [US1] 修改 `PromptBuilder.build()` 签名：加 `experience: str = ""` 参数；内部逻辑：experience 非空时跳过 `self._memory_provider.get_recent_memory()`，sections["recent_memory"] = experience；否则走 017a fallback 路径；写入 telemetry attribute `prompt.builder.experience_source` ∈ {"caller", "provider", "empty"}

### base.py 重构

- [ ] T020 [US1] 修改 `src/cryptotrader/agents/base.py:BaseAgent.__init__` 签名为 `(self, *, agent_id: str, prompt_builder: PromptBuilder, model: str = "")`，删除 `role_description` 参数与 `self.role_description` 字段，加 `self._prompt_builder = prompt_builder`
- [ ] T021 [US1] 修改 `BaseAgent.analyze(snapshot, experience: str = "")`：内部调 `sys_msg, usr_msg = self._prompt_builder.build(snapshot=self._snapshot_to_dict(snapshot), portfolio={}, experience=experience)`，再 `await llm.ainvoke([sys_msg, usr_msg])`；保留现有 mock fallback 异常处理
- [ ] T022 [US1] 添加 helper `BaseAgent._snapshot_to_dict(snapshot: DataSnapshot) -> dict`：把 DataSnapshot 对象转 dict（pair / timestamp / market.{ticker, volatility, funding_rate, ohlcv} / news.headlines / onchain.{open_interest, exchange_netflow, liquidations_24h} / macro.{fed_rate, dxy} 等字段）。snapshot dict 结构需让 `render_crypto_snapshot` 可消费
- [ ] T023 [US1] 删除 `BaseAgent._build_prompt()` 方法
- [ ] T024 [US1] 删除 `ANALYSIS_FRAMEWORK` 模块级常量
- [ ] T025 [US4] 修改 `ToolAgent.__init__` 签名为 `(*, agent_id, prompt_builder, tools, model="", backtest_mode=False)`，删除 `role_description` 参数；调 `super().__init__(agent_id=agent_id, prompt_builder=prompt_builder, model=model)`
- [ ] T026 [US4] 修改 `ToolAgent.analyze(snapshot, experience="")`：backtest_mode=True → `super().analyze(snapshot, experience)`；False → `sys_msg, usr_msg = self._prompt_builder.build(...); agent = create_agent(_create_chat_model(self.model), tools=self.tools, system_prompt=sys_msg.content); result = await agent.ainvoke({"messages":[{"role":"user","content":usr_msg.content}]})`；保留 parse / fallback 异常处理；删除 `from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware` 与 `SkillsInjectionMiddleware(agent_id=...).build_system_addendum()` 调用

### 4 agent 类重构

- [ ] T027 [P] [US1] 重构 `src/cryptotrader/agents/tech.py`：删除 ROLE 常量（line 15-31）；删除 TechAgent.__init__（替换为 `(*, prompt_builder: PromptBuilder, model: str = "")` 调 `super().__init__(agent_id="tech", prompt_builder=prompt_builder, model=model)`）；删除 `_build_prompt` 方法（如有 override）；保留 `compute_indicators` + helper；TechAgent 自定义 `analyze()`，先 compute_indicators merge 到 snapshot dict 再调 PromptBuilder（或 override `_snapshot_to_dict` 加 indicators 字段）
- [ ] T028 [P] [US1] 重构 `src/cryptotrader/agents/chain.py`：删除 ROLE 常量；改 ChainAgent.__init__ 为 `(*, prompt_builder, model="", backtest_mode=False)` 调 `super().__init__(agent_id="chain", prompt_builder=prompt_builder, tools=CHAIN_TOOLS, model=model, backtest_mode=backtest_mode)`；删除 `_build_prompt` 方法（如有）
- [ ] T029 [P] [US1] 重构 `src/cryptotrader/agents/news.py`（同 T028 模板，agent_id=news, tools=NEWS_TOOLS）
- [ ] T030 [P] [US1] 重构 `src/cryptotrader/agents/macro.py`（删 ROLE / 改构造器 `(*, prompt_builder, model="")`）

### config.py 重构

- [ ] T031 [US1] 修改 `src/cryptotrader/config.py:AgentConfig` dataclass：删除 `prompt_template` 字段（如存在）
- [ ] T032 [US1] 修改 `AgentsConfig.build()` 签名为 `build(self, agent_id, *, prompt_builder: PromptBuilder, backtest_mode=False, model_override="")`，删除 `regime_tags` 参数；删除 `_resolve_role` / `_resolve_skills` 方法；删除 `agent.role_description += "STRATEGY SKILLS"` 拼接代码（共 2 处，line 509 与 527）；`_build_builtin` 重构为接受 `prompt_builder` 参数并按新签名实例化 4 builtin

### nodes/agents.py wiring

- [ ] T033 [US1] 在 `src/cryptotrader/nodes/agents.py` 顶层添加 module-level singleton：
  ```python
  from pathlib import Path
  from cryptotrader.agents.prompt_builder import (
      DefaultMemoryProvider, DefaultSkillProvider, PromptBuilder,
  )

  _memory_provider: DefaultMemoryProvider | None = None
  _skill_provider: DefaultSkillProvider | None = None
  _prompt_builders: dict[str, PromptBuilder] = {}

  def _get_or_build_pb(agent_id: str, model: str) -> PromptBuilder:
      global _memory_provider, _skill_provider
      if _memory_provider is None:
          _memory_provider = DefaultMemoryProvider(memory_root=Path("agent_memory"))
          _skill_provider = DefaultSkillProvider(skills_root=Path("agent_skills"))
      if agent_id not in _prompt_builders:
          _prompt_builders[agent_id] = PromptBuilder(
              agent_id=agent_id,
              config_dir=Path("config/agents"),
              memory_provider=_memory_provider,
              skill_provider=_skill_provider,
              model=model,
          )
      return _prompt_builders[agent_id]
  ```
- [ ] T034 [US1] 修改 `nodes/agents.py:53` 调用 `cfg.agents.build(...)` 处：传 `prompt_builder=_get_or_build_pb(agent_id, model_override)`；删除 `regime_tags=regime_tags` 传参；同步删除 line 42 `regime_tags = state["data"].get("regime_tags", [])`
- [ ] T035 [US2] 在 nodes/agents.py 实例化 ToolAgent（chain / news）路径前，import load_skill_tool 并加到 tools：
  ```python
  from cryptotrader.agents.skills.tool import load_skill_tool
  ```
  确保 ToolAgent 实例化时 `tools` 包含 `load_skill_tool`（具体方式：可在 _build_builtin 内的 ChainAgent / NewsAgent factory 中包装 tools=[*CHAIN_TOOLS, load_skill_tool]，或者在 nodes/agents.py 实例化后修改 agent.tools）

### 删除 SkillsInjectionMiddleware

- [ ] T036 [US2] 删除 `src/cryptotrader/agents/skills/middleware.py` 整个文件
- [ ] T037 [US2] grep 全 src/，删除任何 `from cryptotrader.agents.skills.middleware import` 引用（base.py:600 ToolAgent.analyze 已在 T026 删除；nodes/ 等其他位置 grep 检查）

### security.py 注释

- [ ] T038 [US1] 修改 `src/cryptotrader/security.py:8` 注释：`role_description, ANALYSIS_FRAMEWORK` → `system_prompt section in config/agents/<id>.md`

### 测试更新

- [ ] T039 [P] [US1] 更新 `tests/test_tech_agent.py`：fixture 注入真实 PromptBuilder（指向 `config/agents/`）；mock LLM ainvoke；断言 SystemMessage.content 含 `config/agents/tech.md` system_prompt 标志性词；断言 indicators 字段被合并
- [ ] T040 [P] [US1] 更新 `tests/test_chain_agent.py`（同 T039 模板）
- [ ] T041 [P] [US1] 更新 `tests/test_news_agent.py`（同 T039）
- [ ] T042 [P] [US1] 更新 `tests/test_macro_agent.py`（同 T039）
- [ ] T043 [US1] 在 `tests/test_prompt_builder.py` 加 1 用例 `test_build_experience_overrides_memory_provider`：构造 mock memory_provider（如被调则记录），传非空 experience 调 build()，断言 mock memory_provider 未被调用，sections["recent_memory"] == experience

### 单测 + atomic gate

- [ ] T044 [US1] 运行 `uv run python -m pytest tests/test_config_loader.py tests/test_token_budget.py tests/test_prompt_builder.py tests/test_snapshot_renderer.py tests/test_tech_agent.py tests/test_chain_agent.py tests/test_news_agent.py tests/test_macro_agent.py -v --no-cov`，全部 PASS
- [ ] T045 [US1] 运行 `uv run python -m pytest tests/ --no-cov -x --ignore=tests/test_e2e_prompt_externalization.py 2>&1 | tail -10`，断言无回归（spec 014 / 015 既有测试 PASS）
- [ ] T046 [US2] 运行 `grep -rn "^ROLE\s*=" src/cryptotrader/agents/`，断言返回空
- [ ] T047 [US1] 运行 `wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py`，断言每个 < 150 行
- [ ] T048 [US2] 运行 `find src/cryptotrader/agents/skills/middleware.py 2>/dev/null`，断言文件不存在
- [ ] T049 [US1] 运行 `grep -rn "ANALYSIS_FRAMEWORK\|role_description\|prompt_template\|_resolve_role\|_resolve_skills\|SkillsInjectionMiddleware" src/cryptotrader/`，断言仅 spec/test 文档命中，无 src/ .py 文件命中
- [ ] T050 **Commit C2**：atomic 一次性提交 `git add -A`（包括 base.py / 4 agent / config.py / nodes/agents.py / prompt_builder.py / security.py / 4 agent test / test_prompt_builder.py / 删除 middleware.py），message：`feat(spec-017b/c2): atomic 4-agent migration to PromptBuilder + delete legacy paths`

**Checkpoint C2**：US-Y1 / US-Y2 / US-Y3 / US-Y4 全部满足；4 agent 真正走 PromptBuilder 路径；ROLE 退役；middleware 删除；ANALYSIS_FRAMEWORK 退役。

---

## Phase 4: User Story 5 — C3 commit（E2E + Final Gate）

**Goal**：US-Y5（Telemetry + E2E）满足；SC-Y10 / SC-Y15 / SC-Y17 满足。

- [ ] T051 [US5] 创建 `tests/test_e2e_prompt_externalization.py`：(a) mock 4 agent 的 LLM ainvoke 返回 fixture analysis；(b) 构造完整 mocked LangGraph cycle（4 agent → debate gate → verdict → risk gate）；(c) 跑 cycle；(d) 断言 4 agent 各自 OpenTelemetry span 含 8 字段（agent_id / sections_included / dropped_sections / degraded_sections / prompt_size_pre / prompt_size_post / budget / duration_ms）；(e) 断言 final verdict 字段完整（含 target_price / stop_loss / take_profit / R:R）
- [ ] T052 [US5] 在 `tests/fixtures/skills/_test_shared/SKILL.md` 创建 fixture skill（scope: shared）；`tests/fixtures/skills/_test_tech/SKILL.md`（scope: agent:tech）。E2E 测试断言 4 agent 都加载到 _test_shared，仅 TechAgent 加载到 _test_tech（覆盖 SC-Y13）
- [ ] T053 [US5] 运行 `uv run python -m pytest tests/test_e2e_prompt_externalization.py -v --no-cov`，全部 PASS
- [ ] T054 [P] 运行 `ruff check src/cryptotrader/agents/ src/cryptotrader/config.py src/cryptotrader/nodes/agents.py tests/`；如有新错误，按 spec 017a 经验加 per-file-ignores 到 pyproject.toml
- [ ] T055 [P] 运行 `ruff format src/cryptotrader/agents/ src/cryptotrader/config.py src/cryptotrader/nodes/agents.py tests/`
- [ ] T056 运行整体回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -10`，确认无回归
- [ ] T057 最终 grep gate：再次运行 T046-T049 的 4 个 grep / wc 命令，断言全部满足
- [ ] T058 **Commit C3**：`git add tests/test_e2e_prompt_externalization.py tests/fixtures/skills/ pyproject.toml`（如有 ruff 修改）后 `git commit -m "feat(spec-017b/c3): e2e test + final gates"`

**Checkpoint C3**：所有 SC（Y1-Y17）满足；3 commit 序列完整。

---

## Phase 5: Polish & Cross-Cutting

- [ ] T059 [P] 跑 `pytest tests/ --no-cov 2>&1 | tail -5` 全套测试，确认整体通过率
- [ ] T060 检查 `wc -l src/cryptotrader/agents/snapshot_renderer.py` ≥ 50 行（SC-Y3 part 4 验证）
- [ ] T061 验证 `pytest tests/test_config_loader.py tests/test_token_budget.py tests/test_prompt_builder.py -v --no-cov 2>&1 | tail -5` 含 `45 passed`（44 + 1 新增 = 45，SC-Y11 验证）
- [ ] T062 在 `specs/017-agent-prompt-externalization/contracts/prompt-builder.md` 顶部增加注释指向 spec 017b `contracts/promptbuilder-experience-extension.md`（跨 spec 文档维护，非阻塞）

---

## 依赖图

```
Phase 2 (T001-T014, C1) ──> Phase 3 (T015-T050, C2 atomic) ──> Phase 4 (T051-T058, C3) ──> Phase 5 (T059-T062)
```

C1 / C2 / C3 是 commit 顺序，每个 commit 必须保证测试 PASS 后才进下一个。

## 并行执行示例

### Phase 2 内部

T001-T004（读 4 个 agent ROLE）/ T006-T009（创建 4 个 config 文件）可分别并行：

```
worker1: T001 → T006   (tech)
worker2: T002 → T007   (chain)
worker3: T003 → T008   (news)
worker4: T004 → T009   (macro)
顺序: T005 → 各 worker 接收 ANALYSIS_FRAMEWORK 拆段后填入 config
顺序: T010 → T011 → T012 → T013 → T014
```

### Phase 3 内部

T027-T030（重构 4 agent 类）+ T039-T042（更新 4 agent test）可 4 路并行：

```
worker1: T027 → T039   (tech)
worker2: T028 → T040   (chain)
worker3: T029 → T041   (news)
worker4: T030 → T042   (macro)
顺序: T015-T026（base.py + prompt_builder.py 重构，必须先于 agent 类重构）
顺序: T031-T038（config.py / nodes/agents.py / 删 middleware / security.py，必须先于 4 agent 重构）
顺序: T043-T050（测试 + atomic commit）
```

## MVP 范围

**MVP**：Phase 2 + Phase 3（C1 + C2 commit）— 4 agent 切换完成 + ROLE 退役 + middleware 删除。
**完整交付**：MVP + Phase 4 + Phase 5（C3 commit + Polish）。

## 任务统计

| Phase | Task 数 | 涉及 user stories | 提交 |
|---|---|---|---|
| 2 Foundational (C1) | 14 | US1 / US3 | C1 |
| 3 Migration (C2) | 36 | US1 / US2 / US3 / US4 | C2 |
| 4 E2E (C3) | 8 | US5 | C3 |
| 5 Polish | 4 | — | — |
| **总计** | **62** | — | 3 commit |

## Implementation Strategy

1. **C1 优先**：T001-T014 完成提供安全 baseline，main 路径无 behavior 变化即可发布
2. **C2 atomic**：T015-T050 必须**全部完成**才能跑测试，中间状态会让 4 agent 实例化失败
3. **C3 验收**：E2E 测试 + final gate 验证 SC 全部满足
4. **回滚策略**：如 C2 commit CI 失败 → revert C2，main 回到 C1 状态（无 behavior 变化）；如 C3 失败 → revert C3，main 回到 C2 状态（4 agent 已切换但 E2E 测试缺失）
