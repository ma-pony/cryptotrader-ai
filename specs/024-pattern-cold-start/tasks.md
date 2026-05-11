# Tasks: Spec 021 — Pattern Cold-Start

**Branch**: `024-pattern-cold-start` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Phase 1: Setup（无新依赖）

复用 stdlib re / collections.Counter / pathlib + 既有 typer + spec 014 helpers。

## Phase 2: Foundational

- [x] T001 [P] 在 `src/cryptotrader/config.py:ExperienceConfig` 加 `min_cases_per_pattern: int = 5` 字段
- [x] T002 [P] 在 `config/default.toml` `[experience]` 段加 `min_cases_per_pattern = 5`

---

## Phase 3: User Story 1 — Distill Cold-Start（P1）

**Goal**：`distill_patterns()` 从 cases 提炼并创建新 PatternRecord 文件。

**Independent Test**：`distill_patterns` 199 fixture cases → ≥ 1 pattern 创建。

- [x] T003 [US1] 在 `src/cryptotrader/learning/memory.py` 加私有 helper `_make_pattern_slug(applied_text: str, existing_dir: Path) -> str`（按 research.md Decision 2：lowercase + 非 alnum 替换 - + 截断 60 + collision `-N` 后缀）
- [x] T004 [US1] 在 `src/cryptotrader/learning/memory.py` 加私有 helper `_create_pattern_from_cases(slug, agent, applied_text, case_data_list) -> PatternRecord`（按 research.md Decision 3：pnls 过滤 None / source_cycles[:5] / regime_tags 频次 top 3 字母序兜底 / maturity="observed"）
- [x] T005 [US1] 修改 `distill_patterns()` 加 cold-start 路径：在统计 `agent_pattern_counts` 后 + 既有 maturity 更新前，按 `count >= cfg.experience.min_cases_per_pattern` 阈值创建 patterns；失败 isolated（单 pattern 失败不影响其他）
- [x] T006 [US1] 在 `distill_patterns()` cold-start 路径 写 OTel span `learning.distill.cold_start` + 3 attr（patterns_created / patterns_updated / cases_processed）
- [x] T007 [P] [US1] 创建 `tests/test_pattern_slug_generation.py`：5 用例（empty input / non-alnum chars / truncate 60 chars / collision -2 -3 / non-ascii fallback "unnamed"）
- [x] T008 [P] [US1] 创建 `tests/test_distill_patterns_cold_start.py`：5 用例（empty cases / freq below threshold / freq above threshold creates pattern / pnl all None creates with empty PnLTrack / regime_tags top 3 voting）

---

## Phase 4: User Story 2 — Daemon Pattern Extraction Action（P1）

**Goal**：daemon 加第 4 个 action `pattern_extraction`。

**Independent Test**：`arena evolution-daemon --once` 4 actions 全 PASS。

- [x] T009 [US2] 修改 `src/cryptotrader/ops/daemon.py`：加 `async def _action_pattern_extraction(self) -> ActionResult` 方法（按 research.md Decision 4）
- [x] T010 [US2] 在 daemon `_run_action()` dispatch 加 `elif name == "pattern_extraction": return await self._action_pattern_extraction()`
- [x] T011 [US2] 修改 `config/default.toml`：`[evolution_daemon].actions` 默认列表加 `"pattern_extraction"`（4 个 actions）
- [x] T012 [P] [US2] 创建 `tests/test_daemon_pattern_extraction.py`：3 用例（action runs PASS / details 含 new_count/updated_count/archived_count/cases_processed / distill_patterns 异常时 SKIP soft degrade）

---

## Phase 5: User Story 3 — CLI Manual Trigger（P2）

**Goal**：`arena experience distill` typer command。

**Independent Test**：CLI 跑完输出 ReflectionRun summary + exit 0。

- [x] T013 [US3] 修改 `src/cli/main.py`：加 `experience_app = typer.Typer()` + `@experience_app.command("distill")` function（按 research.md Decision 5）
- [x] T014 [US3] 在 cli/main.py 加 `app.add_typer(experience_app, name="experience")` 注册子命令
- [x] T015 [P] [US3] 创建 `tests/test_cli_experience_distill.py`：3 用例（默认参数 PASS / `--memory-dir custom` 路径生效 / `--cycles-window N` 限制生效）

---

## Phase 6: Polish & E2E

- [x] T016 [P] 创建 `tests/test_e2e_pattern_cold_start.py`：端到端 fixture 200+ cases → 跑 daemon `pattern_extraction` action → ≥ 3 patterns 创建 → API `/api/memory/rules` 返回 total > 0
- [x] T017 跑 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -3` 验证 ≥ 2458 passed / 0 failed（SC-P7）→ 2476 passed
- [x] T018 跑 `uv run ruff check src/cryptotrader/learning/memory.py src/cryptotrader/ops/daemon.py src/cryptotrader/config.py src/cli/main.py tests/test_distill_patterns_cold_start.py tests/test_pattern_slug_generation.py tests/test_daemon_pattern_extraction.py tests/test_cli_experience_distill.py tests/test_e2e_pattern_cold_start.py` clean
- [x] T019 跑 SC-P1：`uv run arena experience distill --memory-dir agent_memory --cycles-window 200` exit 0 + ≥ 1 patterns created → 3 patterns created
- [x] T020 跑 SC-P2：`find agent_memory/{tech,chain,news,macro}/patterns -name "*.md" 2>/dev/null | wc -l` ≥ 3 → 3 files
- [ ] T021 跑 SC-P3：`curl /api/memory/rules` total > 0（API 重启可能需要）
- [x] T022 跑 SC-P4：`uv run arena evolution-daemon --once` 4 actions 全 PASS → pareto/regime/skill_proposal/pattern_extraction PASS
- [x] T023 跑 `git log --oneline 024-pattern-cold-start..main | wc -l` ≤ 4 commit（SC-P10）

---

## Dependencies

```
Phase 2 (T001-T002 config) ──┐
                              ├─→ US1 (T003-T008 distill + tests)
                              ├─→ US2 (T009-T012 daemon + test)
                              └─→ US3 (T013-T015 CLI + test)
                                      ↓
                              Phase 6 Polish (T016-T023)
```

T003/T004（helper 函数）必须在 T005（distill_patterns 改）前。
T009（daemon action 方法）必须在 T010（dispatch）前。
T013（typer.Typer 实例）必须在 T014（app.add_typer 注册）前。
US1/US2/US3 之间无依赖（不同文件 / 不同模块），可并行。

## Parallel Execution

US1 内：T007 / T008 可与 T003-T006 并行（独立测试文件）
US2 内：T012 可与 T009-T011 并行
US3 内：T015 可与 T013-T014 并行
跨 US：US1 + US2 + US3 完全独立可并行（不同 caller 不同文件）

## Implementation Strategy

### MVP scope

**最小可发布**：T001-T008（Phase 2 + US1）= config + distill cold-start + 单测。可独立验证（不依赖 daemon）。

### 4-commit 切分

| commit | tasks | 说明 |
|---|---|---|
| C1 | T001-T008 | distill cold-start + helpers + config + 单测（US1）|
| C2 | T009-T015 | daemon action + CLI + 单测（US2+US3）|
| C3 | T016 | E2E |
| C4 | T017-T023 | final gate（SC 全套验证）|

## Validation

任务总数：23
- Foundational: 2
- US1: 6
- US2: 4
- US3: 3
- Polish: 8

每 US 含 independent test；每 task 含具体路径；checklist format 全部合规。
