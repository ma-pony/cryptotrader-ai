# spec 025 任务清单 — Agent-Native Skill Protocol Layer

## 阶段 1：路径迁移（Commit 1）

- [X] **T016** [US1] `git mv agent_skills/{tech,chain,news,macro,trading-knowledge}-analysis → agent_skills/_internal/` — 5 个 skill 目录整体迁移
- [X] **T017** [US1] `src/cryptotrader/learning/evolution/skill_provider.py` — 默认 `skill_root` 从 `agent_skills` 改为 `agent_skills/_internal`；`src/cryptotrader/agents/skills/_constants.py` `DEFAULT_AGENT_SKILLS_DIR` 同步更新；`src/cryptotrader/nodes/agents.py` 初始化路径同步更新
- [X] **T018** [US1] 更新 `tests/` 中所有 `agent_skills/` fixture 路径引用到 `agent_skills/_internal/`：`test_two_layer_architecture.py`、`test_e2e_prompt_externalization.py`、`test_security.py`；`src/cli/main.py` `skills_list()` 路径同步
- [X] **T019** [P] [US1] 新建 `tests/test_skill_provider_internal_path.py` — 验证 EvolvingSkillProvider 从 `_internal/` 加载 4 个 skill + spec 019 既有行为不回归

## 阶段 2：外部 SKILL.md + /skill/ 端点（Commit 2）

- [X] **T001** [P] 创建 `agent_skills/_external/{cryptotrader,verdict-feed,market-intel,evolution-insights,execution-replay}/` 目录结构
- [X] **T020** [P] [US1] `agent_skills/_external/cryptotrader/SKILL.md` — bootstrap skill（install + 5 child 路由 + auth + JWT 设计存档）
- [X] **T021** [P] [US1] `agent_skills/_external/verdict-feed/SKILL.md` — verdict 决策流 curl 示例
- [X] **T022** [P] [US1] `agent_skills/_external/market-intel/SKILL.md` — 4 agent 输出 + snapshot 数据
- [X] **T023** [P] [US1] `agent_skills/_external/evolution-insights/SKILL.md` — trilogy 产出 + PatternRecord/SkillRecord schema
- [X] **T024** [P] [US1] `agent_skills/_external/execution-replay/SKILL.md` — journal events + OCO 查询
- [X] **T025** [US1] `src/api/routes/skills.py` — `SkillRecord` Pydantic v2 schema + `GET /skill/{name}?format=markdown|json` handler
- [X] **T026** [US1] `src/api/main.py` — 注册 skills router（`Depends(verify_api_key)`）
- [X] **T027** [US1] `skills.py` handler — `ExternalSkillFetchAggregator.record()` 集成（graceful degrade on ImportError）
- [X] **T028** [P] [US1] `tests/test_skills_endpoint.py` — 5 测试用例（markdown / json / 404 / 401 / frontmatter parse）
