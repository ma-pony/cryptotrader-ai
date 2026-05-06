# Quickstart — 014 双层架构

## 前置

- Python 3.12 + uv
- checkout 分支 `014-agent-skills-protocol-migration`
- `uv sync` 已跑过

## 验证实现

```bash
# 1. 跑新增测试（Phase 1 + Phase 2 完整覆盖）
uv run pytest tests/test_agent_memory_writer.py \
              tests/test_skills_loader.py \
              tests/test_skills_middleware.py \
              tests/test_load_skill_tool.py \
              tests/test_reflection_pattern_distill.py \
              tests/test_anti_overfitting_equivalence.py \
              tests/test_skills_curation.py \
              tests/test_skill_proposal.py \
              tests/test_applied_pattern_parser.py \
              tests/test_two_layer_architecture.py \
              -v

# 2. 跑全套件确认无 regression
uv run pytest --no-cov -q
```

## 文件布局检查

```bash
# git 跟踪的 5 个 SKILL.md（已实现）
ls -la agent_skills/
# tech-analysis/SKILL.md
# chain-analysis/SKILL.md
# news-analysis/SKILL.md
# macro-analysis/SKILL.md
# trading-knowledge/SKILL.md

# gitignored 的 memory 层（每 cycle 写入）
ls agent_memory/   # 应在 .gitignore 中
git check-ignore agent_memory/   # 应输出 agent_memory/

# 跑一次 cycle 看 memory 写入
uv run arena run --pair BTC/USDT --mode paper
ls agent_memory/cases/           # 新的 per-cycle 文件
git status                       # agent_memory/ 不在 untracked 列表

# 整理 SKILL.md（手工 review LLM 输出后 merge）
uv run arena skills curate tech-analysis --llm
diff agent_skills/tech-analysis/SKILL.md.draft \
     agent_skills/tech-analysis/SKILL.md
mv agent_skills/tech-analysis/SKILL.md.draft \
   agent_skills/tech-analysis/SKILL.md

# 列出所有 skills
uv run arena skills list

# 提案新 skill
uv run arena skills propose-new --scope agent:tech
```

## 实施完成 checklist

- [x] `.gitignore` 已加 `agent_memory/`
- [x] `agent_skills/` 5 个 SKILL.md 已初始化（含 Anthropic protocol frontmatter）
- [x] `learning/context.py` 已删除（FR-029）
- [x] `learning/reflect.py` 已删除（FR-029 / FR-031 DB 路径）
- [x] `models.py` 中 `ExperienceMemory` / `ExperienceRule` 已删除（FR-030）
- [x] `decision_commits.experience_memory` 列 DROP migration 已加入（FR-031）
- [x] `arena experience` CLI 子命令已移除（FR-032）
- [x] `test_context.py` / `test_reflect.py` 已删除（FR-033）
- [x] `arena skills curate / propose-new / list` CLI 命令已实装（T030）
- [x] `SkillsInjectionMiddleware` 已接入 `ToolAgent.analyze()`（T036）
- [x] GSSC path 已从 `nodes/agents.py` 移除（T037）
- [x] `parse_applied()` 已接入 journal 位置关闭时 PnL 回填（T041）
- [x] verdict prompt 已加 `applied:` 格式说明（T039）
- [ ] paper cycle 跑通验证 middleware 注入（需活跃凭证）
- [ ] `arena reflect` 跑通验证 memory 蒸馏（需活跃 cycle 数据）

## 验证 middleware 注入

```bash
LOG_LEVEL=DEBUG uv run arena run --pair BTC/USDT --mode paper 2>&1 | \
  grep -A 30 "system_message"
```

应看到：
- system_message.content_blocks 含原 system_prompt + 注入的 SKILL.md body
- body 含 `## Role`、`## Active Patterns`（如有）、`## Forbidden Zones`（如有）、`## Shared Trading Knowledge`

## 调试 tip

- 直接读 SKILL.md：`cat agent_skills/tech-analysis/SKILL.md`
- 直接读 case：`cat agent_memory/tech/cases/$(ls -t agent_memory/tech/cases/ | head -1)`
- 测 `load_skill`：`uv run python -c "from cryptotrader.agents.skills.tool import load_skill; print(load_skill('tech-analysis'))"`
