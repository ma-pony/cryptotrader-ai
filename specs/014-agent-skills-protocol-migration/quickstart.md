# Quickstart — 014 双层架构

## 前置

- Python 3.12 + uv
- checkout 分支 `014-agent-skills-protocol-migration`
- `uv sync` 已跑过

## 验证 Phase 1 实现

```bash
# 1. 跑新增测试
uv run pytest tests/test_agent_memory_writer.py \
              tests/test_skills_loader.py \
              tests/test_skills_middleware.py \
              tests/test_load_skill_tool.py \
              tests/test_reflection_pattern_distill.py \
              tests/test_anti_overfitting_equivalence.py \
              tests/test_skills_curation.py \
              -v

# 2. 跑全套件确认无 regression
uv run pytest --no-cov -q
```

## 文件布局检查

```bash
# git 跟踪的 5 个 SKILL.md
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
ls agent_memory/tech/cases/      # 新文件
git status                       # agent_memory/ 不在 untracked 列表

# 跑 reflection 看蒸馏
uv run arena reflect --commits-since=24h
ls agent_memory/tech/patterns/

# 整理 SKILL.md（手工 review LLM 输出后 merge）
uv run arena skills curate tech-analysis --llm
diff agent_skills/tech-analysis/SKILL.md.draft \
     agent_skills/tech-analysis/SKILL.md
mv agent_skills/tech-analysis/SKILL.md.draft \
   agent_skills/tech-analysis/SKILL.md
```

## 实施前 checklist

- [ ] `.gitignore` 已加 `agent_memory/`
- [ ] `agent_skills/` 5 个 SKILL.md 已手工初始化（agent role 文本搬入）
- [ ] `agent_memory/` 4 个 agent 子目录骨架已创建（cases/ patterns/ archive/ 各 .gitkeep）
  - 注：.gitkeep 也在 gitignored 目录下，但本地需要存在以让目录被 loader 识别
- [ ] `learning/context.py` 已删除
- [ ] `models.py` 中 `ExperienceMemory` / `ExperienceRule` 已删除
- [ ] `decision_commits.experience_json` 列已 drop（auto-migration 通过）
- [ ] `arena experience` CLI 子命令已移除
- [ ] 4 个 GSSC 测试文件已删除
- [ ] 全套件测试通过
- [ ] paper cycle 跑通验证 middleware 注入
- [ ] `arena reflect` 跑通验证 memory 蒸馏
- [ ] `arena skills curate` CLI 命令存在

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
