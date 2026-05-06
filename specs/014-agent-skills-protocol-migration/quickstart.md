# Quickstart — Agent Skills 协议迁移开发

## 前置

- Python 3.12 + uv
- 当前已 checkout `014-agent-skills-protocol-migration` 分支
- 跑过一次 `uv sync` 确保依赖装好

## 本地验证 Phase 1 实现

```bash
# 1. 跑新增的 skill loader / middleware 测试
uv run pytest tests/test_agent_skills_loader.py tests/test_skills_middleware.py tests/test_load_skill_tool.py -v

# 2. 跑 reflection writer 测试
uv run pytest tests/test_skills_reflection.py tests/test_skills_anti_overfitting.py -v

# 3. 跑 paper-mode 完整 cycle，验证 middleware 自动注入
uv run arena run --pair BTC/USDT --mode paper

# 4. 看注入的 system prompt（DEBUG 模式）
LOG_LEVEL=DEBUG uv run arena run --pair BTC/USDT --mode paper 2>&1 | grep -A 50 "system_message"

# 5. 手动跑一次 reflection job（CLI）
uv run arena reflect --commits-since=24h
```

## 检查文件布局

```bash
tree agent_skills -L 3 -I 'archive'
```

预期：

```text
agent_skills/
├── tech
│   ├── instructions.md
│   ├── patterns/      # 初期空
│   ├── forbidden/     # 初期空
│   └── archive/       # 初期空
├── chain/...
├── news/...
├── macro/...
└── shared
    ├── funding_rate.md
    ├── regime_definitions.md
    └── trading_pair_semantics.md
```

## 验证 LangChain middleware 注入是否生效

通过 grep DEBUG 日志找到形如：

```
system_message:
  content_blocks:
    - type: text
      text: "## Agent Instructions\n\n{tech instructions body}\n\n## Available Patterns (3 matched current regime: ['range_bound'])\n\n- **tech::funding_squeeze_long**: ...\n..."
```

confirm 如下：
1. 上方有 `## Agent Instructions`（来自 `tech/instructions.md`）
2. 中段有 `## Available Patterns` + ≥ 0 条用 `**agent::name**` 格式
3. 含 `## Loading Rule` 段，提示 agent 用 `load_skill` tool

## 触发 `load_skill` 的方式

agent 内部会自动决定调用，无需手动触发。但可以单测：

```bash
uv run python -c "
from cryptotrader.agents.skills.tool import load_skill
print(load_skill('tech::funding_squeeze_long'))
print(load_skill('nonexistent_pattern'))
"
```

## 触发 reflection job（手工）

```bash
# 一次性反思最近 24h 的 commits
uv run arena reflect --commits-since=24h

# 看反思生成 / 更新的文件
git status agent_skills/
```

## 进入实施前的最终 checklist

- [ ] `agent_skills/` 17 个种子文件已创建（4 instructions + shared 3 + 8 .gitkeep + ...）
- [ ] `learning/context.py` 已删除
- [ ] `models.py` 中 ExperienceMemory / ExperienceRule 已删除
- [ ] `decision_commits.experience_json` 列已 drop（auto-migration 通过）
- [ ] `arena experience` CLI 子命令已移除
- [ ] 4 个 GSSC 测试文件已删除
- [ ] 全套件 `uv run pytest --no-cov -q` 通过（≥ 2003 测试）
- [ ] 至少跑过 1 次 paper cycle 验证 middleware 注入
- [ ] 至少跑过 1 次 `arena reflect` 验证 reflection 写文件
- [ ] `grep -rn "ExperienceMemory\|ExperienceRule\|gather_packets" src/ tests/` 返回 0 结果
