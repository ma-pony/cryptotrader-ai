# Quickstart：Memory Evolution（spec 018）

本文档展示 spec 018 落地后的开发者使用入口。

## 落地后目录结构

```
agent_memory/
├── cases/<cycle_id>.md                # spec 014 既有 + 3 段新增
├── tech/patterns/<rule_name>.md       # spec 014 既有 + 5 字段新增
├── tech/patterns/.archived/           # NEW (本 spec)
├── chain/patterns/...                 # 同上
├── news/patterns/...
└── macro/patterns/...

src/cryptotrader/learning/evolution/   # NEW (C2)
├── __init__.py
├── fsm.py                             # evaluate_transitions
├── pareto.py                          # rank_rules
├── ive.py                             # classify_case
└── provider.py                        # EvolvingMemoryProvider (C3)

src/cryptotrader/nodes/
└── evolution.py                       # NEW (C3) — evaluate_node

src/api/routes/
└── memory.py                          # NEW (C4) — 4 endpoints

web/src/pages/memory/                  # NEW (C4)
├── MemoryPage.tsx
├── components/{RulesGrid,CasesTimeline,ArchivedRules,RecentTransitions}.tsx
└── queries.ts

scripts/
└── migrate_017_to_018.py              # NEW (C1)
```

## 开发者使用场景

### 场景 1：观察某 rule 的进化状态

打开 web UI `/memory`，按 agent 筛选 rule，看 maturity 状态分布。

或调 API：

```bash
curl 'http://localhost:8000/api/memory/rules?agent=tech&status=active'
```

### 场景 2：诊断某次亏损的 IVE 分类

```bash
# 查最近 24h cases
curl 'http://localhost:8000/api/memory/cases?from=2026-05-08T00:00:00Z'
```

或在 web UI `/memory` 的 Cases Timeline section 点击查看详情。

### 场景 3：查看自动归档历史

```bash
curl 'http://localhost:8000/api/memory/archived'
```

### 场景 4：手动触发 cycle 看进化事件

```bash
arena cycle run --pair BTC/USDT
# 完成后看 /memory 页面 Recent Transitions section
```

### 场景 5：实施迁移（生产环境）

```bash
# 1. 备份
cp -r agent_memory agent_memory.backup_pre_018_$(date +%Y%m%d_%H%M%S)

# 2. dry-run 验证
python scripts/migrate_017_to_018.py --dry-run

# 3. 实跑
python scripts/migrate_017_to_018.py

# 4. 部署 spec 018 代码
git pull && arena scheduler restart

# 5. 监控第 1 个 cycle
tail -f logs/arena.log | grep -E "evaluate_node|memory.evolution"
```

### 场景 6：本地单测 EvolvingMemoryProvider

```python
import pytest
from pathlib import Path
from cryptotrader.learning.evolution.provider import EvolvingMemoryProvider

@pytest.fixture()
def provider(tmp_path):
    # tmp_path 含 fixture pattern + case 文件
    return EvolvingMemoryProvider(memory_root=tmp_path)

def test_get_recent_memory_returns_active_rules(provider):
    md = provider.get_recent_memory(agent_id="tech", snapshot={})
    assert "active rule body" in md
    assert "archived rule" not in md
```

## 验证清单（C4 完成后跑）

```bash
# SC-Z1: cases schema 含新 3 段
ls agent_memory/cases/*.md | head -3 | xargs -I {} grep -l "## Trade Execution" {}

# SC-Z2: patterns schema 含新 5 字段
grep -l "importance:\|access_count:" agent_memory/*/patterns/*.md 2>/dev/null

# SC-Z3 / Z4 / Z6 / Z7 / Z8: 各模块单测 PASS
uv run python -m pytest tests/test_migrate_017_to_018.py tests/test_evolving_memory_provider.py tests/test_fsm.py tests/test_pareto.py tests/test_ive.py -v --no-cov

# SC-Z5: DefaultMemoryProvider 退役
grep -rn "class DefaultMemoryProvider" src/cryptotrader/
# 期望：返回空

# SC-Z9: 1 cycle IVE Classification 段非空
arena cycle run --pair BTC/USDT --once
ls agent_memory/cases/*.md | tail -1 | xargs grep -A 5 "## IVE Classification"

# SC-Z11: graph 节点位置
python -c "from cryptotrader.graph import build_trading_graph; print(build_trading_graph().get_graph().to_text())" | grep -E "risk_gate|evaluate|journal"

# SC-Z12: API 测试
uv run python -m pytest tests/test_api_memory.py -v --no-cov

# SC-Z13: 前端测试
cd web && pnpm vitest run pages/memory

# SC-Z14: Sidebar 含 /memory 链接
grep "to: '/memory'" web/src/components/layout/sidebar.tsx

# SC-Z15: E2E
uv run python -m pytest tests/test_e2e_memory_evolution.py -v --no-cov

# SC-Z16: 全套回归
uv run python -m pytest tests/ --no-cov 2>&1 | tail -5
```

## 与 spec 019 / 020 的衔接

spec 019（待立项，Skill 子域）启动后，无需改 spec 018 任何代码：

```python
# spec 019 示意
from cryptotrader.evolution.skill_provider import EvolvingSkillProvider

# nodes/agents.py 顶层
_skill_provider = EvolvingSkillProvider(...)  # 替代 spec 017a/b 的 DefaultSkillProvider
# EvolvingMemoryProvider 不变
```

spec 020（待立项，Ops 子域）启动后：

```python
# spec 020 加 cron daemon
from cryptotrader.ops.daemon import EvolutionDaemon

EvolutionDaemon(
    memory_provider=_memory_provider,  # 本 spec 实例
    interval_seconds=3600,  # 1h 跑一次 evaluate_all_rules
).start()
```

或加 Anthropic prompt cache 配置：

```python
# spec 020 修 PromptBuilder
prompt_cache_config = {"system_prompt": "cached", ...}
```

均不破坏本 spec 接口契约。
