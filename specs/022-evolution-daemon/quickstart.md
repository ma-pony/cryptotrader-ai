# Quickstart：Evolution Daemon（spec 020b）

本文档展示 spec 020b 落地后的开发者使用入口。

## 落地后目录结构

```
src/cryptotrader/
├── ops/                                # NEW (本 spec)
│   ├── __init__.py
│   └── daemon.py                       # EvolutionDaemon 类
└── observability/
    └── daemon_metrics.py               # NEW (本 spec): redis 接口

src/api/routes/metrics.py               # MODIFY: 加 3 Gauge
src/cli/main.py                          # MODIFY: 加 evolution-daemon 命令
src/cryptotrader/learning/memory.py     # MODIFY: 加 refilter_records_by_regime wrapper
src/cryptotrader/config.py              # MODIFY: 加 EvolutionDaemonConfig

config/default.toml                     # MODIFY: 加 [evolution_daemon] section
docker-compose.yml                      # MODIFY: 加 evolution-daemon service

tests/
├── test_evolution_daemon.py            # NEW
├── test_daemon_metrics.py              # NEW
├── test_e2e_evolution_daemon.py        # NEW
└── test_evolution_daemon_cli.py        # NEW
```

## 开发者使用场景

### 场景 1：单次 dry-run（dev 机）

```bash
uv run arena evolution-daemon --once

# 期望输出
# [evolution-daemon] run_once exit_code=0
#   [pareto] PASS 145ms
#   [regime] PASS 67ms
#   [skill_proposal] PASS 2341ms
```

### 场景 2：观察 daemon 指标

```bash
curl http://localhost:8000/metrics | grep -E "evolution_daemon|skill_proposal_draft"

# 期望输出
# evolution_daemon_run_count_24h 1.0
# evolution_daemon_llm_failure_rate_24h 0.0
# skill_proposal_draft_count_7d 2.0
```

### 场景 3：生产部署（docker-compose）

```bash
# 启动 evolution-daemon service（独立容器）
docker compose up -d evolution-daemon

# 查看日志
docker compose logs -f evolution-daemon

# daemon 跑首次 daily（UTC midnight）后 redis 写事件
docker compose exec redis redis-cli ZRANGE evolution_daemon:events 0 -1 WITHSCORES
```

### 场景 4：临时禁用 daemon

```bash
# 方式 A：env 全局关闭
EVOLUTION_DAEMON_ENABLED=false docker compose up -d evolution-daemon
# daemon entrypoint 检测后立即 exit 0

# 方式 B：toml 关闭
# 编辑 config/default.toml: [evolution_daemon].enabled = false
# 重启 service: docker compose restart evolution-daemon
```

### 场景 5：本地单测 EvolutionDaemon

```python
import pytest
from unittest.mock import patch
from cryptotrader.ops.daemon import EvolutionDaemon
from cryptotrader.config import EvolutionDaemonConfig

@pytest.fixture()
def daemon(tmp_path):
    cfg = EvolutionDaemonConfig(
        enabled=True,
        cron="0 0 * * *",
        actions=["pareto", "regime", "skill_proposal"],
        propose_threshold=10,
    )
    return EvolutionDaemon(config=cfg)

@pytest.mark.asyncio
async def test_run_once_all_pass(daemon):
    with patch("langchain_openai.ChatOpenAI.ainvoke") as mock_llm:
        mock_llm.return_value = AIMessage(...)
        result = await daemon.run_once()
        assert result.exit_code == 0
        assert all(a.status == "PASS" for a in result.actions_run)

@pytest.mark.asyncio
async def test_run_once_soft_degrade_on_llm_failure(daemon):
    with patch("langchain_openai.ChatOpenAI.ainvoke", side_effect=OpenAIAPIError("boom")):
        result = await daemon.run_once()
        assert result.exit_code == 0  # soft degrade
        actions_by_name = {a.name: a for a in result.actions_run}
        assert actions_by_name["pareto"].status == "PASS"  # 算法部分仍跑
        assert actions_by_name["regime"].status == "PASS"
        assert actions_by_name["skill_proposal"].status == "SKIP"  # LLM-dependent → SKIP
```

### 场景 6：本地单测 daemon_metrics

```python
from cryptotrader.observability.daemon_metrics import (
    DaemonRunCountAggregator,
    DaemonLLMFailureAggregator,
    SkillProposalDraftAggregator,
)
from time import time, sleep

def test_run_count_aggregator_sliding_window():
    agg = DaemonRunCountAggregator()
    agg.record()
    assert agg.count() == 1

    # 24h 之外的 entry 应被 evict
    # （实际测试用 fixture mock time）
    ...

def test_llm_failure_rate():
    agg = DaemonLLMFailureAggregator()
    agg.record(failed=False)
    agg.record(failed=False)
    agg.record(failed=True)
    assert abs(agg.failure_rate() - 1/3) < 0.01
```

### 场景 7：手动触发 docker daemon 跑一次

```bash
docker compose exec evolution-daemon arena evolution-daemon --once
```

## 验证清单（C4 完成后跑）

```bash
# SC-D1: dry-run exit 0
uv run arena evolution-daemon --once
echo $?  # 应为 0

# SC-D3: grep daemon class
grep -n "class EvolutionDaemon" src/cryptotrader/ops/daemon.py
# 期望：≥ 1 hit

# SC-D2 + SC-D4: pytest e2e + 全套回归
uv run python -m pytest tests/test_e2e_evolution_daemon.py tests/test_evolution_daemon.py tests/test_daemon_metrics.py tests/test_evolution_daemon_cli.py -v --no-cov
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
# 期望：≥ 2391 passed / 0 failed

# SC-D5: Prometheus gauge
curl http://localhost:8000/metrics | grep -c "evolution_daemon\|skill_proposal_draft"
# 期望：≥ 3

# SC-D6: docker-compose service 可解析 + 启动
docker compose config evolution-daemon
docker compose up -d evolution-daemon
docker compose ps evolution-daemon
# 期望：service 存在 + state=running

# SC-D7: soft degrade scenario（在测试中）
uv run python -m pytest tests/test_e2e_evolution_daemon.py::test_soft_degrade_on_llm_failure -v --no-cov
# 期望：PASS

# SC-D10: commit 数
git log --oneline 022-evolution-daemon..main | wc -l
# 期望：≤ 4（C1-C4，外加 ship pipeline 的 spec docs / review 等 artifact 不算）
```

## 与 spec 020c 的衔接

spec 020c 启动后：

```python
# 020c 加 git lineage hook（在 daemon run_once 末尾）
from cryptotrader.ops.lineage import GitLineageHook

hook = GitLineageHook(branch="evolution", batch_mode=True)
hook.commit_changes(actions_run=result.actions_run)

# 020c 同时收尾 020a P2 advisory
# - staging step 4 集成测试缺口
# - SkillsGrid badge a11y aria-label
```

均不破坏本 spec 接口契约。
