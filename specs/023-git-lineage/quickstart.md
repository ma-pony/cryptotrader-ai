# Quickstart：Git Lineage（spec 020c）

## 落地后目录结构

```
src/cryptotrader/
├── ops/
│   ├── daemon.py                       # MODIFY
│   └── lineage.py                      # NEW
├── observability/daemon_metrics.py     # MODIFY
└── learning/evolution/                 # 不修改 fsm.py 内部算法

src/api/routes/metrics.py               # MODIFY
web/src/pages/memory/components/SkillsGrid.tsx  # MODIFY

tests/
├── test_lineage.py                     # NEW
├── test_daemon_lineage_integration.py  # NEW
├── test_daemon_signal_handler.py       # NEW
└── test_e2e_git_lineage.py             # NEW
```

## 开发者使用场景

### 场景 1：观察 daemon 跑出的 lineage commit

```bash
# 跑 daemon
uv run arena evolution-daemon --once

# 查看 evolution branch 历史
git log evolution --oneline | head -5
# d4f8a01 evolution: daemon run summary
# ...

# 查看完整 commit message
git log evolution -1 --format=%B
# evolution: daemon run summary
#
# Pareto: archived=12 processed=50
# Regime: changed=33 total=142
# Skill proposal: drafts_created=2 agents_checked=4
#
# Auto-Generated-By: spec-020c
```

### 场景 2：审计某条 rule 状态变化历史

```bash
git log evolution --grep="rule_id=foo"
# 输出该 rule 所有 maturity 转换
```

### 场景 3：观察 lineage 健康度

```bash
curl http://localhost:8000/metrics | grep evolution_commit
# evolution_commit_count_24h 1.0
# evolution_commit_failure_rate_24h 0.0
```

### 场景 4：daemon graceful shutdown

```bash
# 启动 daemon
docker compose up -d evolution-daemon

# 发 SIGTERM（docker stop 即 SIGTERM）
docker compose stop evolution-daemon

# daemon ≤ 30s 内 graceful exit（等当前 run_once 完成）
docker compose logs evolution-daemon | tail -5
# [INFO] daemon: SIGTERM received, waiting for current run_once to finish
# [INFO] daemon: scheduler shutdown complete
# [INFO] daemon: redis closed
# [INFO] daemon: OTel flush complete, exit 0
```

### 场景 5：本地单测 GitLineageHook

```python
import pytest
from pathlib import Path
from cryptotrader.ops.lineage import GitLineageHook

@pytest.fixture()
def temp_repo(tmp_path):
    # subprocess git init in tmp_path
    ...

def test_commit_changes_creates_orphan_evolution_branch(temp_repo):
    hook = GitLineageHook(branch="evolution", repo_path=temp_repo)
    # 触发 daemon-style commit
    result = hook.commit_changes({"type": "daemon", "actions": [{"name": "pareto", "status": "PASS", "details": {"archived_count": 1}}]})
    assert result.success
    assert result.commit_sha is not None
    # 验证 evolution branch 存在
    branches = subprocess.check_output(["git", "branch", "-a"], cwd=temp_repo).decode()
    assert "evolution" in branches

def test_commit_failure_returns_soft_fail():
    # mock subprocess.run 抛 CalledProcessError
    ...
```

### 场景 6：测试 SIGTERM graceful shutdown

```python
@pytest.mark.asyncio
async def test_run_forever_sigterm_graceful_shutdown():
    daemon = EvolutionDaemon(config=cfg)
    task = asyncio.create_task(daemon.run_forever())
    await asyncio.sleep(0.1)  # 让 daemon 启动
    # 模拟 SIGTERM
    daemon._shutdown_flag.set()
    await asyncio.wait_for(task, timeout=30)  # ≤ 30s 内 exit
    # 验证 OTel flushed / redis closed
```

### 场景 7：前端 a11y 验证

```bash
# Vitest
cd web && pnpm vitest run src/pages/memory/components/SkillsGrid.test.tsx

# 手工验证（屏幕阅读器模拟）
# Chrome DevTools > Accessibility tab
# 选中 regime badge → 应显示 "Regime: high_funding"
```

## 验证清单（C4 完成后跑）

```bash
# SC-L1 + SC-L2: trailer
git log evolution -1 --format=%B | grep "Auto-Generated-By: spec-020c"

# SC-L3: time.sleep 清零
grep -n "time.sleep" src/cryptotrader/ops/daemon.py
# 期望：返回空

# SC-L4: SIGTERM graceful
uv run python -m pytest tests/test_daemon_signal_handler.py --no-cov -v

# SC-L5: aria-label
grep -c "aria-label" web/src/pages/memory/components/SkillsGrid.tsx
# 期望：≥ 3

# SC-L6: soft fail
uv run python -m pytest tests/test_lineage.py::test_commit_failure_soft_fail --no-cov -v

# SC-L7: 全套回归
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
# 期望：≥ 2439 passed / 0 failed

# SC-L10: commit count
git log --oneline 023-git-lineage..main | wc -l
# 期望：≤ 4
```

## Trilogy 终结

spec 020c 落地后 trilogy 完整收尾：
- ✅ D-ENG-01 reflect daemon（spec 020b）
- ✅ D-ENG-02 git lineage（spec 020c）
- ✅ 3 高 ROI P2 advisory（asyncio.sleep / SIGTERM / a11y）

剩余 6 项 P2 advisory（TimeoutError 双检 / `raise exc` / 类型注解 / e2e 断言路径 / `llm_call_failed` deprecate / staging step 4 集成测试）作为 fix-only PR backlog 后续单独处理。
