# Quickstart：Pattern Cold-Start（spec 021）

## 落地后目录结构

```
src/cryptotrader/
├── learning/memory.py         # MODIFY: distill_patterns + 2 helper
├── ops/daemon.py              # MODIFY: _action_pattern_extraction
└── config.py                  # MODIFY: ExperienceConfig.min_cases_per_pattern

src/cli/main.py                # MODIFY: arena experience distill

config/default.toml            # MODIFY: [experience] + [evolution_daemon].actions

tests/
├── test_distill_patterns_cold_start.py    # NEW
├── test_pattern_slug_generation.py        # NEW
├── test_daemon_pattern_extraction.py      # NEW
├── test_cli_experience_distill.py         # NEW
└── test_e2e_pattern_cold_start.py         # NEW
```

## 开发者使用场景

### 场景 1：首次部署 backfill（一次性）

```bash
# 跑一次手动 backfill 246 cases → 创建首批 patterns
uv run arena experience distill --memory-dir agent_memory --cycles-window 200

# 期望输出
# cases_processed: 199
# patterns_created: 12
# patterns_updated: 0
# patterns_archived: 0

# 验证 patterns 文件创建
ls agent_memory/tech/patterns/*.md | head -5
```

### 场景 2：dashboard 验证

```bash
# API 返回非空
curl http://localhost:8003/api/memory/rules | python3 -m json.tool | head -20

# 期望：total > 0，items 含 maturity="observed" PatternRecord

# 前端访问
open http://localhost:5173/memory
# 看 RulesGrid section 显示实际数据（不再 0）
```

### 场景 3：daemon daily 自动跑

```bash
# 修改 docker-compose evolution-daemon 启用 daily cron（spec 020b 既有）
# 或 manual 触发
uv run arena evolution-daemon --once

# 期望输出
# [pareto]              PASS  archived=0 processed=12
# [regime]              PASS  changed=0 total=246
# [skill_proposal]      PASS  agents_checked=4 drafts_created=0
# [pattern_extraction]  PASS  new=2 updated=8 archived=0    ← spec 021 新增
```

### 场景 4：daemon docker 模式启动

```bash
# 启动 daemon docker container（spec 020b 既有）
docker compose up -d evolution-daemon

# 每天 UTC 0:00 自动跑 4 actions（含 pattern_extraction）
docker compose logs -f evolution-daemon
```

### 场景 5：本地单测

```python
# tests/test_distill_patterns_cold_start.py
def test_cold_start_creates_pattern_when_frequency_above_threshold(tmp_path):
    """5 个 cases 引用 'Volume Spike' → 创建 1 个 pattern."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    for i in range(5):
        (cases_dir / f"case-{i}.md").write_text(_make_case_body(applied_patterns={"tech": ["Volume Spike"]}))
    run = distill_patterns(memory_dir=tmp_path, cycles_window=10)
    assert run.patterns_created == 1
    assert (tmp_path / "tech" / "patterns" / "volume-spike.md").exists()

def test_cold_start_below_threshold_no_pattern_created(tmp_path):
    """4 个 cases（< default 5）→ 不创建."""
    ...
    assert run.patterns_created == 0

def test_pattern_slug_collision_uses_n_suffix(tmp_path):
    """同名 pattern 已存在 → slug 加 -2 后缀."""
    (tmp_path / "tech" / "patterns").mkdir(parents=True)
    (tmp_path / "tech" / "patterns" / "volume-spike.md").write_text("...")  # existing
    ...
    assert (tmp_path / "tech" / "patterns" / "volume-spike-2.md").exists()
```

## 验证清单（C4 完成后跑）

```bash
# SC-P1: CLI distill 创建 ≥ 1 pattern
uv run arena experience distill --cycles-window 200
echo $?  # 应为 0

# SC-P2: ≥ 3 pattern 文件总数
find agent_memory/{tech,chain,news,macro}/patterns -name "*.md" 2>/dev/null | wc -l
# 期望：≥ 3

# SC-P3: API 返回非空
curl -s http://localhost:8003/api/memory/rules | python3 -c "import sys,json;d=json.load(sys.stdin);print('total:',d['total']);print('items:',len(d['items']))"

# SC-P4: daemon --once 4 actions PASS
uv run arena evolution-daemon --once

# SC-P5: 单测 PASS
uv run python -m pytest tests/test_distill_patterns_cold_start.py tests/test_pattern_slug_generation.py tests/test_daemon_pattern_extraction.py tests/test_cli_experience_distill.py -v --no-cov

# SC-P6: e2e PASS
uv run python -m pytest tests/test_e2e_pattern_cold_start.py -v --no-cov

# SC-P7: 全套回归
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
# 期望：≥ 2458 passed / 0 failed

# SC-P10: commit 数 ≤ 4
git log --oneline 024-pattern-cold-start..main | wc -l
```

## Trilogy 数据链补完后效果

```
[before spec 021]
cases/ (246) → distill → patterns/ (0) ❌ → evaluate (空) → transitions (0) → dashboard (空)

[after spec 021]
cases/ (246) → distill (含 cold-start) → patterns/ (~12) ✅ → evaluate (FSM 跑) → transitions (≥1) → dashboard (有数据)
```

整 trilogy 进化系统第一次进入"自循环"运转状态。后续 spec 020b daemon daily 跑会持续蒸馏新 patterns + spec 018 FSM 自动 promote + spec 019 skill proposal 自动 trigger。
