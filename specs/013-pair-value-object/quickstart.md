# Quickstart: Pair 值对象重构

**Feature**: 013-pair-value-object
**Audience**: 开发者本地验证 + 真盘 sandbox 验证

---

## Prerequisites

```bash
# Backend
uv sync --extra test --extra otel
.venv/bin/python --version  # 3.12+

# Frontend (Phase 5)
cd web && pnpm install

# Infra
docker compose up -d postgres redis  # 已配
```

---

## Phase 1: Pair Module 验证

```bash
# 单测全绿
.venv/bin/pytest tests/test_pair.py -v --no-cov

# 性能 benchmark
.venv/bin/pytest tests/test_pair.py -k benchmark --benchmark-only

# Doctest 覆盖
.venv/bin/python -m doctest src/cryptotrader/pair.py -v
```

期望：
- 全部用例 pass
- `Pair.__init__` < 5μs（M1 实测 ~0.6μs）

---

## Phase 2: Config 升级验证

### 测试旧配置（向后兼容）

```toml
# config/local.toml
[scheduler]
pairs = ["BTC/USDT", "ETH/USDT"]
```

```bash
.venv/bin/python -c "
from cryptotrader.config import load_config
cfg = load_config()
for p in cfg.scheduler.pairs:
    print(f'  {p.canonical()}  market={p.market_type}')
"
```

期望输出：
```
  BTC/USDT  market=spot
  ETH/USDT  market=spot
```

### 测试新配置（per-pair market type）

```toml
[[scheduler.pairs]]
symbol = "BTC/USDT"
market = "swap"
settle = "USDT"
```

```bash
.venv/bin/python -c "...同上..."
```

期望输出：
```
  BTC/USDT:USDT  market=swap
```

启动 API 时应见 `pair_init` 日志：

```bash
AUTH_MODE=disabled .venv/bin/python -m uvicorn api.main:app --port 8003 --app-dir src 2>&1 | grep pair_init
```

---

## Phase 3a/3b/3c: Trading Cycle 真盘验证

### 起完整服务

```bash
pkill -9 -f "uvicorn api.main"
AUTH_MODE=disabled nohup .venv/bin/python -m uvicorn api.main:app \
  --host 0.0.0.0 --port 8003 --no-access-log --app-dir src \
  > /tmp/cryptotrader-logs/api.log 2>&1 &
disown
until curl -sf http://127.0.0.1:8003/health > /dev/null; do sleep 1; done
```

### 触发一次 cycle（等 4h interval 或重启）

```bash
# 看 cycle 完成 log
tail -f /tmp/cryptotrader-logs/api.log | grep -E "Cycle complete|cycle snapshot|place_order"
```

期望：
- log 含 `Cycle complete [BTC/USDT] action=close confidence=...`
- log 含 `Placing order: sell BTC/USDT 0.02 @ ...`（perp 真下单）
- DB `decision_commits.order_data` 字段非 null（之前是 null = no-op）

### 验证 acceptance scenarios

对照 spec.md User Story 1 三条：

1. ✅ `close` verdict + 已有 0.02 BTC perp → `place_order` symbol = `"BTC/USDT:USDT"`，amount = 0.02
2. ✅ `long scale=0.5` + 已有 0.02 → 下单 amount = `target - 0.02`
3. ✅ `market_type = "spot"` 配置 → ccxt symbol 不带 `:USDT`

```bash
# 查最近 commit
PGPASSWORD=123456 psql -h localhost -U postgres -d cryptotrader -c "
SELECT to_char(timestamp, 'HH24:MI:SS') t,
       pair, market_type, verdict::jsonb->>'action' AS act,
       order_data IS NOT NULL AS placed
FROM decision_commits ORDER BY timestamp DESC LIMIT 5;
"
```

---

## Phase 4: DB Migration 验证

```bash
# Apply migration
.venv/bin/alembic upgrade head

# 验证列已加
PGPASSWORD=123456 psql -h localhost -U postgres -d cryptotrader -c "\d portfolios"
# 应见 market_type 列，default 'spot'

PGPASSWORD=123456 psql -h localhost -U postgres -d cryptotrader -c "\d decision_commits"
# 同上

# 验证存量数据完整
PGPASSWORD=123456 psql -h localhost -U postgres -d cryptotrader -c "
SELECT pair, market_type, count(*) FROM portfolios GROUP BY 1, 2;
"
# 期望：所有存量 row market_type='spot'

# Downgrade test (alembic 反向迁移)
.venv/bin/alembic downgrade -1
.venv/bin/alembic upgrade head
```

---

## Phase 5: Frontend 验证

### 后端响应字段

```bash
curl -sS http://127.0.0.1:8003/api/portfolio/snapshot | python3 -m json.tool | head -20
```

期望 positions[0] 含：
- `pair`: `"BTC/USDT"` 或 `"BTC/USDT:USDT"`
- `pair_display`: `"BTC/USDT"` 或 `"BTC/USDT (perp)"`
- `market_type`: `"spot"` 或 `"swap"`

### 前端组件单测

```bash
cd web
pnpm test pair-badge.test.tsx
```

期望全绿。

### 浏览器手测

```bash
cd web && pnpm dev
# 打开 http://localhost:5173/
# 1. Dashboard 持仓表 — 每行 pair 列含小徽章 [PERP]/[SPOT]
# 2. 点 Decisions 列表 — 详情面板头部 pair 也含徽章
# 3. Market view / Chat / TradingView — 仍用旧 string，不变（D6 范围外）
```

---

## "撤回 Phase 0 band-aid" 验证

Phase 3c 完成后：

```bash
# band-aid 函数应已删除
grep -rn "_canonical_pair" src/cryptotrader/execution/exchange.py
# 期望：0 命中

# 全套测试仍绿
.venv/bin/pytest tests --no-cov -q
```

---

## Rollback Plan

每个 phase 都设计为可独立回滚：

| Phase | Rollback |
|---|---|
| 1 | 删除 `src/cryptotrader/pair.py` + `tests/test_pair.py`，0 影响（无调用方） |
| 2 | `git revert <phase-2-commit>`，`config.py` 回滚 list[Pair] → list[str] |
| 3a | revert adapter；nodes 仍用 str |
| 3b | revert verdict + execution；其他 nodes 不受影响 |
| 3c | revert state schema bump；**重启 scheduler 清 in-flight checkpoint** |
| 4 | `alembic downgrade -1`，drop column |
| 5 | revert API + frontend；后端字段保留无害（旧前端不读） |

---

## Troubleshooting

### "No available channel" LLM 错误
不是本 spec 范围。见 [issue #LLM-gateway](#) 或检查 new-api distributor。

### `Pair.from_ccxt` raises NotImplementedError
当前只支持 spot / swap / future。`option` 市场需先扩展 `Pair.market_type` Literal。

### state checkpoint 反序列化失败
旧 checkpoint 用 `str` pair，新代码期望 `Pair`。Phase 3c 部署前必须 `pkill scheduler` 清空 in-flight state。

### DB migration 卡住
PG 11+ `ADD COLUMN ... DEFAULT` 是 fast path（毫秒级）。卡住一定是其他长事务持有表锁，先 `pg_stat_activity` 查。
