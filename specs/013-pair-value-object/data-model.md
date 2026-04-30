# Phase 1 Data Model: Pair 值对象 + DB schema 变更

**Feature**: 013-pair-value-object
**Date**: 2026-04-30

## New Entity: `Pair`

### Definition

```python
# src/cryptotrader/pair.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Pair:
    """A trading pair with explicit market type semantics.

    Single source of truth = ccxt unified symbol. spot pairs are
    `BASE/QUOTE` (no suffix); derivatives are `BASE/QUOTE:SETTLE` with
    optional date / strike encoding for futures and options.
    """

    base: str             # "BTC"
    quote: str            # "USDT"
    ccxt_symbol: str      # "BTC/USDT" 或 "BTC/USDT:USDT"
```

### Invariants

| Invariant | Enforcement |
|---|---|
| `ccxt_symbol` 必须含 `/` | `__post_init__` 校验，缺失抛 `ValueError` |
| `base` 和 `quote` 必须非空 | 同上 |
| `ccxt_symbol` 起首应以 `f"{base}/{quote}"` 开头 | 同上（catch 笔误） |
| frozen — 实例化后字段不可变 | `frozen=True` 自带 |
| 可作 dict key | `frozen=True` 自动生成 `__hash__` / `__eq__` |

### Public API

| Method / Property | Signature | Behavior |
|---|---|---|
| `Pair.parse(s)` | `classmethod (str) -> Pair` | 从 canonical 字符串解析；没有 `:` 视作 spot |
| `Pair.from_ccxt(exchange, sym)` | `classmethod (Any, str) -> Pair` | 用 `exchange.market(sym)` 元信息构造；缺 metadata 降级为 `parse()` |
| `Pair.to_ccxt()` | `() -> str` | 返回 `self.ccxt_symbol`（已是 ccxt 形式） |
| `Pair.canonical()` | `() -> str` | ≡ `to_ccxt()` |
| `Pair.display()` | `() -> str` | UI/AI 友好：spot 返回 `"BTC/USDT"`，swap 返回 `"BTC/USDT (perp)"` |
| `Pair.market_type` | `@property -> str` | `"spot" / "swap" / "future" / "option"` 推断 |
| `Pair.settle` | `@property -> str \| None` | 解析 `ccxt_symbol` 提取；spot 为 None |
| `Pair.__hash__` | auto | dataclass 自动 |
| `Pair.__eq__` | auto | dataclass 自动 |
| `Pair.__str__` | auto override | 等同 `canonical()`，方便 logger / format |

### State Transitions

`Pair` 无状态变更（frozen value object）。

### Relationships

- `cfg.scheduler.pairs: list[Pair]` — 配置层
- `state.metadata.pair: Pair` — LangGraph state（Phase 3c 后）
- `Order.pair: str` ≡ `pair.canonical()` — 序列化字符串（D3）

---

## Modified Entities (DB Schema)

### `portfolios` 表 (Phase 4)

| Column | Type | Default | Constraints | 备注 |
|---|---|---|---|---|
| `id` | VARCHAR | — | PK | 不变 |
| `pair` | VARCHAR(20) | — | NOT NULL | 不变；存量 row `"BTC/USDT"` 保留 |
| `amount` | DOUBLE PRECISION | — | nullable | 不变 |
| `avg_price` | DOUBLE PRECISION | — | nullable | 不变 |
| `updated_at` | TIMESTAMP WITH TZ | — | nullable | 不变 |
| **`market_type`** | **VARCHAR(20)** | **`'spot'`** | **NOT NULL** | **NEW** |

### `decision_commits` 表 (Phase 4)

新增同样的 `market_type VARCHAR(20) NOT NULL DEFAULT 'spot'` 列。其他列不变。

### `portfolio_snapshots` 表 (Phase 4)

`portfolio_snapshots` 当前 schema 没有 `pair` 列（按 `account_id` 聚合）。**无需修改**。如未来加 per-pair snapshot，参照 portfolios 模式。

### Migration Strategy (D5)

```python
# migrations/versions/XXXX_add_market_type.py
def upgrade() -> None:
    op.add_column(
        "portfolios",
        sa.Column("market_type", sa.String(20), nullable=False, server_default="spot")
    )
    op.add_column(
        "decision_commits",
        sa.Column("market_type", sa.String(20), nullable=False, server_default="spot")
    )
    # 不动任何 row 数据；存量自动取默认值

def downgrade() -> None:
    op.drop_column("decision_commits", "market_type")
    op.drop_column("portfolios", "market_type")
```

---

## Modified Entities (In-Memory)

### `state["metadata"]["pair"]` (Phase 3c)

```diff
- pair: str  # "BTC/USDT"
+ pair: Pair  # Pair(base="BTC", quote="USDT", ccxt_symbol="BTC/USDT:USDT")
```

LangGraph state checkpoint 反序列化时（如启用 PostgresSaver）通过 `Pair.parse(saved_str)` 重建。

### `cfg.scheduler.pairs` (Phase 2)

```diff
@dataclass
class SchedulerConfig:
-   pairs: list[str] = field(default_factory=list)
+   pairs: list[Pair] = field(default_factory=list)
```

`load_config()` 解析 TOML 时按 R4 决议处理两种形式（list[str] 视作全 spot；list[dict] 解析 market/settle）。

### `Order.pair` (D3, 不变)

```python
# src/cryptotrader/models.py
@dataclass
class Order:
    pair: str  # 必须是 pair.canonical() 形式（即 ccxt unified symbol）
    ...
```

不改字段类型，但**所有构造点必须传 `pair.canonical()` 字符串**。

---

## Validation Rules

### Pair-level (FR-104)
- `market != "spot"` 时 `settle` 必填，否则启动时 raise `ConfigurationError`
- `Pair.parse()` 检测到 `:` 后缀但提取 settle 失败时 raise `ValueError`
- `Pair.from_ccxt()` 检测到 `option` market 时 raise `NotImplementedError`（本 spec scope 外）

### Cross-entity
- `Order.pair` 字符串必须能成功 `Pair.parse()`（journal 写入前 sanity check）
- `state.metadata.pair` 反序列化失败时 fallback 为 spot + 写 warning 日志

---

## Backwards Compatibility Matrix

| 升级前 | 升级后 | 兼容性 |
|---|---|---|
| 配置 `pairs = ["BTC/USDT"]` | 仍接受 | ✅ 视作 spot |
| 配置 `pairs = [{symbol="..."}]` | 新支持 | ✅ |
| DB row `pair="BTC/USDT"`（无 market_type 列） | 加列后默认 `'spot'` | ✅ |
| 现有 SQL `WHERE pair = 'BTC/USDT'` | 仍匹配存量 spot 数据 | ✅ |
| 新 SQL 查 perp `WHERE pair = 'BTC/USDT:USDT' AND market_type='swap'` | 双条件 | ✅ |
| 旧 frontend URL `?pair=BTC/USDT` | 解析为 spot Pair | ✅ |
| `Order(pair="BTC/USDT", ...)` 老调用点 | 仍 work（视作 spot canonical） | ✅ |
| `Order(pair="BTC/USDT:USDT", ...)` perp 用法 | 新 work | ✅ |

---

**Output**: 1 new entity (`Pair`)，3 表加 1 列，2 in-memory 字段类型升级，4 项 invariant，6 条向后兼容场景。
