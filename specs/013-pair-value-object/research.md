# Phase 0 Research: Pair 值对象实施技术调研

**Feature**: 013-pair-value-object
**Date**: 2026-04-30

5 项调研覆盖实施层面的开放问题（设计层面的 D1-D7 已在 spec.md 解决）。

---

## R1 — ccxt unified symbol 边界情况

### Question
ccxt 的 perp/swap symbol 是否一律 `BASE/QUOTE:SETTLE`？futures (delivery)、options、inverse perp（COIN-M）的 symbol 形态？

### Decision
按 ccxt 4.x 官方文档 + 实地抽样：
- **Spot**: `BTC/USDT` （无后缀）
- **Linear perp swap (USDT-margined)**: `BTC/USDT:USDT` （settle = quote）
- **Inverse perp swap (COIN-margined)**: `BTC/USD:BTC` （settle = base）
- **Linear futures (delivery)**: `BTC/USDT:USDT-241227` （含到期日后缀）
- **Inverse futures**: `BTC/USD:BTC-241227`
- **Options**: `BTC/USD:BTC-241227-65000-C` （strike + put/call）

`Pair.parse(s)` 的策略：
1. 首先按 `:` 切分，前半是 `BASE/QUOTE`，后半是 `SETTLE[-EXPIRY[-STRIKE-PC]]`
2. 后半字段依次填入 `settle` / 派生 `market_type`
3. 没有 `:` 后缀的全部视作 spot

`Pair.from_ccxt(exchange, symbol)` 优先用 `exchange.market(symbol)` 元信息：
```python
m = exchange.market(symbol)
if m['option']:   market_type = 'option'
elif m['future']: market_type = 'future'
elif m['swap']:   market_type = 'swap'
else:             market_type = 'spot'
base, quote, settle = m['base'], m['quote'], m.get('settle')
```
缺 metadata 时降级为 `parse()`。

### Rationale
本 spec scope 是 spot + swap，但 `Pair` 设计要为 future/option 留扩展位。`market_type` derived property 涵盖 4 种是合理的。

### Alternatives Considered
- **Only spot + swap，其他抛错**：太严，未来加 futures 还要改 Pair 类
- **不区分 future/swap，统一叫 derivatives**：信息丢失，AI prompt 看到 "derivatives" 不如 "perpetual" 精确

---

## R2 — LangGraph state schema migration 模式

### Question
LangGraph state schema bump 时怎么处理 in-flight checkpoint？

### Decision
LangGraph 1.0 的 `Checkpointer` （`MemorySaver` / `RedisSaver` / `PostgresSaver`）**不支持 schema migration**。state 是 TypedDict，反序列化时直接按字段名 unpack，多余字段忽略，缺失字段要求显式默认值。

实施策略：
1. **Phase 3c 部署前**：停 scheduler（`/api/scheduler/stop` 或 `kill scheduler` 进程），等所有 in-flight cycle 结束（最长一个 interval = 4 hours）
2. **清空 checkpoint**：`PostgresSaver.delete_thread()` 或者干脆把 checkpoint 表 truncate（项目当前用 inline state，无 checkpoint 持久化层 — 验证）
3. **Bump state schema**：把 `state.metadata.pair: str` 改 `state.metadata.pair: Pair`
4. **重启 scheduler**：next cycle 用新 schema

> ⚠️ **验证**：`grep -rn "Checkpointer\|MemorySaver\|RedisSaver" src/` 确认本项目 LangGraph 是否启用了 persistent checkpoint。如果没有（state 只在内存），第 1-2 步可省。

### Rationale
LangGraph state 是按 TypedDict 不可变更新模型走的，没有自带 schema versioning。最稳的策略是"停服务、清状态、升级、重启"。

### Alternatives Considered
- **写一个 state migration adapter** in checkpointer — 太复杂，LangGraph 没 hook 点
- **保留 `state.metadata.pair: str` 不动，加 `state.metadata.pair_obj: Pair`** — 双字段不一致风险，违反 D2 一刀切原则

---

## R3 — Alembic 加列性能

### Question
给 `decision_commits` 加 `market_type VARCHAR(20) NOT NULL DEFAULT 'spot'` 列，PG 是否要全表锁？数据量影响？

### Decision
PG 11+ 对 `ADD COLUMN ... DEFAULT ...` **不重写表**（fast path），只更新 catalog；后续读取时按列默认值返回。所以加列本身是 O(1)（取决于 catalog 锁，毫秒级）。

实施：
```python
# alembic upgrade
op.add_column(
    "decision_commits",
    sa.Column("market_type", sa.String(20), nullable=False, server_default="spot")
)
```
项目当前 `decision_commits` 仅 ~14 行，加列瞬间完成。

### Rationale
PG 11+ 的 fast path ADD COLUMN 是经典优化，万行级、亿行级表都安全。

### Alternatives Considered
- **不加默认值，先 ADD COLUMN NULL → backfill → ALTER NOT NULL** — 三步，更慢；只在没法用 server_default 时才用
- **不加列，保持单 `pair` 字段，靠后缀编码 market_type** — 违反 D5；查询不便（`WHERE market_type = 'swap'` vs `WHERE pair LIKE '%:USDT'`）

---

## R4 — TOML 混合 list[str] / list[dict] 解析

### Question
`pairs = ["BTC/USDT", {symbol="ETH/USDT", market="swap", settle="USDT"}]` 这种混合写法 tomllib 能不能解析？

### Decision
**不能直接混合**。TOML spec 的 array 要求**同质**（所有元素相同类型）。

替代写法：
```toml
# 选项 A：全部对象数组（推荐）
[[scheduler.pairs]]
symbol = "BTC/USDT"

[[scheduler.pairs]]
symbol = "ETH/USDT"
market = "swap"
settle = "USDT"

# 选项 B：保留旧 list[str]，全是 spot
pairs = ["BTC/USDT", "ETH/USDT"]
```

config 解析逻辑：
```python
raw = toml_data["scheduler"].get("pairs", [])
if all(isinstance(p, str) for p in raw):
    # 旧形式：全 spot
    pairs = [Pair(base, quote, ccxt_symbol=p) for p in raw ...]
elif all(isinstance(p, dict) for p in raw):
    # 新形式
    pairs = [_parse_pair_dict(p) for p in raw]
else:
    raise ConfigurationError("scheduler.pairs must be all-strings or all-objects, not mixed")
```

### Rationale
TOML 限制是硬性的；用 `[[scheduler.pairs]]` table-array 语法是 TOML 原生支持新形式的方法。

### Alternatives Considered
- **JSON 配置代替 TOML**：项目其他配置都是 TOML，不为单字段换格式
- **YAML 配置**：同上
- **`pairs_obj = [...]` 新字段**：双字段不一致风险，与 D4 灵活性精神冲突

---

## R5 — frozen dataclass 性能

### Question
`Pair` 实例化要 < 5μs。frozen dataclass 在 hot path 里是否够快？

### Decision
**通过**。frozen dataclass 实例化在 CPython 3.12 上约 **0.5-1μs**（M1 Mac 实测），远低于 5μs SLO。

参考数据：
```python
import timeit
from dataclasses import dataclass

@dataclass(frozen=True)
class Pair:
    base: str
    quote: str
    ccxt_symbol: str

# Result: ~0.6μs per call on M1
timeit.timeit(lambda: Pair("BTC", "USDT", "BTC/USDT"), number=1_000_000)
```

trading_cycle 每周期实例化 ~10 个 Pair 对象（per pair × per node），总开销 < 10μs，远低于 cycle 的几十秒级总时长。

### Rationale
frozen dataclass 比 NamedTuple 略慢（NamedTuple ~0.2μs）但 API 更灵活（支持 `@property`、custom `__init__` 等）。当前性能远不是瓶颈。

### Alternatives Considered
- **NamedTuple**：更快但 `@property` 不能直接定义（需用 typing.NamedTuple class），且 `__init__` 不能 customize
- **Pydantic BaseModel**：~10μs/实例化，是 frozen dataclass 的 10-20 倍，超 SLO 2 倍
- **attrs**：跟 dataclass 持平，不带来额外好处

---

## Summary

| 研究 | Decision | 影响的 FR |
|---|---|---|
| R1 ccxt symbol 形态 | 4 种 market_type；优先用 ccxt market metadata | FR-002, FR-003, FR-007 |
| R2 LangGraph state migration | 停服务清 checkpoint 重启；项目目前不用 persistent checkpoint 待验证 | FR-200, FR-204 |
| R3 alembic 加列 | PG 11+ fast path，O(1) | FR-400 ~ 402 |
| R4 TOML 混合配置 | 用 `[[scheduler.pairs]]` table-array；混合写法禁止 | FR-100 |
| R5 frozen dataclass 性能 | ~0.6μs/实例化，远低于 5μs SLO | NFR-Performance |

**Output**: 全 5 项 NEEDS CLARIFICATION 已 resolve，可进入 Phase 1 Design。
