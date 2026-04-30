# Contract: `[scheduler].pairs` TOML Schema

**Config file**: `config/local.toml` / `config/default.toml`
**Section**: `[scheduler]`
**Field**: `pairs`

## Two Accepted Forms

### Form A — Legacy `list[str]` (backwards-compatible, all spot)

```toml
[scheduler]
pairs = ["BTC/USDT", "ETH/USDT"]
```

**Semantics**: each string is a ccxt unified spot symbol. Internally parsed into:

```python
[
    Pair(base="BTC", quote="USDT", ccxt_symbol="BTC/USDT"),
    Pair(base="ETH", quote="USDT", ccxt_symbol="ETH/USDT"),
]
```

All `market_type == "spot"`.

### Form B — New `list[dict]` (per-pair market type)

```toml
[[scheduler.pairs]]
symbol = "BTC/USDT"
# market defaults to "spot" if omitted

[[scheduler.pairs]]
symbol = "ETH/USDT"
market = "swap"
settle = "USDT"

[[scheduler.pairs]]
symbol = "BTC/USD"
market = "swap"
settle = "BTC"   # inverse perpetual
```

**Semantics**: each table-array element constructs a `Pair`:

```python
[
    Pair(base="BTC", quote="USDT", ccxt_symbol="BTC/USDT"),
    Pair(base="ETH", quote="USDT", ccxt_symbol="ETH/USDT:USDT"),
    Pair(base="BTC", quote="USD",  ccxt_symbol="BTC/USD:BTC"),
]
```

## Field Schema (Form B)

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `symbol` | str | yes | — | must contain `/`; format `BASE/QUOTE` |
| `market` | str | no | `"spot"` | one of `"spot" / "swap" / "future"` |
| `settle` | str | when `market != "spot"` | — | currency code (e.g. `"USDT"`, `"BTC"`) |

## Validation Rules

1. **Homogeneous array**: TOML cannot mix `str` and `dict` in one array. Either all-strings (Form A) or all-tables (Form B). Mixed raises `ConfigurationError`.
2. **`market` value**: only `"spot" / "swap" / "future"` accepted. `"option"` raises `NotImplementedError`.
3. **`settle` mandatory** when `market != "spot"`. Missing raises `ConfigurationError("scheduler.pairs[N].settle required when market != 'spot'")`.
4. **`symbol` format**: must match regex `^[A-Z0-9]+/[A-Z0-9]+$` (BASE/QUOTE letters/digits only). Bad format raises `ConfigurationError`.
5. **No duplicate canonical**: two pairs whose `Pair.canonical()` collide raises `ConfigurationError("duplicate pair: BTC/USDT")`.

## Resulting `cfg.scheduler.pairs` Type

```python
@dataclass
class SchedulerConfig:
    pairs: list[Pair] = field(default_factory=list)
    # ... (other fields unchanged)
```

## Migration Path

| Step | User Action |
|---|---|
| Day 0 | Existing `pairs = ["BTC/USDT"]` works unchanged → all spot |
| Day 1 | User edits to `[[scheduler.pairs]]` table-array form when they need perp |
| Day N | Form A (list[str]) remains supported indefinitely; no deprecation planned |

## Boot Log (FR-103)

On startup, scheduler emits one structured log line:

```json
{
  "event": "pair_init",
  "spot": ["BTC/USDT"],
  "swap": ["ETH/USDT:USDT", "BTC/USD:BTC"],
  "future": [],
  "level": "info"
}
```

For grep-friendly ops verification.
