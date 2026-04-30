# Contract: `Pair` Public API

**Module**: `src/cryptotrader/pair.py`

```python
from typing import Any, Literal


@dataclass(frozen=True)
class Pair:
    base: str             # "BTC"
    quote: str            # "USDT"
    ccxt_symbol: str      # "BTC/USDT" or "BTC/USDT:USDT"

    def __post_init__(self) -> None: ...
        # raises ValueError if base/quote empty or ccxt_symbol malformed

    # ── Constructors ──

    @classmethod
    def parse(cls, s: str) -> "Pair":
        """Parse a canonical (ccxt unified) string into a Pair.

        >>> Pair.parse("BTC/USDT")
        Pair(base='BTC', quote='USDT', ccxt_symbol='BTC/USDT')
        >>> Pair.parse("BTC/USDT:USDT")
        Pair(base='BTC', quote='USDT', ccxt_symbol='BTC/USDT:USDT')

        Raises:
            ValueError: when ``s`` lacks `/` or has malformed structure.
        """

    @classmethod
    def from_ccxt(cls, exchange: Any, symbol: str) -> "Pair":
        """Build a Pair from a ccxt exchange's market metadata.

        Falls back to ``parse(symbol)`` when ``exchange.market(symbol)`` raises
        or returns an empty dict (rare; only when symbol is unknown to ccxt).

        Raises:
            NotImplementedError: market type is `option` (out of spec 013 scope).
        """

    # ── Serialization ──

    def to_ccxt(self) -> str:
        """Return the ccxt unified symbol (= ``self.ccxt_symbol``)."""

    def canonical(self) -> str:
        """Project canonical string form (≡ ``to_ccxt()``)."""

    def display(self) -> str:
        """Human/AI friendly form.

        >>> Pair.parse("BTC/USDT").display()
        'BTC/USDT'
        >>> Pair.parse("BTC/USDT:USDT").display()
        'BTC/USDT (perp)'
        >>> Pair.parse("BTC/USDT:USDT-241227").display()
        'BTC/USDT (futures 241227)'
        """

    def __str__(self) -> str:
        """Equivalent to ``canonical()``."""

    # ── Derived properties ──

    @property
    def market_type(self) -> Literal["spot", "swap", "future", "option"]:
        """Inferred from ``ccxt_symbol`` suffix structure."""

    @property
    def settle(self) -> str | None:
        """Settlement currency for derivatives; None for spot."""

    # ── Identity ──

    def __hash__(self) -> int: ...
        # auto-generated; pairs hash by (base, quote, ccxt_symbol)

    def __eq__(self, other: object) -> bool: ...
        # auto-generated; pairs equal iff all 3 fields match
```

## Invariants

| # | Description | Enforcement |
|---|---|---|
| I1 | `ccxt_symbol` contains exactly one `/` | `__post_init__` |
| I2 | `base` and `quote` are non-empty | `__post_init__` |
| I3 | `ccxt_symbol` starts with `f"{base}/{quote}"` | `__post_init__` |
| I4 | Equal pairs hash equal | dataclass auto |
| I5 | `parse(p.canonical()) == p` for any constructed `p` (round-trip) | unit test FR-009 |
| I6 | `Pair` instances are immutable | `frozen=True` |

## Performance

- `Pair.__init__` ≤ 5μs (NFR-Performance)
- `Pair.market_type` ≤ 1μs (string suffix check, no allocation)
- `Pair.parse` ≤ 10μs (one string split + Pair construction)
- `Pair.from_ccxt` ≤ 50μs (depends on ccxt `market()` lookup, dict access)

## Error Taxonomy

| Exception | When |
|---|---|
| `ValueError` | malformed input string in `parse()` or invariant violation in `__post_init__` |
| `NotImplementedError` | option market in `from_ccxt()` |
| `KeyError` | `from_ccxt()` with unknown symbol AND no fallback path (very rare) |

## Examples

```python
# Spot
btc = Pair.parse("BTC/USDT")
assert btc.market_type == "spot"
assert btc.settle is None
assert btc.canonical() == "BTC/USDT"
assert btc.display() == "BTC/USDT"

# Linear perpetual
btc_perp = Pair.parse("BTC/USDT:USDT")
assert btc_perp.market_type == "swap"
assert btc_perp.settle == "USDT"
assert btc_perp.canonical() == "BTC/USDT:USDT"
assert btc_perp.display() == "BTC/USDT (perp)"

# Inverse perpetual
btc_inverse = Pair.parse("BTC/USD:BTC")
assert btc_inverse.settle == "BTC"

# As dict key
positions: dict[Pair, dict] = {btc: {...}, btc_perp: {...}}
assert positions.get(Pair.parse("BTC/USDT")) is not None
```
