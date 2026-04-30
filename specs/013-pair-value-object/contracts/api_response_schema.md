# Contract: API Response Schema (Phase 5)

## `/api/portfolio/snapshot`

### Before (current)

```json
{
  "equity": 3979.94,
  "cash": 3979.94,
  "positions": [
    {
      "pair": "BTC/USDT",
      "side": "long",
      "size": 0.02,
      "avg_price": 84708.8
    }
  ]
}
```

### After (Phase 5)

```json
{
  "equity": 3979.94,
  "cash": 3979.94,
  "positions": [
    {
      "pair": "BTC/USDT:USDT",
      "pair_display": "BTC/USDT (perp)",
      "market_type": "swap",
      "side": "long",
      "size": 0.02,
      "avg_price": 84708.8
    }
  ]
}
```

### New Fields

| Field | Type | Description |
|---|---|---|
| `pair` | str | Canonical (ccxt unified symbol) — was already there but now may include `:SETTLE` |
| `pair_display` | str | Human/AI friendly form (`"BTC/USDT (perp)"` for swap) |
| `market_type` | str | One of `"spot" / "swap" / "future"` — derived from canonical |

### Backwards Compatibility

- Old frontend code reading `position.pair` continues to work; spot pairs still return `"BTC/USDT"`
- New frontend code uses `pair_display` for rendering and `market_type` for badge color

---

## `/api/decisions/{commit_hash}`

### Before (excerpt)

```json
{
  "commit_hash": "06ec8434...",
  "pair": "ETH/USDT",
  "verdict": { "action": "close" }
}
```

### After (Phase 5)

```json
{
  "commit_hash": "06ec8434...",
  "pair": "ETH/USDT:USDT",
  "pair_display": "ETH/USDT (perp)",
  "market_type": "swap",
  "verdict": { "action": "close" }
}
```

Same field semantics as portfolio snapshot.

---

## `/api/decisions` (list endpoint)

Each `DecisionListItem` gains the same three fields. URL filter `?pair=BTC/USDT` continues to match historical spot data; new perp filter requires `?pair=BTC/USDT:USDT&market_type=swap`.

---

## OpenAPI Schema Updates

### Pydantic models (Phase 5)

```python
# src/api/routes/portfolio_v2.py
class PositionOut(BaseModel):
    pair: str            # canonical (ccxt symbol)
    pair_display: str    # NEW
    market_type: Literal["spot", "swap", "future"]  # NEW
    side: Literal["long", "short"]
    size: float
    avg_price: float
    # ... other fields unchanged

# src/api/routes/decisions.py
class DecisionListItem(BaseModel):
    commit_hash: str
    ts: str
    pair: str            # canonical
    pair_display: str    # NEW
    market_type: Literal["spot", "swap", "future"]  # NEW
    # ... other fields
```

## Frontend Type Mirror

```ts
// web/src/lib/api/types.ts
export interface PositionOut {
  pair: string;            // canonical
  pair_display: string;    // NEW
  market_type: 'spot' | 'swap' | 'future';  // NEW
  side: 'long' | 'short';
  size: number;
  avg_price: number;
}
```

## `<PairBadge>` Component Contract

```tsx
// web/src/components/PairBadge.tsx
interface PairBadgeProps {
  pair_display: string;
  market_type: 'spot' | 'swap' | 'future';
  className?: string;
}

// Renders:
// "BTC/USDT (perp)"  [PERP]   ← swap, badge yellow
// "BTC/USDT"         [SPOT]   ← spot, badge gray
// "BTC/USDT (futures)" [FUT]  ← future, badge blue
```

## Acceptance

- ✅ `/api/portfolio/snapshot` returns 3 fields per position
- ✅ `/api/decisions/*` returns 3 fields top-level
- ✅ TypeScript DTO mirrors backend
- ✅ `<PairBadge>` rendered in `<PortfolioPositions>` and `<DecisionDetail>` (D6)
- ⚠️ Other views (market view, chat, TradingView widget) continue using `pair` string; **out of scope for this spec**
