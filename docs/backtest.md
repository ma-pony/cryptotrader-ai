# Backtest Guide

## Usage

```bash
arena backtest --pair BTC/USDT --start 2025-01-01 --end 2025-12-31 --interval 4h
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pair` | BTC/USDT | Trading pair |
| `--start` | required | Start date (YYYY-MM-DD) |
| `--end` | required | End date (YYYY-MM-DD) |
| `--interval` | 4h | Candle interval (1m/5m/15m/1h/4h/1d) |
| `--capital` | 10000 | Initial capital |

## Output

- Total return, Sharpe ratio, max drawdown, win rate
- Trade list with entry/exit/PnL
- Equity curve (viewable in dashboard)

## Anti-Lookahead

Strict time ordering — each step only sees data up to current timestamp.

## Data Caching

Historical OHLCV cached in `~/.cryptotrader/ohlcv_cache.db` (SQLite) to avoid re-fetching.
