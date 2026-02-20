# Risk Management

## 11 Hard Checks (all must pass)

1. **MaxPositionSize** — Single pair max 10% of portfolio
2. **MaxTotalExposure** — Total exposure max 50%
3. **DailyLossLimit** — Daily loss > 3% triggers circuit breaker
4. **DrawdownLimit** — Max drawdown 10% forces close all
5. **CVaR** — 60-day CVaR(95) must stay under 5%
6. **Correlation** — Avoid correlated position concentration
7. **Cooldown** — 60min between same-pair trades
8. **VolatilityGate** — 5min drop > 5% pauses trading
9. **FundingRateGate** — Funding > 0.1%/8h pauses
10. **RateLimit** — Max 6/hour, 20/day
11. **ExchangeHealth** — API latency > 2s pauses

## Circuit Breaker

Triggered by daily loss limit. Requires manual reset via Redis or dashboard.

## Redis Fallback

When Redis is unavailable, cooldown and rate limit checks **reject conservatively**.

## Configuration

All parameters in `config/risk.toml`. See [configuration.md](configuration.md).
