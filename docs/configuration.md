# Configuration

All configuration lives in `config/` as TOML files.

## config/default.toml

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `mode.mode` | standalone/api/external | standalone | Operating mode |
| `execution.engine` | paper/live | paper | Trading engine |
| `models.analysis` | model name | gpt-4o-mini | Analysis agent model |
| `models.verdict` | model name | gpt-4o | Verdict model |
| `debate.max_rounds` | int | 3 | Max debate rounds |
| `debate.divergence_hold_threshold` | float | 0.7 | Hold if divergence exceeds |
| `scheduler.enabled` | bool | false | Enable auto-scheduling |
| `scheduler.pairs` | list | ["BTC/USDT","ETH/USDT"] | Pairs to trade |
| `scheduler.interval_minutes` | int | 240 | Run interval |
| `notifications.webhook_url` | string | "" | Webhook endpoint |

## config/risk.toml

See [risk-management.md](risk-management.md) for all risk parameters.

## Environment Variables

Override config via environment:
- `DATABASE_URL` — PostgreSQL async connection string
- `REDIS_URL` — Redis connection string
- Provider API keys: set in `[providers]` section or as env vars
