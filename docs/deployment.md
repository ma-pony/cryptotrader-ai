# Deployment

## Docker

```bash
docker compose up -d
```

Services:
- **postgres** — Decision journal + portfolio storage
- **redis** — Risk state (cooldowns, circuit breaker)
- **app** — Trading engine
- **dashboard** — Streamlit UI on port 8501

## Environment

Set in `.env` or docker-compose environment:

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/cryptotrader
REDIS_URL=redis://redis:6379
OPENAI_API_KEY=sk-...
```

## Production Notes

- Use proper secrets management for API keys
- Set `execution.engine = "live"` in config for real trading
- Monitor via dashboard or webhook notifications
