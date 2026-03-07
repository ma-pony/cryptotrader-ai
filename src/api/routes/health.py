"""Health and metrics endpoints."""

import time

from fastapi import APIRouter

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health():
    import redis.asyncio as aioredis

    from cryptotrader.config import load_config

    config = load_config()

    checks = {"api": "ok"}
    redis_url = config.infrastructure.redis_url
    if redis_url:
        try:
            r = aioredis.from_url(redis_url)
            await r.ping()
            checks["redis"] = "ok"
            await r.aclose()
        except Exception:
            checks["redis"] = "unavailable"

    db_url = config.infrastructure.database_url
    checks["db"] = "configured" if db_url else "not_configured"

    status = "ok" if checks["api"] == "ok" else "degraded"
    return {"status": status, "checks": checks}


@router.get("/metrics")
async def metrics():
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import JournalStore

    config = load_config()

    store = JournalStore(config.infrastructure.database_url)
    commits = await store.log(limit=1000)
    total = len(commits)
    wins = sum(1 for c in commits if c.pnl is not None and c.pnl > 0)
    with_pnl = sum(1 for c in commits if c.pnl is not None)
    win_rate = wins / with_pnl if with_pnl else 0.0

    divergences = [c.divergence for c in commits if c.divergence]
    avg_div = sum(divergences) / len(divergences) if divergences else 0.0

    return {
        "decisions_total": total,
        "win_rate": round(win_rate, 3),
        "avg_divergence": round(avg_div, 3),
        "uptime_seconds": round(time.time() - _start_time),
    }
