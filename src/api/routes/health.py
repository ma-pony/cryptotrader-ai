"""Health and metrics endpoints."""

import os
import time

from fastapi import APIRouter

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health():
    import redis.asyncio as aioredis

    checks = {"api": "ok"}
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            r = aioredis.from_url(redis_url)
            await r.ping()
            checks["redis"] = "ok"
            await r.aclose()
        except Exception:
            checks["redis"] = "unavailable"

    db_url = os.environ.get("DATABASE_URL")
    checks["db"] = "configured" if db_url else "not_configured"

    status = "ok" if checks["api"] == "ok" else "degraded"
    return {"status": status, "checks": checks}


@router.get("/metrics")
async def metrics():
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(os.environ.get("DATABASE_URL"))
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
