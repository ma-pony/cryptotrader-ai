"""Health and metrics endpoints."""

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health():
    from cryptotrader.config import load_config

    config = load_config()
    checks = {"api": "ok"}

    # Real Redis check
    redis_url = config.infrastructure.redis_url
    if redis_url:
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(redis_url)
            await r.ping()
            checks["redis"] = "ok"
            await r.aclose()
        except Exception:
            checks["redis"] = "unavailable"
    else:
        checks["redis"] = "not_configured"

    # Real DB check — execute a query
    db_url = config.infrastructure.database_url
    if db_url:
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine

            engine = create_async_engine(db_url, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            checks["db"] = "ok"
        except Exception:
            checks["db"] = "unavailable"
    else:
        checks["db"] = "not_configured"

    # Status: degraded if any configured service is unavailable
    degraded = any(v == "unavailable" for v in checks.values())
    status = "degraded" if degraded else "ok"
    code = 503 if degraded else 200

    return JSONResponse(
        status_code=code,
        content={
            "status": status,
            "checks": checks,
            "uptime_seconds": round(time.time() - _start_time),
        },
    )


@router.get("/metrics")
async def metrics(request: Request):
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
    uptime = round(time.time() - _start_time)

    # Prometheus text format if Accept header requests it
    accept = request.headers.get("accept", "")
    if "text/plain" in accept or "application/openmetrics" in accept:
        lines = [
            "# HELP cryptotrader_decisions_total Total trading decisions made.",
            "# TYPE cryptotrader_decisions_total counter",
            f"cryptotrader_decisions_total {total}",
            "# HELP cryptotrader_win_rate Ratio of profitable trades.",
            "# TYPE cryptotrader_win_rate gauge",
            f"cryptotrader_win_rate {win_rate:.3f}",
            "# HELP cryptotrader_avg_divergence Average agent divergence score.",
            "# TYPE cryptotrader_avg_divergence gauge",
            f"cryptotrader_avg_divergence {avg_div:.3f}",
            "# HELP cryptotrader_uptime_seconds Seconds since API started.",
            "# TYPE cryptotrader_uptime_seconds gauge",
            f"cryptotrader_uptime_seconds {uptime}",
            "",
        ]
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")

    return {
        "decisions_total": total,
        "win_rate": round(win_rate, 3),
        "avg_divergence": round(avg_div, 3),
        "uptime_seconds": uptime,
    }
