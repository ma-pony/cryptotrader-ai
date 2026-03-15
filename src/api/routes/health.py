"""Health check endpoint (Requirement 10.4).

Returns detailed component status for Docker orchestrator decisions:
- api        : always "ok" (the process is alive)
- db         : database connectivity via SELECT 1
- redis      : Redis connectivity via PING
- llm        : LLM API reachability via a lightweight HEAD/GET request

HTTP 200 when all configured components are healthy.
HTTP 503 when any configured component is "unavailable" (degraded).
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cryptotrader.config import load_config

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = time.time()

# ---------------------------------------------------------------------------
# Optional heavy dependencies -- imported at module level so tests can patch
# them by name (e.g. patch("api.routes.health.aioredis")).
# ---------------------------------------------------------------------------

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover -- optional dependency
    aioredis = None  # type: ignore[assignment]

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError:  # pragma: no cover -- optional dependency
    text = None  # type: ignore[assignment]
    create_async_engine = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers -- extracted so tests can mock them independently
# ---------------------------------------------------------------------------


async def _check_llm(base_url: str, api_key: str) -> str:
    """Probe the LLM API gateway with a lightweight GET request.

    Returns one of: "ok", "unavailable".
    HTTP 4xx (e.g. 401 Unauthorized) still means the endpoint is reachable,
    so any status < 500 is treated as "ok".
    """
    # Resolve probe URL: prefer explicit base_url, fallback to OpenAI
    probe_url = base_url.rstrip("/") if base_url else "https://api.openai.com"
    # /v1/models is a lightweight, cacheable listing endpoint
    if not probe_url.endswith("/v1/models"):
        probe_url = probe_url.rstrip("/") + "/v1/models"

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(probe_url, headers=headers)
            # 2xx or 4xx -- endpoint is reachable
            if resp.status_code < 500:
                return "ok"
        return "unavailable"
    except Exception:
        logger.debug("LLM health check failed", exc_info=True)
        return "unavailable"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/health")
async def health():
    """Return detailed health status for all configured components.

    Designed for Docker HEALTHCHECK and orchestrator liveness/readiness probes.
    Returns HTTP 503 when any configured component is unavailable so that the
    orchestrator can decide to restart or reroute traffic.
    """
    config = load_config()
    checks: dict[str, str] = {"api": "ok"}

    # --- Redis check ---
    redis_url = config.infrastructure.redis_url
    if redis_url:
        try:
            r = aioredis.from_url(redis_url)
            await r.ping()
            checks["redis"] = "ok"
            await r.aclose()
        except Exception:
            logger.debug("Redis health check failed", exc_info=True)
            checks["redis"] = "unavailable"
    else:
        checks["redis"] = "not_configured"

    # --- DB check ---
    db_url = config.infrastructure.database_url
    if db_url:
        engine = None
        try:
            engine = create_async_engine(db_url, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["db"] = "ok"
        except Exception:
            logger.debug("DB health check failed", exc_info=True)
            checks["db"] = "unavailable"
        finally:
            if engine is not None:
                await engine.dispose()
    else:
        checks["db"] = "not_configured"

    # --- LLM API check ---
    llm_api_key = config.llm.api_key
    if llm_api_key:
        llm_base_url = config.llm.base_url
        checks["llm"] = await _check_llm(llm_base_url, llm_api_key)
    else:
        checks["llm"] = "not_configured"

    # --- Aggregate status ---
    # "degraded" if any *configured* service is unreachable
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
