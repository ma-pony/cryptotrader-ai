"""FastAPI application."""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.routes import analyze, health, journal, portfolio
from cryptotrader.tracing import set_trace_id

logger = logging.getLogger(__name__)

# ── API Key auth ──

API_KEY = os.environ.get("API_KEY", "")


async def verify_api_key(request: Request):
    """Require X-API-Key header on protected endpoints when API_KEY is set."""
    if not API_KEY:
        return  # No key configured — skip auth (dev mode)
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Lifespan ──


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown lifecycle."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    yield
    logger.info("Shutting down")


app = FastAPI(title="CryptoTrader AI", version="0.1.0", lifespan=lifespan)


# ── Global exception handler ──


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — never leak stack traces."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Rate limiting ──

RATE_LIMIT = int(os.environ.get("API_RATE_LIMIT", "60"))  # requests per minute
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Sliding window rate limiter. Returns True if allowed."""
    now = time.time()
    window = _rate_buckets[client_ip]
    # Prune entries older than 60s
    cutoff = now - 60
    _rate_buckets[client_ip] = [t for t in window if t > cutoff]
    if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
        return False
    _rate_buckets[client_ip].append(now)
    return True


# ── Middleware ──


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Inject trace ID and enforce rate limit."""
    # Rate limit (skip health/metrics for probes)
    if request.url.path not in ("/health", "/metrics"):
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    trace_id = set_trace_id(request.headers.get("X-Trace-ID"))
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


# ── Routes ──
# /health is public (for load balancers / k8s probes)
app.include_router(health.router)

# Protected routes require API key
app.include_router(analyze.router, dependencies=[Depends(verify_api_key)])
app.include_router(journal.router, dependencies=[Depends(verify_api_key)])
app.include_router(portfolio.router, dependencies=[Depends(verify_api_key)])
