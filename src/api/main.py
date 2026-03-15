"""FastAPI application."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import analyze, health, journal, metrics, portfolio, scheduler
from cryptotrader.tracing import set_trace_id

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

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

    from cryptotrader.otel import setup_otel

    setup_otel()

    yield
    logger.info("Shutting down")


# ── Docs endpoint control ──
# Read DOCS_ENABLED env var (default: "false").  When false/absent, Swagger UI
# and ReDoc are disabled to prevent API schema information leakage in production
# (Requirement 7.6).
_docs_enabled = os.environ.get("DOCS_ENABLED", "false").lower() == "true"
_docs_url = "/docs" if _docs_enabled else None
_redoc_url = "/redoc" if _docs_enabled else None

app = FastAPI(
    title="CryptoTrader AI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)


# ── Global exception handlers ──


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — never leak stack traces."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors (422).

    Logs a sanitized request summary — method, path, and error count — without
    echoing the raw request body or any sensitive field values (Requirement 7.4).
    """
    error_count = len(exc.errors())
    logger.warning(
        "Request validation failed: %s %s — %d error(s)",
        request.method,
        request.url.path,
        error_count,
    )
    # Return structured error details from Pydantic (field paths + messages only,
    # never raw body values).
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# ── Client IP masking ──


def _mask_client_ip(ip: str) -> str:
    """Mask the last octet of an IPv4 address for privacy compliance.

    IPv6 addresses and non-standard values (``unknown``, empty string) are
    returned unchanged.

    Examples::

        _mask_client_ip("192.168.1.100") == "192.168.1.xxx"
        _mask_client_ip("::1")           == "::1"
        _mask_client_ip("unknown")       == "unknown"
    """
    if not ip:
        return ip
    parts = ip.split(".")
    if len(parts) == 4:  # IPv4
        return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
    return ip


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
    """Inject trace ID, enforce rate limit, and emit structured request log.

    Emits a structlog event with the following standard fields (Requirement 9.7):
    - ``method`` — HTTP method (GET, POST, …)
    - ``path`` — URL path
    - ``status_code`` — HTTP response status code
    - ``response_time_ms`` — end-to-end response time in milliseconds
    - ``client_ip`` — remote IP with last octet masked for IPv4
    """
    raw_ip = request.client.host if request.client else "unknown"
    client_ip = _mask_client_ip(raw_ip)

    # Rate limit (skip health/metrics for probes)
    if request.url.path not in ("/health", "/metrics") and not _check_rate_limit(raw_ip):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    trace_id = set_trace_id(request.headers.get("X-Trace-ID"))
    t0 = time.monotonic()
    response = await call_next(request)
    response_time_ms = int((time.monotonic() - t0) * 1000)

    response.headers["X-Trace-ID"] = trace_id

    _slog.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        response_time_ms=response_time_ms,
        client_ip=client_ip,
        trace_id=trace_id,
    )

    return response


# -- Routes --
# /health, /metrics and /scheduler/status are public (load balancer probes / Dashboard polling)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(scheduler.router)

# Protected routes require API key
app.include_router(analyze.router, dependencies=[Depends(verify_api_key)])
app.include_router(journal.router, dependencies=[Depends(verify_api_key)])
app.include_router(portfolio.router, dependencies=[Depends(verify_api_key)])
