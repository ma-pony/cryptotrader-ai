"""FastAPI application."""

from __future__ import annotations

# Load .env into os.environ BEFORE any project import. api.dependencies
# validates AUTH_MODE / API_KEY at module-import time and SystemExits if
# misconfigured, so dotenv has to land first.
from dotenv import load_dotenv

load_dotenv()

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import verify_api_key
from api.routes import (
    backtest,
    chat,
    chat_control,
    decisions,
    health,
    hitl,
    market,
    memory,
    metrics,
    portfolio_v2,
    risk,
    scheduler,
)
from cryptotrader.tracing import set_trace_id

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown lifecycle."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    from cryptotrader.otel import setup_otel

    setup_otel()

    # Initialize trigger engine if enabled
    await _init_trigger_engine(_app)

    # Initialize HITL Telegram bot if enabled
    await _init_hitl_telegram(_app)

    # Initialize trading scheduler if enabled
    await _init_scheduler(_app)

    yield

    # Shutdown scheduler first (it owns trading-cycle + daily-summary jobs)
    await _shutdown_scheduler(_app)

    # Shutdown HITL Telegram bot
    telegram_bot = getattr(_app.state, "telegram_bot", None)
    if telegram_bot is not None:
        await telegram_bot.stop()

    # Shutdown trigger engine
    trigger_engine = getattr(_app.state, "trigger_engine", None)
    if trigger_engine is not None:
        await trigger_engine.stop()

    from cryptotrader.nodes.execution import close_live_exchanges

    await close_live_exchanges()
    logger.info("Shutting down")


async def _init_trigger_engine(app_instance: FastAPI) -> None:
    """Initialize PriceTriggerEngine and attach to app.state if triggers enabled."""
    from functools import partial

    from cryptotrader.config import load_config
    from cryptotrader.db import get_async_session
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.triggers.engine import PriceTriggerEngine
    from cryptotrader.triggers.store import TriggerRuleStore

    config = load_config()
    if not config.triggers.enabled:
        app_instance.state.trigger_engine = None
        app_instance.state.trigger_store = None
        return

    db_url = config.infrastructure.database_url
    if not db_url:
        logger.warning("Triggers enabled but no database_url configured; skipping")
        app_instance.state.trigger_engine = None
        app_instance.state.trigger_store = None
        return

    # Ensure trigger tables exist
    await TriggerRuleStore.ensure_tables(db_url)

    session_factory = partial(get_async_session, db_url)
    store = TriggerRuleStore(session_factory)
    redis_state = RedisStateManager(config.infrastructure.redis_url)

    async def _trigger_callback(pair: str, meta: dict) -> None:
        logger.info("Trigger fired for %s: %s", pair, meta)

    engine = PriceTriggerEngine(store, redis_state, _trigger_callback, config.triggers)
    await engine.start()

    app_instance.state.trigger_engine = engine
    app_instance.state.trigger_store = store
    logger.info("PriceTriggerEngine initialized")


async def _init_scheduler(app_instance: FastAPI) -> None:
    """Start the trading Scheduler in a background task if scheduler.enabled.

    The Scheduler runs trading_cycle (interval) + daily_summary (cron) jobs.
    trigger_engine is intentionally NOT injected here — it is owned by
    _init_trigger_engine and started independently. Keeping the two surfaces
    separate means each can be enabled/disabled in config without coupling.
    """
    import asyncio

    from cryptotrader.config import load_config
    from cryptotrader.scheduler import Scheduler

    config = load_config()
    if not config.scheduler.enabled:
        app_instance.state.scheduler = None
        app_instance.state.scheduler_task = None
        logger.info("Scheduler disabled by config; skipping autostart")
        return

    scheduler = Scheduler(
        pairs=config.scheduler.pairs,
        interval_minutes=config.scheduler.interval_minutes,
        daily_summary_hour=config.scheduler.daily_summary_hour,
    )
    # Scheduler.start() is blocking (awaits a stop_event), so run as a task.
    task = asyncio.create_task(scheduler.start(), name="trading-scheduler")
    app_instance.state.scheduler = scheduler
    app_instance.state.scheduler_task = task
    logger.info(
        "Scheduler autostarted: pairs=%s interval=%dm daily_summary_hour=%d",
        [p.canonical() for p in config.scheduler.pairs],
        config.scheduler.interval_minutes,
        config.scheduler.daily_summary_hour,
    )


async def _shutdown_scheduler(app_instance: FastAPI) -> None:
    """Signal the Scheduler to stop and await its background task."""
    import asyncio

    scheduler = getattr(app_instance.state, "scheduler", None)
    task = getattr(app_instance.state, "scheduler_task", None)
    if scheduler is None or task is None:
        return
    import contextlib

    scheduler.stop()
    try:
        await asyncio.wait_for(task, timeout=10)
    except TimeoutError:
        logger.warning("Scheduler did not exit within 10s; cancelling")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _init_hitl_telegram(app_instance: FastAPI) -> None:
    """Start Telegram HITL bot + expire stale approvals on startup."""
    from cryptotrader.config import load_config

    config = load_config()
    app_instance.state.telegram_bot = None

    db_url = config.infrastructure.database_url
    if db_url:
        try:
            from cryptotrader.hitl.store import ApprovalStore

            expired = await ApprovalStore.expire_stale(db_url)
            if expired:
                logger.info("Expired %d stale HITL approvals on startup", expired)
        except Exception:
            logger.info("Failed to expire stale HITL approvals", exc_info=True)

    if not config.hitl.telegram.enabled or not config.hitl.telegram.bot_token:
        return

    if not db_url:
        logger.warning("HITL Telegram enabled but no database_url configured; skipping")
        return

    try:
        from cryptotrader.hitl.telegram import TelegramApprovalBot

        bot = TelegramApprovalBot(
            bot_token=config.hitl.telegram.bot_token,
            chat_id=config.hitl.telegram.chat_id,
            db_url=db_url,
        )
        await bot.start()
        app_instance.state.telegram_bot = bot
    except Exception:
        logger.warning("Failed to start HITL Telegram bot", exc_info=True)


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

# ── CORS (dev only — frontend served at :5173) ──
# SEC-M2: explicit method/header allowlist instead of "*". With allow_credentials
# wildcard methods/headers grant cross-origin requests full access including
# X-API-Key. Restrict to what the frontend actually uses.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Trace-ID", "Last-Event-ID"],
    expose_headers=["X-Trace-ID"],
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
    # never raw body values). Strip non-JSON-serializable ``ctx`` payloads which
    # Pydantic v2 includes for value_error types.
    sanitized = []
    for err in exc.errors():
        clean = {k: v for k, v in err.items() if k != "ctx"}
        clean["loc"] = list(clean.get("loc", ()))
        sanitized.append(clean)
    return JSONResponse(
        status_code=422,
        content={"detail": sanitized},
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
# SEC-M4: Prefer Redis-backed counter (multi-process safe). Falls back to the
# in-process dict when Redis is unavailable so local dev / unit tests work.
# Fixed-window approximation: per-IP counter with 60s TTL. Good enough for our
# 60 req/min ceiling — sliding-window precision is not needed for abuse defence.

RATE_LIMIT = int(os.environ.get("API_RATE_LIMIT", "60"))  # requests per minute
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_redis_client: Any = None


def _get_redis_for_rate_limit() -> Any:
    """Lazily build a Redis client for rate limiting; returns None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = os.environ.get("CRYPTOTRADER_INFRASTRUCTURE__REDIS_URL", "")
    if not url or url == "DISABLED":
        return None
    try:
        import redis.asyncio as redis  # type: ignore[import-not-found]

        _redis_client = redis.from_url(url, decode_responses=True)
        return _redis_client
    except Exception:
        logger.info("Rate-limit Redis client unavailable", exc_info=True)
        return None


def _check_rate_limit_inproc(client_ip: str) -> bool:
    """In-process fallback (single-worker only)."""
    now = time.time()
    window = _rate_buckets[client_ip]
    cutoff = now - 60
    _rate_buckets[client_ip] = [t for t in window if t > cutoff]
    if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
        return False
    _rate_buckets[client_ip].append(now)
    return True


async def _check_rate_limit(client_ip: str) -> bool:
    """Returns True if request is allowed. Redis-backed when configured."""
    r = _get_redis_for_rate_limit()
    if r is None:
        return _check_rate_limit_inproc(client_ip)
    bucket = int(time.time() // 60)
    key = f"ratelimit:{client_ip}:{bucket}"
    try:
        count = await r.incr(key)
        if count == 1:
            await r.expire(key, 60)
        return count <= RATE_LIMIT
    except Exception:
        # Fail-open with in-process fallback so a Redis blip doesn't 429-storm.
        logger.warning("Redis rate-limit failed, falling back to in-process", exc_info=True)
        return _check_rate_limit_inproc(client_ip)


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

    # Rate limit (skip health/metrics for probes; also skip when AUTH_MODE
    # is disabled — i.e. trusted local dev: the dashboard fires 10+ memory
    # endpoints on page load + 30s polling that quickly drains the 60 rpm
    # bucket and shows persistent "加载中…" spinners).
    auth_mode = os.environ.get("AUTH_MODE", "enabled").lower()
    skip_rl = auth_mode == "disabled" or request.url.path in ("/health", "/metrics")
    if not skip_rl and not await _check_rate_limit(raw_ip):
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
app.include_router(portfolio_v2.router, dependencies=[Depends(verify_api_key)])
app.include_router(decisions.router, dependencies=[Depends(verify_api_key)])
app.include_router(backtest.router, dependencies=[Depends(verify_api_key)])
app.include_router(risk.router, dependencies=[Depends(verify_api_key)])
app.include_router(scheduler.api_router, dependencies=[Depends(verify_api_key)])
app.include_router(metrics.api_router, dependencies=[Depends(verify_api_key)])
app.include_router(chat.router, dependencies=[Depends(verify_api_key)])
app.include_router(chat_control.router, dependencies=[Depends(verify_api_key)])
app.include_router(hitl.router, dependencies=[Depends(verify_api_key)])
app.include_router(market.router, dependencies=[Depends(verify_api_key)])
app.include_router(memory.router, prefix="/api/memory", dependencies=[Depends(verify_api_key)])
