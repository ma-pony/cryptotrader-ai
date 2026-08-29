"""FastAPI application."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import partial
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
    config,
    cycles,
    decisions,
    events,
    health,
    hitl,
    market,
    memory,
    metrics,
    portfolio_books,
    portfolio_v2,
    risk,
    scheduler,
    skills,
    venues,
)
from cryptotrader.tracing import set_trace_id

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown lifecycle."""
    from cryptotrader.log_config import setup_logging

    await _init_runtime(_app)
    _app.state.refresh_runtime_owners = lambda snapshot=None: _refresh_runtime_owners(_app, snapshot=snapshot)
    _app.state.clear_runtime_owners = lambda: _clear_runtime_owners(_app)
    runtime = _app.state.runtime
    setup_logging(runtime.snapshot.document)

    from cryptotrader.otel import setup_otel

    setup_otel(runtime.snapshot.document)
    active = not runtime.snapshot.setup_required

    try:
        if active:
            # Initialize trigger engine if enabled
            await _init_trigger_engine(_app)

            # Initialize trading scheduler if enabled
            await _init_scheduler(_app)

        yield
    finally:
        await _shutdown_runtime_owners(_app, runtime, active=active)
        logger.info("Shutting down")


async def _init_runtime(app_instance: FastAPI) -> None:
    from cryptotrader.runtime import build_runtime

    app_instance.state.runtime = await build_runtime()


async def _shutdown_runtime_owners(app_instance: FastAPI, runtime, *, active: bool) -> None:
    failures: list[BaseException] = []
    if active:
        try:
            from cryptotrader.chat.task_manager import BackgroundTaskManager

            await BackgroundTaskManager.get_instance().drain()
        except BaseException as error:
            failures.append(error)
        try:
            await _shutdown_scheduler(app_instance)
        except BaseException as error:
            failures.append(error)

        trigger_engine = getattr(app_instance.state, "trigger_engine", None)
        if trigger_engine is not None:
            try:
                await trigger_engine.stop()
            except BaseException as error:
                failures.append(error)

    try:
        await runtime.close()
    except BaseException as error:
        failures.append(error)

    control_flow = next((error for error in failures if not isinstance(error, Exception)), None)
    if control_flow is not None:
        raise control_flow
    if failures:
        raise failures[0]


async def _clear_runtime_owners(app_instance) -> None:
    """Stop app-owned execution resources without closing the runtime graph."""
    failures: list[BaseException] = []
    try:
        await _shutdown_scheduler(app_instance)
    except BaseException as error:
        failures.append(error)
    finally:
        app_instance.state.scheduler = None
        app_instance.state.scheduler_task = None
    trigger_engine = getattr(app_instance.state, "trigger_engine", None)
    try:
        if trigger_engine is not None:
            await trigger_engine.stop()
    except BaseException as error:
        failures.append(error)
    finally:
        app_instance.state.trigger_engine = None
        app_instance.state.trigger_store = None
    if failures:
        control_flow = next((error for error in failures if not isinstance(error, Exception)), None)
        if control_flow is not None:
            raise control_flow
        raise RuntimeError("runtime owner cleanup incomplete") from failures[0]


async def _refresh_runtime_owners(app_instance, *, snapshot=None) -> None:
    """Replace app-owned scheduler and trigger resources for the published graph."""
    await _clear_runtime_owners(app_instance)
    runtime = app_instance.state.runtime
    owner_snapshot = snapshot or runtime.snapshot
    if not owner_snapshot.document.system.active:
        return
    await _init_trigger_engine(app_instance, snapshot=owner_snapshot)
    await _init_scheduler(app_instance, snapshot=owner_snapshot)


async def _init_trigger_engine(app_instance: FastAPI, *, snapshot=None) -> None:
    """Initialize PriceTriggerEngine and attach to app.state if triggers enabled."""
    from cryptotrader.db import get_async_session
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.triggers.engine import PriceTriggerEngine
    from cryptotrader.triggers.store import TriggerRuleStore

    runtime = app_instance.state.runtime
    config = (snapshot or runtime.snapshot).document
    if not config.triggers.enabled:
        app_instance.state.trigger_engine = None
        app_instance.state.trigger_store = None
        return

    db_url = getattr(runtime.repository, "database_url", None)
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
        from cryptotrader.decision.models import CycleRequest
        from cryptotrader.pair import Pair

        async with runtime.execution_lease(pair) as cycle:
            await cycle.run(CycleRequest(Pair.parse(pair)))

    engine = PriceTriggerEngine(store, redis_state, _trigger_callback, config.triggers)
    await engine.start()

    app_instance.state.trigger_engine = engine
    app_instance.state.trigger_store = store
    logger.info("PriceTriggerEngine initialized")


async def _init_scheduler(app_instance: FastAPI, *, snapshot=None) -> None:
    """Start the trading Scheduler in a background task if scheduler.enabled.

    The Scheduler runs trading_cycle (interval) + daily_summary (cron) jobs.
    trigger_engine is intentionally NOT injected here — it is owned by
    _init_trigger_engine and started independently. Keeping the two surfaces
    separate means each can be enabled/disabled in config without coupling.
    """
    import asyncio

    from cryptotrader.scheduler import Scheduler

    runtime = app_instance.state.runtime
    config = (snapshot or runtime.snapshot).document
    if not config.scheduler.enabled:
        app_instance.state.scheduler = None
        app_instance.state.scheduler_task = None
        logger.info("Scheduler disabled by config; skipping autostart")
        return

    scheduler = Scheduler(config.scheduler, runtime)
    # Scheduler.start() is blocking (awaits a stop_event), so run as a task.
    task = asyncio.create_task(scheduler.start(), name="trading-scheduler")
    app_instance.state.scheduler = scheduler
    app_instance.state.scheduler_task = task
    # The graph has already been published while the application barrier is
    # held.  Let the task enter ``start`` now, so an immediate admission failure
    # turns this revision into a failed application instead of a dead owner.
    await asyncio.sleep(0)
    if task.done():
        await task
    logger.info(
        "Scheduler autostarted: pairs=%s interval=%dm daily_summary_hour=%d",
        list(config.scheduler.pairs),
        config.scheduler.interval_minutes,
        config.scheduler.daily_summary_hour,
    )


async def _shutdown_scheduler(app_instance: FastAPI) -> None:
    """Signal the Scheduler to stop and await its background task."""
    scheduler = getattr(app_instance.state, "scheduler", None)
    task = getattr(app_instance.state, "scheduler_task", None)
    if scheduler is None or task is None:
        return
    scheduler.stop()
    await task


app = FastAPI(
    title="CryptoTrader AI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# ── CORS (dev only — frontend served at :5173) ──
# SEC-M2: explicit method/header allowlist instead of "*". With allow_credentials
# wildcard methods/headers grant cross-origin requests full access including
# X-API-Key. Restrict to what the frontend actually uses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
        clean = {k: v for k, v in err.items() if k not in {"ctx", "input"}}
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

RATE_LIMIT = 60
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_redis_client: Any = None


def _get_redis_for_rate_limit() -> Any:
    """Lazily build a Redis client for rate limiting; returns None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    runtime = getattr(app.state, "runtime", None)
    url = runtime.snapshot.document.infrastructure.redis_url if runtime is not None else ""
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

    if request.url.path not in ("/health", "/metrics") and not await _check_rate_limit(raw_ip):
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
app.include_router(config.router, dependencies=[Depends(verify_api_key)])
app.include_router(venues.router, dependencies=[Depends(verify_api_key)])
app.include_router(portfolio_books.router, dependencies=[Depends(verify_api_key)])
app.include_router(cycles.router, dependencies=[Depends(verify_api_key)])
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
app.include_router(events.router, dependencies=[Depends(verify_api_key)])
app.include_router(skills.router, dependencies=[Depends(verify_api_key)])
