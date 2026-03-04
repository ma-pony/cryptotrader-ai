"""FastAPI application."""

from fastapi import FastAPI, Request
from api.routes import analyze, journal, health, portfolio
from cryptotrader.tracing import set_trace_id

app = FastAPI(title="CryptoTrader AI", version="0.1.0")


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Inject trace ID for each request."""
    trace_id = set_trace_id(request.headers.get("X-Trace-ID"))
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(journal.router)
app.include_router(portfolio.router)
