"""FastAPI application."""

from fastapi import FastAPI
from api.routes import analyze, journal, health

app = FastAPI(title="CryptoTrader AI", version="0.1.0")
app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(journal.router)
