"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request


async def verify_api_key(request: Request):
    """Require the runtime-configured API key for protected endpoints."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "snapshot", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    security = runtime.snapshot.document.security
    if not security.enabled:
        return
    key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(key, security.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
