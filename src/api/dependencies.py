"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from cryptotrader.runtime_config.repository import API_ACCESS_CREDENTIAL_REF, CredentialNotConfigured


def _commissioning_route_kind(request: Request) -> str | None:
    """Classify the single setup surface once for both setup and application admission."""
    method, path = request.method, request.url.path
    if (method, path) in {("GET", "/api/config"), ("GET", "/api/config/catalog")}:
        return "read"
    if (method, path) in {
        ("PUT", "/api/config"),
        ("PUT", "/api/config/credentials/llm-gateway"),
        ("PUT", "/api/config/credentials/api-access"),
        ("PUT", "/api/config/credentials/news-provider"),
        ("POST", "/api/venue-connections"),
    }:
        return "mutation"
    segments = request.url.path.split("/")
    if len(segments) == 4 and segments[:3] == ["", "api", "venue-connections"] and method == "PUT":
        return "mutation"
    if len(segments) == 5 and segments[:3] == ["", "api", "venue-connections"]:
        if segments[4] == "credentials" and method == "PUT":
            return "mutation"
        if segments[4] == "test" and method == "POST":
            return "read"
    return None


async def verify_api_key(request: Request):
    """Require the runtime-configured API key for protected endpoints."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "snapshot", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    commissioning_route = _commissioning_route_kind(request)
    if getattr(runtime, "application_in_progress", False) is True and commissioning_route != "mutation":
        raise HTTPException(status_code=503, detail="Runtime configuration is being applied")
    security = runtime.snapshot.document.security
    if runtime.snapshot.setup_required is True and commissioning_route is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    if not security.enabled:
        return
    key = request.headers.get("X-API-Key")
    if not isinstance(key, str) or not key.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        expected = (await runtime.repository.reveal_token(API_ACCESS_CREDENTIAL_REF)).token
    except CredentialNotConfigured:
        raise HTTPException(status_code=503, detail="API access credential is not configured") from None
    if not isinstance(expected, str) or not expected.strip() or not secrets.compare_digest(key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
