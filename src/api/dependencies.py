"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from cryptotrader.runtime_config.repository import API_ACCESS_CREDENTIAL_REF, CredentialNotConfigured

_SETUP_COMMISSIONING_ROUTES = frozenset(
    {
        ("GET", "/api/config"),
        ("PUT", "/api/config"),
        ("PUT", "/api/config/credentials/llm-gateway"),
        ("PUT", "/api/config/credentials/api-access"),
    }
)


def _is_setup_commissioning_request(request: Request) -> bool:
    if (request.method, request.url.path) in _SETUP_COMMISSIONING_ROUTES:
        return True
    if request.url.path == "/api/venue-connections" and request.method == "POST":
        return True
    segments = request.url.path.split("/")
    return (len(segments) == 4 and segments[:3] == ["", "api", "venue-connections"] and request.method == "PUT") or (
        len(segments) == 5
        and segments[:3] == ["", "api", "venue-connections"]
        and segments[4] in {"credentials", "test"}
        and request.method == ("PUT" if segments[4] == "credentials" else "POST")
    )


async def verify_api_key(request: Request):
    """Require the runtime-configured API key for protected endpoints."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "snapshot", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    security = runtime.snapshot.document.security
    if runtime.snapshot.setup_required is True and not _is_setup_commissioning_request(request):
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
