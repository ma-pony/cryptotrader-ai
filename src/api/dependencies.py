"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# AUTH_MODE controls how the API key check behaves:
#   - "enabled"  (default): API_KEY MUST be set or import fails — protects against
#                accidentally deploying without auth.
#   - "disabled": Auth is bypassed; every request logs a WARNING so the operator
#                can never silently lose the protection.
AUTH_MODE = os.environ.get("AUTH_MODE", "enabled").lower()
API_KEY = os.environ.get("API_KEY", "")

if AUTH_MODE not in ("enabled", "disabled"):
    raise SystemExit(
        f"FATAL: AUTH_MODE must be 'enabled' or 'disabled', got {AUTH_MODE!r}.",
    )

if AUTH_MODE == "enabled" and not API_KEY:
    raise SystemExit(
        "FATAL: AUTH_MODE=enabled but API_KEY is empty.\n"
        "Set API_KEY in the environment, or explicitly opt out with AUTH_MODE=disabled "
        "(disabled mode logs a warning on every request).",
    )

if AUTH_MODE == "disabled":
    logger.warning(
        "SECURITY: AUTH_MODE=disabled — all protected endpoints are publicly accessible. "
        "Use only in local development.",
    )


async def verify_api_key(request: Request):
    """Require X-API-Key header on protected endpoints when auth is enabled."""
    if AUTH_MODE == "disabled":
        logger.warning(
            "AUTH bypassed for %s %s — set AUTH_MODE=enabled + API_KEY for production.",
            request.method,
            request.url.path,
        )
        return
    key = request.headers.get("X-API-Key", "")
    # secrets.compare_digest avoids timing attacks against API_KEY.
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
