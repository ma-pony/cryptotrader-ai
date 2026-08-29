"""Legacy partial signal-profile API is unavailable after the Runtime cutover."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "PUT"])
async def test_legacy_signal_profile_route_is_not_mounted(method: str) -> None:
    from api.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(method, "/api/signal-profile", json={})

    assert response.status_code == 404
