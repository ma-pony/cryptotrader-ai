"""Read-only canonical component evaluations endpoint."""

from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
async def test_component_evaluations_route_exists_and_is_read_only():
    from api.main import app

    assert any(route.path == "/api/components/{component_id}/evaluations" for route in app.routes), (
        "component evaluation GET is missing"
    )


@pytest.mark.asyncio
async def test_get_filters_and_restart_do_not_touch_predictions_or_market(tmp_path, monkeypatch):
    from fastapi import Depends, FastAPI

    from api.dependencies import verify_api_key
    from api.routes.components import router
    from cryptotrader.journal.store import _component_signal_payload, _run_payload
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
    from tests.factories.runtime_config import active_document
    from tests.test_signal_evaluation import BASE, candle, record, setup

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    f = await setup(tmp_path, [record(), record("other", interval="4h")])
    f.market.rows["1m"] = (candle(BASE + timedelta(hours=2) - timedelta(minutes=1), "110"),)
    await f.service.evaluate_due(BASE + timedelta(hours=2))
    original = await f.journal.get("one")
    before = (_run_payload(original.run), _component_signal_payload(original.component_signals[0]))
    call_count = len(f.market.calls)
    runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(99, active_document(), BASE),
        evaluation_store=type(f.store)(f.store.database_url),
        application_in_progress=True,
    )
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        result = await client.get(
            "/api/components/custom/evaluations?mode=analysis&config_revision=7&interval=2h&status=evaluated"
        )
        assert result.status_code == 200
        payload = result.json()
        assert payload["total"] == 1
        assert payload["items"][0]["actual_price"] == "110"
        assert payload["summary"]["groups"][0]["hit_rate"] == 1
        assert (await client.get("/api/components/custom/evaluations?mode=bogus")).status_code == 422
        assert (await client.post("/api/components/custom/evaluations")).status_code == 405
    saved = await f.journal.get("one")
    assert (_run_payload(saved.run), _component_signal_payload(saved.component_signals[0])) == before
    assert len(f.market.calls) == call_count


@pytest.mark.asyncio
async def test_real_app_read_during_apply_uses_no_runtime_resources(api_harness, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from tests.test_signal_evaluation import BASE, candle, record, setup

    f = await setup(tmp_path, [record()])
    due = BASE + timedelta(hours=2)
    f.market.rows["1m"] = (candle(due - timedelta(minutes=1), "110"),)
    await f.service.evaluate_due(due)
    api_harness.runtime.evaluation_store = type(f.store)(f.store.database_url)
    api_harness.runtime.application_in_progress = True
    reject = AsyncMock(side_effect=AssertionError("GET cannot use market/model/account capabilities"))
    monkeypatch.setattr("cryptotrader.data.market.MarketCollector.read_candles", reject)
    monkeypatch.setattr("cryptotrader.signals.components.kronos.KronosComponent.evaluate", reject)
    for session in api_harness.runtime.sessions.values():
        monkeypatch.setattr(session, "fetch_portfolio", reject)
    response = await api_harness.client.get("/api/components/custom/evaluations")
    assert response.status_code == 200
    assert response.json()["items"][0]["hit"] is True
    assert response.json()["summary"]["groups"][0]["matured_directional"] == 1
    reject.assert_not_awaited()
    assert all(adapter.connect_calls == [] for adapter in api_harness.adapters.values())
