"""网页动态策略配置后端契约。"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from cryptotrader.signals.models import ComponentSignal, DataRequirements
from tests.factories.signal_fusion import profile


class FakeComponent:
    description = "test component"

    def __init__(self, component_id, display_name):
        self.id = component_id
        self.display_name = display_name

    def requirements(self):
        return DataRequirements()

    async def evaluate(self, context):
        return ComponentSignal(self.id, "neutral", 0.0, "test")


def _payload(kronos=0.6, llm=0.4, **overrides):
    payload = {
        "components": [
            {"component_id": "kronos", "enabled": True, "weight": kronos},
            {"component_id": "llm_committee", "enabled": True, "weight": llm},
        ],
        "neutral_threshold": 0.2,
        "max_target_ratio": 1.0,
        "atr_stop_multiplier": 2.0,
        "reward_ratio": 2.0,
        "hitl_required": False,
    }
    return payload | overrides


@pytest_asyncio.fixture
async def api_client(tmp_path):
    from api.main import app
    from cryptotrader.profiles.repository import SignalProfileRepository
    from cryptotrader.signals.registry import SignalComponentRegistry

    repository = SignalProfileRepository(f"sqlite+aiosqlite:///{tmp_path / 'profile.db'}")
    await repository.get_or_create(profile())
    registry = SignalComponentRegistry(
        (
            FakeComponent("kronos", "Kronos"),
            FakeComponent("llm_committee", "LLM 四智能体委员会"),
        )
    )
    old_repository = getattr(app.state, "signal_profile_repository", None)
    old_registry = getattr(app.state, "signal_registry", None)
    app.state.signal_profile_repository = repository
    app.state.signal_registry = registry
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.state.signal_profile_repository = old_repository
    app.state.signal_registry = old_registry


@pytest.mark.asyncio
async def test_get_profile_returns_registry_metadata(api_client):
    response = await api_client.get("/api/signal-profile")

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 1
    assert datetime.fromisoformat(body["updated_at"]).tzinfo is not None
    assert {item["component_id"] for item in body["installed_components"]} == {"kronos", "llm_committee"}


@pytest.mark.asyncio
async def test_put_profile_rejects_weights_not_equal_to_one(api_client):
    response = await api_client.put("/api/signal-profile", json=_payload(kronos=0.6, llm=0.3))

    assert response.status_code == 422
    assert "1.0" in response.json()["detail"]


@pytest.mark.asyncio
async def test_put_profile_replaces_all_fields_and_increments_revision(api_client):
    response = await api_client.put(
        "/api/signal-profile",
        json=_payload(
            kronos=0.75,
            llm=0.25,
            neutral_threshold=0.3,
            max_target_ratio=0.8,
            hitl_required=True,
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 2
    assert datetime.fromisoformat(body["updated_at"]).tzinfo is not None
    assert body["neutral_threshold"] == 0.3
    assert body["max_target_ratio"] == 0.8
    assert body["hitl_required"] is True


@pytest.mark.asyncio
async def test_put_profile_rejects_uninstalled_component(api_client):
    payload = _payload()
    payload["components"].append({"component_id": "missing", "enabled": False, "weight": 0.0})

    response = await api_client.put("/api/signal-profile", json=payload)

    assert response.status_code == 422
    assert "uninstalled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_put_profile_rejects_read_only_response_fields(api_client):
    response = await api_client.put(
        "/api/signal-profile",
        json=_payload() | {"revision": 99, "updated_at": "2026-08-28T00:00:00Z"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_app_exposes_seeded_profile_repository_without_database(monkeypatch):
    from fastapi import FastAPI

    from api.main import _init_signal_profile
    from cryptotrader.config import AppConfig

    built = []

    def fake_builder(
        config,
        mode,
        event_sink=None,
        *,
        profile_repository=None,
        approval_store=None,
    ):
        cycle = type(
            "Cycle",
            (),
            {
                "mode": mode,
                "profiles": profile_repository,
                "approvals": approval_store,
                "registry": object(),
            },
        )()
        built.append(cycle)
        return cycle

    monkeypatch.setattr("cryptotrader.config.load_config", lambda: AppConfig())
    monkeypatch.setattr("cryptotrader.bootstrap.build_trading_cycle", fake_builder)
    isolated_app = FastAPI()

    await _init_signal_profile(isolated_app)

    second = isolated_app.state.trading_cycle_builder("paper")
    assert isolated_app.state.signal_profile_repository is built[0].profiles
    assert second.profiles is built[0].profiles
    assert second.approvals is built[0].approvals
