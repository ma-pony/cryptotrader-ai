from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.factories.signal_fusion import cycle_record


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _config() -> MagicMock:
    config = MagicMock()
    config.infrastructure.database_url = None
    return config


def _record():
    return cycle_record(
        cycle_id="cycle-detail",
        status="completed",
        profile_revision=3,
        component_signals=(
            {
                "component_id": "kronos",
                "direction": "long",
                "confidence": 0.8,
                "reasoning": "forecast",
                "details": {"raw_signal": 0.03},
            },
            {
                "component_id": "llm_committee",
                "direction": "short",
                "confidence": 0.3,
                "reasoning": "committee summary",
                "details": {
                    "analyses": {"technical": {"direction": "long", "confidence": 0.7}},
                    "debate_turns": [{"round": 1, "from": "technical", "to": "macro"}],
                },
            },
        ),
        fused_signal={
            "score": 0.36,
            "direction": "long",
            "reasoning": "weighted signal score=0.36",
            "contributions": [
                {"component_id": "kronos", "weighted_value": 0.48},
                {"component_id": "llm_committee", "weighted_value": -0.12},
            ],
        },
        target_position={"side": "long", "size_ratio": 0.2},
        trade_plan={"target": {"side": "long", "size_ratio": 0.2}, "stop_loss": 90, "take_profit": 120},
        risk_result={"passed": True, "rejected_by": "", "reason": ""},
        execution_result={"succeeded": True, "orders": []},
    )


def test_detail_exposes_components_contributions_and_internal_debate(client: TestClient) -> None:
    store = MagicMock()
    store.get = AsyncMock(return_value=_record())
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions/cycle-detail")

    assert response.status_code == 200
    body = response.json()
    assert body["cycle_id"] == "cycle-detail"
    assert body["components"][0]["component_id"] == "kronos"
    assert body["fusion"]["score"] == pytest.approx(0.36)
    assert body["fusion"]["contributions"][0]["weighted_value"] == pytest.approx(0.48)
    assert body["target_position"] == {"side": "long", "size_ratio": 0.2}
    committee = body["components"][1]
    assert committee["details"]["analyses"]["technical"]["direction"] == "long"
    assert committee["details"]["debate_turns"][0]["round"] == 1


def test_detail_returns_404_for_unknown_cycle(client: TestClient) -> None:
    store = MagicMock()
    store.get = AsyncMock(return_value=None)
    with (
        patch("cryptotrader.config.load_config", return_value=_config()),
        patch("cryptotrader.journal.store.CycleJournalStore", return_value=store),
    ):
        response = client.get("/api/decisions/missing")

    assert response.status_code == 404


def test_detail_exposes_the_complete_frozen_profile_snapshot() -> None:
    from api.routes.decisions import _detail

    record = _record()
    frozen_profile = {
        "revision": 3,
        "components": [
            {"component_id": "kronos", "enabled": True, "weight": 0.6},
            {"component_id": "llm_committee", "enabled": True, "weight": 0.4},
        ],
        "neutral_threshold": 0.2,
        "max_target_ratio": 1.0,
        "atr_stop_multiplier": 2.0,
        "reward_ratio": 2.0,
        "hitl_required": False,
        "updated_at": "2026-08-28T01:02:03+00:00",
    }
    object.__setattr__(record, "profile_snapshot", frozen_profile)

    assert _detail(record).model_dump()["profile"] == frozen_profile
