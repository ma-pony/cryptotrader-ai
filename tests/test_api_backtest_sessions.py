"""History replaces file sessions; snapshot reuse is an explicit new experiment."""

import pytest

from tests.factories.research import research as research
from tests.factories.research import research_payload
from tests.factories.research_offline import research_offline  # noqa: F401


@pytest.mark.asyncio
async def test_empty_history_then_unnamed_run_remains_on_second_visit(research):
    client, service = research
    assert (await client.get("/api/backtest/runs")).json()["items"] == []
    run_id = (await client.post("/api/backtest/runs", json=research_payload())).json()["run_id"]
    await service.task_manager.drain()
    assert (await client.get("/api/backtest/runs")).json()["items"][0]["run_id"] == run_id
    assert (await client.get("/api/backtest/runs?limit=0")).status_code == 422


@pytest.mark.asyncio
async def test_reuse_freezes_old_risk_and_component_weight_then_compare_shows_cost_difference(research):
    client, service = research
    a = (await client.post("/api/backtest/runs", json=research_payload())).json()["run_id"]
    await service.task_manager.drain()
    old = (await service.store.get(a)).config_snapshot
    snapshot = service.repository.snapshot
    from cryptotrader.runtime_config.models import PositionConfig, RiskConfig

    service.repository.snapshot = type(snapshot)(
        9,
        snapshot.document.model_copy(update={"risk": RiskConfig(position=PositionConfig(max_single_pct=0.2))}),
        snapshot.updated_at,
    )
    b = (await client.post("/api/backtest/runs", json=research_payload(snapshot_run_id=a, fee_rate="0.002"))).json()[
        "run_id"
    ]
    await service.task_manager.drain()
    assert (await service.store.get(b)).config_snapshot == old
    response = await client.get(f"/api/backtest/runs/compare?left={a}&right={b}")
    assert response.status_code == 200
    body = response.json()
    assert body["comparable"] is False
    assert body["condition_differences"]["fee_rate"] == {"left": "0.001", "right": "0.002"}
    assert body["left"]["run_id"] == a
    assert body["right"]["run_id"] == b
