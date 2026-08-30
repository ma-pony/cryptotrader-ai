"""Real admission, persistence and the documented typed-plugin example."""

from importlib import metadata

import httpx
import pytest


async def test_preview_uses_real_catalog_persistence_and_refuses_activation(tmp_path, monkeypatch):
    from tests.manual.configuration_preview import create_preview

    entry = metadata.EntryPoint(
        name="configuration_example",
        value="examples.configuration_plugin.configuration_example:create_component",
        group="cryptotrader.signal_components",
    )
    original = metadata.entry_points
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: (*original(group=group), entry) if group == entry.group else original(group=group),
    )
    app = await create_preview(tmp_path, tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        catalog = (await client.get("/api/config/catalog")).json()
        custom = next(item for item in catalog["components"] if item["id"] == "configuration_example")
        assert {item["key"]: item["kind"] for item in custom["fields"]} == {
            "window": "integer",
            "timeframe": "select",
            "diagnostics.enabled": "boolean",
        }
        runtime = app.state.runtime
        document = runtime.snapshot.document.model_dump(mode="json")
        document["risk"]["loss"]["max_drawdown_pct"] = 0.075
        document["signals"]["components"].append(
            {
                "component_id": "configuration_example",
                "enabled": False,
                "weight": 0,
                "parameters": {"window": 24, "timeframe": "15m", "diagnostics": {"enabled": True}},
            }
        )
        saved = await client.put("/api/config", json={"expected_revision": 1, "document": document})
        assert saved.status_code == 200
        assert saved.json()["document"]["risk"]["loss"]["max_drawdown_pct"] == 0.075
        assert saved.json()["document"]["system"]["active"] is False
        assert (await client.put("/api/config", json={"expected_revision": 1, "document": document})).status_code == 409
        invalid = dict(document)
        invalid["signals"] = dict(document["signals"])
        invalid["signals"]["components"] = [
            *document["signals"]["components"][:-1],
            {
                **document["signals"]["components"][-1],
                "parameters": {"window": 0},
            },
        ]
        assert (await client.put("/api/config", json={"expected_revision": 2, "document": invalid})).status_code == 422
        document["system"]["active"] = True
        assert (await client.put("/api/config", json={"expected_revision": 2, "document": document})).status_code == 403
        assert (await client.post("/api/backtest/run", json={})).status_code == 403
        assert (await client.get("/api/config")).json()["revision"] == 2


def test_documented_plugin_uses_typed_values_without_running_a_provider():
    from cryptotrader.runtime_config.models import SignalComponentConfig
    from examples.configuration_plugin.configuration_example import create_component
    from tests.factories.runtime_config import runtime_document, signal_config

    definition = create_component.configuration
    with pytest.raises(ValueError):
        definition.parameter_model.model_validate({"unknown": True})
    document = runtime_document(
        signals=signal_config(
            components=(
                SignalComponentConfig(
                    component_id="configuration_example",
                    enabled=True,
                    weight=1,
                    parameters={"window": 24, "timeframe": "15m", "diagnostics": {"enabled": True}},
                ),
            )
        )
    )
    component = create_component(document, None)
    assert [(item.timeframe, item.limit) for item in component.requirements().candles] == [("15m", 24)]
    assert component.parameters.diagnostics.enabled is True


async def test_preview_retains_real_auth_and_failed_publication_contract(tmp_path):
    from tests.manual.configuration_preview import create_preview

    app = await create_preview(tmp_path, tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/__fixture__/protect")
        assert (await client.get("/api/config")).status_code == 401
        assert (await client.get("/api/config", headers={"X-API-Key": "wrong-fixture-key"})).status_code == 401
        client.headers["X-API-Key"] = "fixture-access-only"
        before = (await client.get("/api/config")).json()
        await client.post("/__fixture__/fail-next-publication")
        document = app.state.runtime.snapshot.document.model_dump(mode="json")
        document["risk"]["loss"]["max_drawdown_pct"] = 0.08
        response = await client.put("/api/config", json={"expected_revision": before["revision"], "document": document})
        assert response.status_code == 503
        after = (await client.get("/api/config")).json()
        assert after["revision"] == before["revision"] + 1
        assert after["apply_status"] == "failed"
        assert after["document"]["risk"]["loss"]["max_drawdown_pct"] == 0.08
        assert after["document"]["system"]["active"] is False
        for path in ("/api/orders", "/api/backtest/run", "/api/chat"):
            assert (await client.post(path, json={})).status_code == 403


async def test_preview_operational_reads_are_seeded_and_csp_blocks_external_feeds(tmp_path):
    from tests.manual.configuration_preview import create_preview

    app = await create_preview(tmp_path, tmp_path, active_ui=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/config")
        assert response.json()["document"]["system"]["active"] is True
        assert response.json()["document"]["execution"]["live_order_execution_enabled"] is False
        assert "connect-src 'self'" in response.headers["content-security-policy"]
        assert (await client.get("/api/scheduler/status")).json()["enabled"] is False
        assert (await client.get("/api/scheduler/triggers")).json()["size"] == 20
        saved = (await client.get("/api/backtest/sessions/fixture-saved-parameters")).json()
        assert saved["params"]["initial_capital"] == 2500
        assert (await client.post("/api/scheduler/rules", json={})).status_code == 403
