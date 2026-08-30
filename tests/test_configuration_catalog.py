"""Installed plugin configuration catalog and parameter-validation contracts."""

from __future__ import annotations

from importlib import metadata

import pytest
from pydantic import BaseModel, Field

from cryptotrader.runtime_config.models import SignalComponentConfig
from tests.factories.runtime_config import runtime_document
from tests.test_runtime_config_api import active_payload

pytest_plugins = ("tests.test_runtime_config_api",)


def _entry_point(name: str, value: str, group: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(name=name, value=value, group=group)


async def test_catalog_exposes_paper_funding_field(api_harness):
    response = await api_harness.client.get("/api/config/catalog")

    assert response.status_code == 200
    paper = next(item for item in response.json()["venues"] if item["id"] == "paper")
    assert paper["environments"] == ["paper"]
    assert paper["credential_fields"] == []
    initial_equity = next(field for field in paper["fields"] if field["key"] == "initial_equity")
    assert initial_equity["kind"] == "number"
    assert initial_equity["unit"] == "USDT"
    assert initial_equity["default_value"] == {
        "kind": "number",
        "boolean_value": None,
        "number_value": "10000",
        "string_value": None,
        "datetime_value": None,
        "pair_value": None,
        "items": [],
        "entries": [],
    }


def test_catalog_lists_unconfigured_installed_factory_without_calling_it(monkeypatch):
    from cryptotrader.configuration.catalog import configuration_catalog

    entry_point = _entry_point(
        "unconfigured",
        "tests.factories.fake_signal_plugin:create_unconfigured_signal",
        "cryptotrader.signal_components",
    )
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: (entry_point,) if group == entry_point.group else ())

    catalog = configuration_catalog()

    definition = catalog.require_component("unconfigured")
    assert definition.parameter_model.model_validate({"window": 12, "debate": {"enabled": True}}).window == 12


def test_catalog_does_not_call_parameter_default_factory_while_describing_an_installed_plugin():
    from cryptotrader.configuration.catalog import PluginConfiguration
    from cryptotrader.configuration.fields import LocalizedText

    class FactoryDefaultParameters(BaseModel):
        value: int = Field(default_factory=lambda: (_ for _ in ()).throw(AssertionError("must not be called")))

    definition = PluginConfiguration(
        id="factory-default",
        label=LocalizedText("默认工厂", "Factory default"),
        description=LocalizedText("测试。", "Test."),
        parameter_model=FactoryDefaultParameters,
    )

    field = definition.fields[0]

    assert field.key == "value"
    assert field.default_value is None


def test_plugin_configuration_rejects_a_non_pydantic_parameter_model():
    from cryptotrader.configuration.catalog import PluginConfiguration
    from cryptotrader.configuration.fields import LocalizedText

    with pytest.raises(TypeError, match="parameter_model must be a BaseModel subclass"):
        PluginConfiguration(
            id="malformed",
            label=LocalizedText("错误", "Malformed"),
            description=LocalizedText("错误。", "Malformed."),
            parameter_model=object,
        )


def test_validation_rejects_unknown_parameters_and_accepts_nested_typed_plugin(monkeypatch):
    from cryptotrader.configuration.catalog import validate_configuration_parameters

    entry_point = _entry_point(
        "unconfigured",
        "tests.factories.fake_signal_plugin:create_unconfigured_signal",
        "cryptotrader.signal_components",
    )
    monkeypatch.setattr(metadata, "entry_points", lambda *, group: (entry_point,) if group == entry_point.group else ())
    document = runtime_document(
        signals=runtime_document().signals.model_copy(
            update={
                "components": (
                    SignalComponentConfig(
                        component_id="unconfigured",
                        enabled=True,
                        weight=1.0,
                        parameters={"window": 12, "debate": {"enabled": True, "rounds": 2}},
                    ),
                )
            }
        )
    )

    validate_configuration_parameters(document)

    invalid = document.model_copy(
        update={
            "signals": document.signals.model_copy(
                update={
                    "components": (
                        SignalComponentConfig(
                            component_id="unconfigured",
                            enabled=True,
                            weight=1.0,
                            parameters={"window": 12, "unknown": "rejected"},
                        ),
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="invalid plugin configuration parameters"):
        validate_configuration_parameters(invalid)


async def test_config_write_rejects_negative_paper_equity_without_echo_or_revision_change(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["connections"][-1]["parameters"] = {
        "initial_equity": -1,
        "raw_secret_marker": "must-not-echo",  # pragma: allowlist secret - redaction regression marker.
    }

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422
    assert "must-not-echo" not in response.text
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"]


async def test_config_write_rejects_only_unknown_paper_parameter_without_echo_or_revision_change(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["connections"][-1]["parameters"] = {
        "initial_equity": 10_000,
        "unknown": "must-not-echo",  # pragma: allowlist secret - redaction regression marker.
    }

    response = await api_harness.client.put(
        "/api/config",
        json={"expected_revision": current.json()["revision"], "document": document},
    )

    assert response.status_code == 422
    assert "must-not-echo" not in response.text
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"]
