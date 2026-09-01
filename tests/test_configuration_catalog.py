"""Code-registered configuration catalog and parameter-validation contracts."""

from __future__ import annotations

from dataclasses import replace
from importlib import metadata

import pytest
from pydantic import BaseModel, Field

from cryptotrader.runtime_config.models import SignalComponentConfig
from tests.factories.runtime_config import runtime_document
from tests.test_runtime_config_api import active_payload


def test_market_adapter_description_names_code_registration_truthfully():
    from cryptotrader.configuration.parameters import DefaultMarketSourceParameters

    description = DefaultMarketSourceParameters.field_descriptions["market_adapter_id"]
    assert "后端已注册" in description.zh_CN
    assert "registered by the backend" in description.en_US
    assert "已安装" not in description.zh_CN
    assert "Installed" not in description.en_US


def test_catalog_builtin_venues_do_not_depend_on_package_metadata(monkeypatch):
    from cryptotrader.configuration.catalog import configuration_catalog

    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: ())
    assert set(configuration_catalog().venues) == {"paper", "okx", "bybit"}


def test_catalog_does_not_instantiate_extensions(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import configuration_catalog
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    definition = configuration_catalog().require_venue("sample_venue")
    assert definition.environments[0].capital_scope == "simulated"
    assert [(field.key, field.required) for field in definition.credential_fields] == [
        ("access_token", True),
        ("tenant_pin", False),
    ]
    assert calls == []


async def test_environment_definition_is_readable_without_constructing_a_session(api_harness):
    response = await api_harness.client.get("/api/config/catalog/venues/paper?environment=paper")
    assert response.status_code == 200
    assert response.json()["environment"]["capital_scope"] == "simulated"
    assert response.json()["credential_fields"] == []
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())


async def test_custom_environment_and_dynamic_credentials_can_be_saved(api_harness, monkeypatch, caplog):
    from cryptotrader.configuration import registry
    from cryptotrader.venues.registry import VenueAdapterRegistry
    from tests.factories.workbench_extensions import sample_registry

    extensions, _ = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    api_harness.runtime.venue_registry = VenueAdapterRegistry.discover()
    body = {
        "expected_revision": 1,
        "id": "sample-account",
        "label": "Sample",
        "adapter_id": "sample_venue",
        "environment": "sandbox",
        "enabled": False,
        "leverage": 1,
        "margin_mode": "cross",
        "canary_only": False,
        "parameters": {"account_code": "account-1"},
    }
    response = await api_harness.client.post("/api/venue-connections", json=body)
    assert response.status_code == 201, response.text
    secret = "sample-secret-value"  # pragma: allowlist secret
    response = await api_harness.client.put(
        "/api/venue-connections/sample-account/credentials",
        json={
            "expected_revision": response.json()["revision"],
            "values": {"access_token": secret},
        },
    )
    assert response.status_code == 200, response.text
    assert secret not in response.text
    payload = await api_harness.runtime.repository.reveal_credentials("venue-connection:sample-account")
    assert payload.values["access_token"].get_secret_value() == secret
    assert "tenant_pin" not in payload.values
    adapter = api_harness.runtime.venue_registry.require("sample_venue")
    saved = await api_harness.runtime.repository.get_or_create()
    connection = saved.document.execution.connections[-1]
    await adapter.connect(connection, payload)
    assert adapter.received_values == {"access_token": secret}
    assert secret not in saved.document.model_dump_json()
    config_response = await api_harness.client.get("/api/config")
    assert secret not in config_response.text
    assert secret not in caplog.text
    invalid = await api_harness.client.put(
        "/api/venue-connections/sample-account/credentials",
        json={
            "expected_revision": saved.revision,
            "values": {"unknown": secret},
        },
    )
    assert invalid.status_code == 422
    assert secret not in invalid.text
    assert (await api_harness.runtime.repository.get_or_create()).revision == saved.revision


def test_custom_environment_capital_scope_comes_from_declaration(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.runtime_config.models import validate_runtime_document
    from tests.factories.runtime_config import allocation, book, connection
    from tests.factories.workbench_extensions import sample_registry

    extensions, _ = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    sample = connection(
        "sample",
        environment="sandbox",
        adapter_id="sample_venue",
        credential_ref="sample-ref",
        parameters={"account_code": "a"},
    )
    document = runtime_document()
    document = document.model_copy(
        update={
            "execution": document.execution.model_copy(
                update={
                    "connections": (sample,),
                    "books": (book("sample-book", allocations=(allocation("sample"),)),),
                }
            )
        }
    )
    validate_runtime_document(
        document, set(extensions.components), set(extensions.venues), set(extensions.market_sources)
    )


@pytest.mark.parametrize(
    ("protected", "applying", "expected"), [(False, False, 200), (True, False, 401), (False, True, 503)]
)
async def test_environment_definition_setup_admission(api_harness, protected, applying, expected):
    snapshot = api_harness.runtime.snapshot
    document = snapshot.document.model_copy(
        update={
            "security": snapshot.document.security.model_copy(update={"enabled": protected}),
        }
    )
    api_harness.runtime.snapshot = replace(snapshot, document=document)
    api_harness.runtime.application_in_progress = applying
    response = await api_harness.client.get("/api/config/catalog/venues/okx?environment=demo")
    assert response.status_code == expected


def test_environment_definition_selects_declared_overrides_without_factory_calls(monkeypatch):
    from api.routes.config import venue_definition
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import EnvironmentDefinition
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.parameters import EmptyParameters
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    item = extensions.venues["sample_venue"]
    configuration = replace(
        item.configuration,
        environments=(
            *item.configuration.environments,
            EnvironmentDefinition(
                "local",
                LocalizedText("本地", "Local"),
                "simulated",
                parameter_model=EmptyParameters,
                credential_model=EmptyParameters,
                margin_modes=("isolated",),
                leverage_maximum=1,
            ),
        ),
    )
    extensions.venues["sample_venue"] = replace(item, configuration=configuration)
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    definition = venue_definition("sample_venue", "local")
    assert definition.fields == []
    assert definition.credential_fields == []
    assert definition.margin_modes == ["isolated"]
    assert definition.leverage_maximum == 1
    assert calls == []


def test_component_dependencies_are_declared_from_parameters_without_resources():
    from cryptotrader.configuration.catalog import configuration_catalog

    catalog = configuration_catalog()
    dependencies = catalog.require_component("kronos").dependencies(
        {
            "gate_path": "missing-gate.pkl",
            "timeframe": "1h",
            "model_name": "fixture/custom-kronos",
            "tokenizer_name": "missing-tokenizer-directory",
        }
    )
    assert [(item.kind, item.key, item.configuration_path) for item in dependencies] == [
        ("market", "1h", "market_data"),
        ("local_artifact", "missing-gate.pkl", "signals.components.kronos.parameters.gate_path"),
        ("local_artifact", "fixture/custom-kronos", "signals.components.kronos.parameters.model_name"),
        ("local_artifact", "missing-tokenizer-directory", "signals.components.kronos.parameters.tokenizer_name"),
        ("context", "kronos_aux", "market_data"),
    ]
    committee = catalog.require_component("llm_committee").dependencies({})
    assert ("model_service", "llm-gateway", "llm") in [
        (item.kind, item.key, item.configuration_path) for item in committee
    ]


def test_credential_validation_errors_and_parameter_documents_exclude_secret_values(monkeypatch):
    import json

    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import (
        CredentialValidationError,
        validate_configuration_parameters,
        validate_venue_credentials,
    )
    from tests.factories.runtime_config import connection
    from tests.factories.workbench_extensions import sample_registry

    extensions, _ = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    secret = "must-not-leak-dynamic-value"  # pragma: allowlist secret
    with pytest.raises(CredentialValidationError) as error:
        validate_venue_credentials("sample_venue", "sandbox", {"access_token": [secret]})
    assert secret not in str(error.value)
    assert secret not in json.dumps(error.value.errors)
    with pytest.raises(ValueError, match="invalid registered component parameters"):
        validate_configuration_parameters(
            runtime_document(
                connections=(
                    connection(
                        "sample",
                        environment="sandbox",
                        adapter_id="sample_venue",
                        credential_ref="sample",
                        parameters={"account_code": "normal", "tenant_pin": secret},
                    ),
                )
            )
        )


def test_builtin_capability_declarations_match_runtime_adapters():
    from cryptotrader.configuration.registry import get_extension_registry

    for entry in get_extension_registry().venues.values():
        adapter = entry.factory()
        for environment in entry.configuration.environments:
            assert entry.configuration.for_environment(environment.id).capabilities == adapter.capabilities(
                environment.id
            )


@pytest.mark.parametrize(
    ("protected", "applying", "expected"),
    [(False, False, 200), (True, False, 401), (False, True, 503)],
)
async def test_inactive_catalog_http_admission_keeps_authentication_and_application_barrier(
    api_harness, protected, applying, expected
):
    snapshot = api_harness.runtime.snapshot
    document = snapshot.document.model_copy(
        update={
            "security": snapshot.document.security.model_copy(update={"enabled": protected}),
            "infrastructure": snapshot.document.infrastructure.model_copy(update={"redis_url": ""}),
        }
    )
    api_harness.runtime.snapshot = replace(snapshot, document=document)
    api_harness.runtime.application_in_progress = applying

    response = await api_harness.client.get("/api/config/catalog")

    assert response.status_code == expected
    if expected == 200:
        assert any(venue["id"] == "paper" for venue in response.json()["venues"])
    elif applying:
        assert response.json() == {"detail": "Runtime configuration is being applied"}
    assert all(not adapter.connect_calls for adapter in api_harness.adapters.values())


def _entry_point(name: str, value: str, group: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(name=name, value=value, group=group)


async def test_catalog_exposes_paper_funding_field(api_harness):
    response = await api_harness.client.get("/api/config/catalog")

    assert response.status_code == 200
    paper = next(item for item in response.json()["venues"] if item["id"] == "paper")
    assert paper["label"] == {"zh_CN": "本地模拟器", "en_US": "Paper trading"}
    assert [(item["id"], item["capital_scope"]) for item in paper["environments"]] == [("paper", "simulated")]
    assert paper["credential_fields"] == []
    assert paper["margin_modes"] == ["cross"]
    initial_equity = next(field for field in paper["fields"] if field["key"] == "initial_equity")
    assert initial_equity["kind"] == "number"
    assert initial_equity["unit"] == "USDT"
    assert initial_equity["minimum"] is None
    assert initial_equity["exclusive_minimum"] == 0
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


async def test_catalog_keeps_consumed_external_margin_modes(api_harness):
    response = await api_harness.client.get("/api/config/catalog")
    venues = {item["id"]: item for item in response.json()["venues"]}
    for adapter in ("okx", "bybit"):
        assert venues[adapter]["margin_modes"] == ["cross", "isolated"]


@pytest.mark.parametrize("constraint", ["gt", "ge", "lt", "le"])
def test_catalog_preserves_numeric_bound_semantics(constraint):
    from api.routes.config import _plugin_field_out
    from cryptotrader.configuration.fields import configuration_fields

    class Parameters(BaseModel):
        amount: float = Field(**{constraint: 1})

    field = _plugin_field_out(configuration_fields(Parameters)[0]).model_dump()
    expected = {"minimum": None, "maximum": None, "exclusive_minimum": None, "exclusive_maximum": None}
    expected[{"gt": "exclusive_minimum", "ge": "minimum", "lt": "exclusive_maximum", "le": "maximum"}[constraint]] = 1
    assert {key: field[key] for key in expected} == expected


async def test_config_write_rejects_zero_paper_equity_before_persistence(api_harness):
    current = await api_harness.client.get("/api/config")
    document = active_payload()
    document["execution"]["connections"][-1]["parameters"] = {"initial_equity": 0}
    response = await api_harness.client.put(
        "/api/config", json={"expected_revision": current.json()["revision"], "document": document}
    )
    assert response.status_code == 422
    assert (await api_harness.client.get("/api/config")).json()["revision"] == current.json()["revision"]


def test_catalog_lists_unconfigured_installed_factory_without_calling_it(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import PluginConfiguration, configuration_catalog
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.registry import ExtensionRegistration
    from tests.factories.fake_signal_plugin import UnconfiguredSignalParameters, create_unconfigured_signal

    extensions = registry.get_extension_registry()
    label = LocalizedText("测试", "Test")
    extensions.components["unconfigured"] = ExtensionRegistration(
        PluginConfiguration("unconfigured", label, label, UnconfiguredSignalParameters),
        lambda context: create_unconfigured_signal(context.document, context.events),
    )
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)

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


def test_validation_rejects_unknown_parameters_and_accepts_nested_typed_component(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import PluginConfiguration, validate_configuration_parameters
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.registry import ExtensionRegistration
    from tests.factories.fake_signal_plugin import UnconfiguredSignalParameters, create_unconfigured_signal

    extensions = registry.get_extension_registry()
    label = LocalizedText("测试", "Test")
    extensions.components["unconfigured"] = ExtensionRegistration(
        PluginConfiguration("unconfigured", label, label, UnconfiguredSignalParameters),
        lambda context: create_unconfigured_signal(context.document, context.events),
    )
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
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
    with pytest.raises(ValueError, match="invalid registered component parameters"):
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
