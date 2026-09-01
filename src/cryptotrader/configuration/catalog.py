"""Code-owned extension declarations; describing them never constructs a runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, SecretStr, ValidationError

from cryptotrader.configuration.fields import (
    ConfigurationField,
    CredentialField,
    LocalizedText,
    configuration_fields,
    credential_fields,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cryptotrader.venues.models import VenueCapabilities


@dataclass(frozen=True)
class EnvironmentDefinition:
    id: str
    label: LocalizedText
    capital_scope: Literal["simulated", "real"]
    parameter_model: type[BaseModel] | None = None
    credential_model: type[BaseModel] | None = None
    margin_modes: tuple[str, ...] | None = None
    leverage_minimum: int | None = None
    leverage_maximum: int | None = None
    capabilities: VenueCapabilities | None = None

    def __post_init__(self):
        if not self.id.strip() or self.capital_scope not in {"simulated", "real"}:
            raise ValueError("invalid environment declaration")


@dataclass(frozen=True)
class ComponentDependency:
    kind: Literal["market", "model_service", "local_artifact", "context"]
    key: str
    label: LocalizedText
    configuration_path: str


@dataclass(frozen=True)
class PluginConfiguration:
    id: str
    label: LocalizedText
    description: LocalizedText
    parameter_model: type[BaseModel]
    environments: tuple[EnvironmentDefinition, ...] = ()
    credential_model: type[BaseModel] | None = None
    margin_modes: tuple[str, ...] = ()
    leverage_minimum: int = 1
    leverage_maximum: int | None = None
    capabilities: VenueCapabilities | None = None
    dependency_resolver: Callable[[BaseModel], tuple[ComponentDependency, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_model, type) or not issubclass(self.parameter_model, BaseModel):
            raise TypeError("parameter_model must be a BaseModel subclass")
        if {field.key for field in self.fields} & {field.key for field in self.credential_fields}:
            raise ValueError("credential fields cannot be ordinary parameters")
        if len({environment.id for environment in self.environments}) != len(self.environments):
            raise ValueError("duplicate environment id")

    @property
    def fields(self) -> tuple[ConfigurationField, ...]:
        return configuration_fields(self.parameter_model)

    @property
    def credential_fields(self) -> tuple[CredentialField, ...]:
        return credential_fields(self.credential_model) if self.credential_model is not None else ()

    def require_environment(self, environment: str) -> EnvironmentDefinition:
        for item in self.environments:
            if item.id == environment:
                return item
        raise ValueError(f"unsupported environment for {self.id}")

    def validate_environment(self, environment: str) -> None:
        self.require_environment(environment)

    def for_environment(self, environment: str) -> PluginConfiguration:
        selected = self.require_environment(environment)
        overrides = {
            key: getattr(selected, key)
            for key in (
                "parameter_model",
                "credential_model",
                "margin_modes",
                "leverage_minimum",
                "leverage_maximum",
                "capabilities",
            )
            if getattr(selected, key) is not None
        }
        return replace(self, **overrides)

    def dependencies(self, parameters: Mapping[str, Any]) -> tuple[ComponentDependency, ...]:
        validated = self.parameter_model.model_validate(dict(parameters))
        return self.dependency_resolver(validated) if self.dependency_resolver else ()


@dataclass(frozen=True)
class ConfigurationCatalog:
    components: dict[str, PluginConfiguration]
    venues: dict[str, PluginConfiguration]
    market_sources: dict[str, PluginConfiguration]

    def require_component(self, component_id: str) -> PluginConfiguration:
        return _require(self.components, "signal component", component_id)

    def require_venue(self, venue_id: str) -> PluginConfiguration:
        return _require(self.venues, "venue", venue_id)

    def require_market_source(self, source_id: str) -> PluginConfiguration:
        return _require(self.market_sources, "market source", source_id)


def _require(definitions, kind, extension_id):
    try:
        return definitions[extension_id]
    except KeyError:
        raise ValueError(f"unregistered {kind}: {extension_id}") from None


def configuration_catalog() -> ConfigurationCatalog:
    from cryptotrader.configuration import registry

    extensions = registry.get_extension_registry()
    return ConfigurationCatalog(
        **{
            group: {key: item.configuration for key, item in getattr(extensions, group).items()}
            for group in ("components", "venues", "market_sources")
        }
    )


def require_environment(adapter_id: str, environment: str) -> EnvironmentDefinition:
    return configuration_catalog().require_venue(adapter_id).require_environment(environment)


def validate_configuration_parameters(document) -> None:
    catalog = configuration_catalog()
    try:
        for component in document.signals.components:
            catalog.require_component(component.component_id).parameter_model.model_validate(dict(component.parameters))
        catalog.require_market_source(document.market_data.source_id).parameter_model.model_validate(
            dict(document.market_data.parameters)
        )
        for connection in document.execution.connections:
            definition = catalog.require_venue(connection.adapter_id).for_environment(connection.environment)
            definition.parameter_model.model_validate(dict(connection.parameters))
            if connection.margin_mode not in definition.margin_modes:
                raise ValueError("unsupported margin mode")
            if connection.leverage < definition.leverage_minimum or (
                definition.leverage_maximum is not None and connection.leverage > definition.leverage_maximum
            ):
                raise ValueError("unsupported leverage")
    except (ValidationError, TypeError, ValueError):
        raise ValueError("invalid registered component parameters") from None


class CredentialValidationError(ValueError):
    """Only field paths and error codes cross the credential-validation boundary."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("invalid venue credentials")


def validate_venue_credentials(adapter_id: str, environment: str, values: dict[str, str]):
    from cryptotrader.runtime_config.secrets import CredentialPayload

    definition = configuration_catalog().require_venue(adapter_id).for_environment(environment)
    if definition.credential_model is None or not definition.credential_fields:
        raise CredentialValidationError([{"field": "values", "code": "credentials_not_supported"}])
    fields = {field.key for field in definition.credential_fields}
    if set(values) - fields:
        raise CredentialValidationError([{"field": "values", "code": "extra_forbidden"}])
    try:
        validated = definition.credential_model.model_validate(values)
    except ValidationError as error:
        errors = [
            {"field": ".".join(map(str, item["loc"])), "code": item["type"]}
            for item in error.errors(include_input=False, include_context=False)
        ]
        raise CredentialValidationError(errors) from None
    payload = {key: value for key, value in validated.model_dump().items() if value is not None}
    empty = [
        key
        for key, value in payload.items()
        if not isinstance(value, SecretStr) or not value.get_secret_value().strip()
    ]
    if empty:
        raise CredentialValidationError([{"field": key, "code": "empty"} for key in empty])
    return CredentialPayload(values=payload)
