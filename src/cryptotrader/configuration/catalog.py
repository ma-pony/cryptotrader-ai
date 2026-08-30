"""Installed plugin definitions and strict parameter validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from cryptotrader.configuration.fields import ConfigurationField, LocalizedText, configuration_fields

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

SIGNAL_COMPONENT_GROUP = "cryptotrader.signal_components"
MARKET_SOURCE_GROUP = "cryptotrader.market_sources"
VENUE_ADAPTER_GROUP = "cryptotrader.venue_adapters"


@dataclass(frozen=True)
class PluginConfiguration:
    id: str
    label: LocalizedText
    description: LocalizedText
    parameter_model: type[BaseModel]
    environments: tuple[str, ...] = ()
    credential_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_model, type) or not issubclass(self.parameter_model, BaseModel):
            raise TypeError("parameter_model must be a BaseModel subclass")

    @property
    def fields(self) -> tuple[ConfigurationField, ...]:
        return configuration_fields(self.parameter_model)

    def validate_environment(self, environment: str) -> None:
        if environment not in self.environments:
            raise ValueError(f"unsupported environment {environment!r} for {self.id}")


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


def _require(definitions: dict[str, PluginConfiguration], kind: str, plugin_id: str) -> PluginConfiguration:
    try:
        return definitions[plugin_id]
    except KeyError:
        raise ValueError(f"uninstalled {kind}: {plugin_id}") from None


def configured_factory(configuration: PluginConfiguration):
    """Attach the required immutable declaration to an installed factory."""

    def decorate(factory):
        factory.configuration = configuration
        return factory

    return decorate


def require_factory_configuration(plugin_id: str, factory: Callable[..., Any]) -> PluginConfiguration:
    if not callable(factory):
        raise TypeError(f"factory for {plugin_id} is not callable")
    configuration = getattr(factory, "configuration", None)
    if not isinstance(configuration, PluginConfiguration):
        raise TypeError(f"factory for {plugin_id} must declare PluginConfiguration")
    if configuration.id != plugin_id:
        raise ValueError(f"plugin configuration id mismatch: entry point {plugin_id}, configuration {configuration.id}")
    return configuration


def installed_plugin_factories(
    group: str,
    builtins: dict[str, Callable[..., Any]],
    *,
    entry_points: Iterable[Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Load factory metadata only; constructing a plugin remains a runtime operation."""
    factories = dict(builtins)
    discovered = metadata.entry_points(group=group) if entry_points is None else entry_points
    for entry_point in discovered:
        factory = entry_point.load()
        installed = factories.get(entry_point.name)
        if installed is not None:
            if installed is factory:
                continue
            raise ValueError(f"duplicate installed plugin id: {entry_point.name}")
        factories[entry_point.name] = factory
    for plugin_id, factory in factories.items():
        require_factory_configuration(plugin_id, factory)
    return factories


def _builtin_factories() -> tuple[dict[str, Callable[..., Any]], dict[str, Callable[..., Any]]]:
    from cryptotrader.market_sources.default import create_source
    from cryptotrader.signals.components.kronos import create_component as create_kronos
    from cryptotrader.signals.components.llm_committee import create_component as create_llm_committee

    return (
        {"kronos": create_kronos, "llm_committee": create_llm_committee},
        {"default": create_source},
    )


def configuration_catalog() -> ConfigurationCatalog:
    components, market_sources = _builtin_factories()
    component_factories = installed_plugin_factories(SIGNAL_COMPONENT_GROUP, components)
    market_source_factories = installed_plugin_factories(MARKET_SOURCE_GROUP, market_sources)
    venue_factories = installed_plugin_factories(VENUE_ADAPTER_GROUP, {})
    return ConfigurationCatalog(
        components={plugin_id: factory.configuration for plugin_id, factory in component_factories.items()},
        venues={plugin_id: factory.configuration for plugin_id, factory in venue_factories.items()},
        market_sources={plugin_id: factory.configuration for plugin_id, factory in market_source_factories.items()},
    )


def validate_configuration_parameters(document) -> None:
    """Reject unknown or invalid installed-plugin parameters before a CAS replacement."""
    catalog = configuration_catalog()
    try:
        for component in document.signals.components:
            catalog.require_component(component.component_id).parameter_model.model_validate(dict(component.parameters))
        catalog.require_market_source(document.market_data.source_id).parameter_model.model_validate(
            dict(document.market_data.parameters)
        )
        for connection in document.execution.connections:
            definition = catalog.require_venue(connection.adapter_id)
            definition.validate_environment(connection.environment)
            definition.parameter_model.model_validate(dict(connection.parameters))
    except (ValidationError, TypeError, ValueError):
        raise ValueError("invalid plugin configuration parameters") from None
