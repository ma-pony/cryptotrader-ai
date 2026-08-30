"""启动时注册的信号组件目录。"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any

from cryptotrader.configuration.catalog import require_factory_configuration
from cryptotrader.signals.component import SignalComponent

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.runtime_config.models import RuntimeConfigDocument

_ENTRY_POINT_GROUP = "cryptotrader.signal_components"


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: str
    display_name: str
    description: str


class SignalComponentRegistry:
    """保存已安装组件; 运行期间不允许动态修改代码。"""

    def __init__(self, components: Iterable[SignalComponent] = ()) -> None:
        self._components: dict[str, SignalComponent] = {}
        self._installed_ids: frozenset[str] = frozenset()
        for component in components:
            self.register(component)

    @classmethod
    def discover(
        cls,
        document: RuntimeConfigDocument,
        sink: CycleEventSink,
        *,
        llm_gateway_key: str = "",
        llm_factory_builder=None,
    ) -> SignalComponentRegistry:
        """Instantiate configured components from code-owned and installed factories."""
        from cryptotrader.signals.components.kronos import create_component as create_kronos
        from cryptotrader.signals.components.llm_committee import create_component as create_llm_committee

        factories: dict[str, Callable[[RuntimeConfigDocument, CycleEventSink], Any]] = {
            "kronos": create_kronos,
            "llm_committee": create_llm_committee,
        }
        for entry_point in metadata.entry_points(group=_ENTRY_POINT_GROUP):
            factory = entry_point.load()
            installed = factories.get(entry_point.name)
            if installed is not None:
                if installed is factory and entry_point.name in {"kronos", "llm_committee"}:
                    continue
                raise ValueError(f"duplicate signal component id: {entry_point.name}")
            factories[entry_point.name] = factory

        for component_id, factory in factories.items():
            require_factory_configuration(component_id, factory)

        configured_ids = tuple(component.component_id for component in document.signals.components)
        missing = sorted(set(configured_ids) - set(factories))
        if missing:
            raise ValueError(f"uninstalled signal component ids: {', '.join(missing)}")

        registry = cls()
        registry._installed_ids = frozenset(factories)
        for component_id in configured_ids:
            if component_id == "llm_committee":
                component = factories[component_id](
                    document,
                    sink,
                    llm_gateway_key=llm_gateway_key,
                    llm_factory_builder=llm_factory_builder,
                )
            else:
                component = factories[component_id](document, sink)
            if not isinstance(component, SignalComponent):
                raise TypeError(f"factory for {component_id} did not return SignalComponent")
            if component.id != component_id:
                raise ValueError(
                    f"signal component factory id mismatch: entry point {component_id}, component {component.id}"
                )
            registry.register(component)
        return registry

    def register(self, component: SignalComponent) -> None:
        if not isinstance(component, SignalComponent):
            raise TypeError("component does not implement SignalComponent")
        if component.id in self._components:
            raise ValueError(f"duplicate signal component id: {component.id}")
        self._components[component.id] = component
        self._installed_ids = self._installed_ids | {component.id}

    def load_factory(self, path: str) -> None:
        module_name, separator, attribute_name = path.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(f"signal component factory must use module:attribute syntax: {path}")
        factory = getattr(importlib.import_module(module_name), attribute_name)
        configuration = getattr(factory, "configuration", None)
        if configuration is None:
            raise TypeError(f"factory {path} must declare PluginConfiguration")
        require_factory_configuration(configuration.id, factory)
        component = factory()
        if not isinstance(component, SignalComponent):
            raise TypeError(f"factory {path} did not return SignalComponent")
        self.register(component)

    def get(self, component_id: str) -> SignalComponent:
        return self._components[component_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(self._components)

    def installed_ids(self) -> frozenset[str]:
        return self._installed_ids

    def components(self) -> tuple[SignalComponent, ...]:
        return tuple(self._components.values())

    def enabled(self, profile: SignalProfile) -> tuple[SignalComponent, ...]:
        return tuple(self._components[item.component_id] for item in profile.components if item.enabled)

    def metadata(self) -> tuple[ComponentMetadata, ...]:
        return tuple(
            ComponentMetadata(
                component_id=component.id,
                display_name=component.display_name,
                description=component.description,
            )
            for component in self._components.values()
        )
