"""启动时注册的信号组件目录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptotrader.configuration import registry as extension_registry
from cryptotrader.configuration.registry import ComponentFactoryContext
from cryptotrader.signals.component import SignalComponent

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.profiles.models import SignalProfile
    from cryptotrader.runtime_config.models import RuntimeConfigDocument


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: str
    display_name: str
    description: str


class SignalComponentRegistry:
    """保存后端已注册组件; 运行期间不允许动态修改代码。"""

    def __init__(self, components: Iterable[SignalComponent] = ()) -> None:
        self._components: dict[str, SignalComponent] = {}
        self._registered_ids: frozenset[str] = frozenset()
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
        """Instantiate configured components from the single code-owned registry."""
        registrations = extension_registry.get_extension_registry().components
        configured_ids = tuple(component.component_id for component in document.signals.components)
        missing = sorted(set(configured_ids) - set(registrations))
        if missing:
            raise ValueError(f"unregistered signal component ids: {', '.join(missing)}")
        registry = cls()
        registry._registered_ids = frozenset(registrations)
        context = ComponentFactoryContext(document, sink, llm_gateway_key, llm_factory_builder)
        for component_id in configured_ids:
            component = registrations[component_id].factory(context)
            if not isinstance(component, SignalComponent):
                raise TypeError(f"factory for {component_id} did not return SignalComponent")
            if component.id != component_id:
                raise ValueError(
                    f"signal component factory id mismatch: registration {component_id}, component {component.id}"
                )
            registry.register(component)
        return registry

    def register(self, component: SignalComponent) -> None:
        if not isinstance(component, SignalComponent):
            raise TypeError("component does not implement SignalComponent")
        if component.id in self._components:
            raise ValueError(f"duplicate signal component id: {component.id}")
        self._components[component.id] = component
        self._registered_ids = self._registered_ids | {component.id}

    def get(self, component_id: str) -> SignalComponent:
        return self._components[component_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(self._components)

    def registered_ids(self) -> frozenset[str]:
        return self._registered_ids

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
