"""启动时注册的信号组件目录。"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cryptotrader.signals.component import SignalComponent

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.profiles.models import SignalProfile


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: str
    display_name: str
    description: str


class SignalComponentRegistry:
    """保存已安装组件; 运行期间不允许动态修改代码。"""

    def __init__(self, components: Iterable[SignalComponent] = ()) -> None:
        self._components: dict[str, SignalComponent] = {}
        for component in components:
            self.register(component)

    def register(self, component: SignalComponent) -> None:
        if not isinstance(component, SignalComponent):
            raise TypeError("component does not implement SignalComponent")
        if component.id in self._components:
            raise ValueError(f"duplicate signal component id: {component.id}")
        self._components[component.id] = component

    def load_factory(self, path: str) -> None:
        module_name, separator, attribute_name = path.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(f"signal component factory must use module:attribute syntax: {path}")
        factory = getattr(importlib.import_module(module_name), attribute_name)
        component = factory()
        if not isinstance(component, SignalComponent):
            raise TypeError(f"factory {path} did not return SignalComponent")
        self.register(component)

    def get(self, component_id: str) -> SignalComponent:
        return self._components[component_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(self._components)

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
