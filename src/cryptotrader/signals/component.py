"""信号组件协议及其单组件执行错误。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cryptotrader.signals.models import ComponentSignal, DataRequirements, SignalContext


@runtime_checkable
class SignalComponent(Protocol):
    id: str
    display_name: str
    description: str

    def requirements(self) -> DataRequirements:
        raise NotImplementedError

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        raise NotImplementedError


class ComponentExecutionError(RuntimeError):
    """一个已识别组件无法产生合法信号。"""

    def __init__(self, component_id: str, cause: BaseException, *, stage: str = "evaluation") -> None:
        self.component_id = component_id
        self.cause = cause
        self.stage = stage
        super().__init__(f"signal component {component_id} failed: {type(cause).__name__}: {cause}")
