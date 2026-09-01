"""Configuration-only capability checks. Never instantiate accounts or load models."""
# ruff: noqa: RUF001 - Chinese public messages use Chinese punctuation.

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cryptotrader.configuration.catalog import configuration_catalog


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadinessReason(_Out):
    code: str
    message: str
    path: str


class CapabilityOut(_Out):
    ready: bool
    reasons: tuple[ReadinessReason, ...]


class DependencyOut(CapabilityOut):
    kind: Literal["market", "model_service", "local_artifact", "context"]
    key: str
    label: str


class ComponentReadinessOut(CapabilityOut):
    component_id: str
    enabled: bool
    dependencies: tuple[DependencyOut, ...]


class ReadinessOut(_Out):
    analysis: CapabilityOut
    trading: CapabilityOut
    components: tuple[ComponentReadinessOut, ...]
    saved_revision: int
    applied_revision: int | None
    apply_error: str | None
    automation_enabled: bool
    latest_run_at: datetime | None
    execution_pairs: tuple[str, ...]


class ScopeConnectionOut(_Out):
    connection_id: str
    label: str
    environment: str
    enabled: bool


class TradingBookOut(_Out):
    book_id: str
    label: str
    capital_scope: Literal["simulated", "real"]
    enabled: bool
    eligible: bool
    reasons: tuple[ReadinessReason, ...]
    hitl_required: bool
    connections: tuple[ScopeConnectionOut, ...]


class TradingScopeOut(_Out):
    pair: str
    saved_revision: int
    ready: bool
    reasons: tuple[ReadinessReason, ...]
    books: tuple[TradingBookOut, ...]


def reason(code, message, path):
    return ReadinessReason(code=code, message=message, path=path)


def local_artifact_available(dependency) -> bool:
    """Inspect exactly the gate file or model weight files; no downloads/deserialization."""
    from huggingface_hub import try_to_load_from_cache

    key = dependency.key.strip()
    if not key:
        return False
    path = Path(key).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    is_model_resource = dependency.configuration_path.endswith(("model_name", "tokenizer_name"))
    if not is_model_resource:
        return path.is_file() and path.stat().st_size > 0
    filenames = ("model.safetensors", "pytorch_model.bin")
    if path.is_dir():
        return any((path / name).is_file() and (path / name).stat().st_size > 0 for name in filenames)
    if path.is_absolute() and key.startswith(("/", ".", "~")):
        return False
    for filename in filenames:
        try:
            cached = try_to_load_from_cache(key, filename)
        except (ValueError, OSError):
            continue
        if isinstance(cached, str) and Path(cached).is_file() and Path(cached).stat().st_size > 0:
            return True
    return False


async def component_readiness(snapshot, repository):
    catalog = configuration_catalog()
    document = snapshot.document
    items = []
    for component in document.signals.components:
        dependencies = []
        definition = catalog.require_component(component.component_id)
        for dependency in definition.dependencies(component.parameters):
            available = True
            if dependency.kind == "local_artifact":
                available = local_artifact_available(dependency)
            elif dependency.kind == "model_service":
                state = await repository.token_state(dependency.key)
                config = getattr(document, dependency.configuration_path, None)
                available = state.configured and bool(getattr(config, "base_url", "").strip())
            else:
                available = document.market_data.source_id in catalog.market_sources
            missing = (
                ()
                if available
                else (
                    reason(
                        "dependency_missing",
                        f"{dependency.label.zh_CN}未就绪，请补齐配置或服务器本地资源。",
                        dependency.configuration_path,
                    ),
                )
            )
            dependencies.append(
                DependencyOut(
                    kind=dependency.kind,
                    key=dependency.key,
                    label=dependency.label.zh_CN,
                    ready=available,
                    reasons=missing,
                )
            )
        reasons = tuple(item for dep in dependencies for item in dep.reasons)
        items.append(
            ComponentReadinessOut(
                component_id=component.component_id,
                enabled=component.enabled,
                ready=not reasons,
                reasons=reasons,
                dependencies=tuple(dependencies),
            )
        )
    reasons = tuple(item for component in items if component.enabled for item in component.reasons)
    if not any(component.enabled for component in items):
        reasons += (reason("no_components", "请启用至少一个信号组件。", "signals.components"),)
    return tuple(items), CapabilityOut(ready=not reasons, reasons=reasons)


async def book_scope(snapshot, repository):
    document = snapshot.document
    connections = {item.id: item for item in document.execution.connections}
    catalog = configuration_catalog()
    books = []
    for book in document.execution.books:
        reasons = []
        members = []
        if not book.enabled:
            reasons.append(reason("book_disabled", "资金池已停用。", f"execution.books.{book.id}.enabled"))
        if book.capital_scope == "real" and not document.execution.live_order_execution_enabled:
            reasons.append(
                reason("real_authorization_missing", "尚未授权真实资金下单。", "execution.live_order_execution_enabled")
            )
        for allocation in book.allocations:
            if not allocation.enabled:
                continue
            connection = connections[allocation.connection_id]
            members.append(
                ScopeConnectionOut(
                    connection_id=connection.id,
                    label=connection.label,
                    environment=connection.environment,
                    enabled=connection.enabled,
                )
            )
            path = f"execution.connections.{connection.id}"
            if not connection.enabled or connection.canary_only:
                reasons.append(reason("connection_disabled", f"账户 {connection.label} 未启用执行。", path))
            definition = catalog.require_venue(connection.adapter_id).for_environment(connection.environment)
            if definition.credential_fields:
                configured = (
                    bool(connection.credential_ref)
                    and (await repository.credential_state(connection.credential_ref)).configured
                )
                if not configured:
                    reasons.append(reason("credentials_missing", f"账户 {connection.label} 缺少凭据。", path))
        if not members:
            reasons.append(reason("no_allocations", "资金池未分配执行账户。", f"execution.books.{book.id}.allocations"))
        books.append(
            TradingBookOut(
                book_id=book.id,
                label=book.label,
                capital_scope=book.capital_scope,
                enabled=book.enabled,
                eligible=not reasons,
                reasons=tuple(reasons),
                hitl_required=book.hitl_required,
                connections=tuple(members),
            )
        )
    return tuple(books)


def trading_reasons(snapshot):
    reasons = []
    if not snapshot.operational:
        reasons.append(reason("configuration_not_applied", "当前保存的配置尚未成功应用。", "applied_revision"))
    if not snapshot.document.infrastructure.redis_url.strip():
        reasons.append(reason("execution_lock_missing", "交易需要配置 Redis 执行锁。", "infrastructure.redis_url"))
    if not snapshot.document.execution.pairs:
        reasons.append(reason("execution_pairs_empty", "请先配置交易品种范围。", "execution.pairs"))
    return reasons
