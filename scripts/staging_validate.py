"""只读 staging 门禁：数据库配置、运行时和已启用平台连接。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StepResult:
    idx: int
    name: str
    status: str
    duration_ms: int
    error: str = ""

    def fmt(self) -> str:
        message = f"[step {self.idx}] {self.name}: {self.status} {self.duration_ms}ms"
        return f"{message}\n  ERROR: {self.error}" if self.error else message


def run_step(idx: int, name: str, fn: Callable[[], None]) -> StepResult:
    started = time.monotonic()
    try:
        fn()
    except Exception as error:
        return StepResult(idx, name, "FAIL", int((time.monotonic() - started) * 1000), str(error))
    return StepResult(idx, name, "PASS", int((time.monotonic() - started) * 1000))


async def _load_runtime_config():
    """Read an existing active snapshot without bootstrap DDL or writes."""
    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    snapshot = await repository.get_existing()
    if snapshot.setup_required:
        raise RuntimeError("runtime configuration is not active")
    return type("StagingRuntime", (), {"repository": repository, "snapshot": snapshot})()


async def _check_runtime_health(runtime) -> None:
    """Discover and validate the candidate graph without opening sessions."""
    from cryptotrader.cycle_events import NullCycleEventSink
    from cryptotrader.runtime import _discover_registry_graph
    from cryptotrader.runtime_config.models import validate_runtime_document

    signals, venues, markets = await _discover_registry_graph(
        runtime.snapshot.document,
        NullCycleEventSink(),
        runtime.repository,
    )
    validate_runtime_document(
        runtime.snapshot.document,
        set(signals.installed_ids()),
        set(venues.installed_ids()),
        set(markets.installed_ids()),
    )
    runtime.signal_registry = signals
    runtime.venue_registry = venues
    runtime.market_registry = markets


async def _check_enabled_connections(runtime) -> None:
    """Open and close each enabled venue session without creating orders."""
    for connection in runtime.snapshot.document.execution.connections:
        if not connection.enabled:
            continue
        credentials = None
        if connection.credential_ref is not None:
            state = await runtime.repository.credential_state(connection.credential_ref)
            if not state.configured:
                raise RuntimeError(f"connection {connection.id} credentials are not configured")
            credentials = await runtime.repository.reveal_credentials(connection.credential_ref)
        session = None
        try:
            adapter = runtime.venue_registry.require(connection.adapter_id)
            session = await adapter.connect(connection, credentials)
            if session.capabilities is None:
                raise RuntimeError(f"connection {connection.id} has no capabilities")
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"connection {connection.id} is unhealthy") from error
        finally:
            if session is not None:
                try:
                    await session.close()
                except Exception as error:
                    raise RuntimeError(f"connection {connection.id} did not close cleanly") from error


async def _run_gate() -> list[StepResult]:
    runtime = None
    steps: tuple[tuple[str, Callable[..., Awaitable[None]]], ...] = (
        ("database schema and config revision", _load_runtime_config),
        ("runtime health", _check_runtime_health),
        ("enabled connection health", _check_enabled_connections),
    )
    results: list[StepResult] = []
    try:
        for index, (name, check) in enumerate(steps, start=1):
            started = time.monotonic()
            try:
                if index == 1:
                    runtime = await check()
                else:
                    await check(runtime)
            except Exception as error:
                result = StepResult(index, name, "FAIL", int((time.monotonic() - started) * 1000), str(error))
            else:
                result = StepResult(index, name, "PASS", int((time.monotonic() - started) * 1000))
            results.append(result)
            print(result.fmt(), flush=True)
            if result.status == "FAIL":
                break
    finally:
        if runtime is not None:
            await runtime.close()
    return results


def main() -> int:
    """Run the readonly gate. Runtime bootstrap accepts only DATABASE_URL and CONFIG_MASTER_KEY."""
    results = asyncio.run(_run_gate())
    failed = [result for result in results if result.status == "FAIL"]
    if failed:
        print(f"\n[summary] {len(failed)}/{len(results)} step(s) FAILED: {failed[0].name}", flush=True)
        return 1
    print(f"\n[summary] All {len(results)} steps PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
