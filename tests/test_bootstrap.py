"""Runtime active/setup 两条装配路径。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cryptotrader.runtime import build_runtime
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SystemConfig
from tests.factories.runtime_config import active_document


class _Repository:
    database_url = None

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def get_or_create(self):
        self.calls += 1
        return self.snapshot


class _Session:
    connection_id = "paper-local"

    def __init__(self) -> None:
        self.closed = 0

    async def close(self):
        self.closed += 1


class _Adapter:
    adapter_id = "paper"

    def __init__(self, session) -> None:
        self.session = session
        self.connect_calls = 0

    async def connect(self, connection, credentials):
        self.connect_calls += 1
        assert credentials is None
        return self.session


class _VenueRegistry:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def installed_ids(self):
        return frozenset({"paper"})

    def require(self, adapter_id):
        assert adapter_id == "paper"
        return self.adapter


class _MarketSource:
    id = "default"


class _MarketRegistry:
    source = _MarketSource()

    def installed_ids(self):
        return frozenset({"default"})

    def require(self, source_id):
        assert source_id == "default"
        return self.source


class _SignalRegistry:
    def installed_ids(self):
        return frozenset({"kronos", "llm_committee"})


@pytest.mark.asyncio
async def test_active_runtime_opens_enabled_sessions_and_builds_one_cycle():
    snapshot = RuntimeConfigSnapshot(3, active_document(), datetime.now(UTC))
    repository = _Repository(snapshot)
    session = _Session()
    adapter = _Adapter(session)

    runtime = await build_runtime(
        repository=repository,
        signal_registry=_SignalRegistry(),
        venue_registry=_VenueRegistry(adapter),
        market_registry=_MarketRegistry(),
    )

    assert repository.calls == 1
    assert adapter.connect_calls == 1
    assert runtime.cycle is not None
    assert runtime.cycle.repository is repository
    assert runtime.sessions == {"paper-local": session}
    await runtime.close()
    await runtime.close()
    assert session.closed == 1


@pytest.mark.asyncio
async def test_injected_snapshot_is_not_reloaded_during_setup_runtime_build():
    document = active_document().model_copy(update={"system": SystemConfig(active=False)})
    snapshot = RuntimeConfigSnapshot(4, document, datetime.now(UTC))
    repository = _Repository(snapshot)

    runtime = await build_runtime(
        repository=repository,
        snapshot=snapshot,
        signal_registry=_SignalRegistry(),
        venue_registry=_VenueRegistry(_Adapter(_Session())),
        market_registry=_MarketRegistry(),
    )

    assert repository.calls == 0
    assert runtime.cycle is None
    assert runtime.sessions == {}
