"""数据库配置 Runtime 是所有生产入口共享的唯一装配边界。"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module

import pytest

from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
from tests.factories.runtime_config import runtime_document


class _Repository:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    async def get_or_create(self):
        self.calls += 1
        return self.snapshot


class _Registry:
    def __init__(self, installed=()) -> None:
        self._installed = frozenset(installed)

    def installed_ids(self):
        return self._installed


@pytest.mark.asyncio
async def test_setup_required_runtime_opens_no_venue_sessions_and_has_no_cycle():
    runtime_module = import_module("cryptotrader.runtime")
    snapshot = RuntimeConfigSnapshot(0, runtime_document(), datetime(2026, 8, 29, tzinfo=UTC))
    repository = _Repository(snapshot)

    runtime = await runtime_module.build_runtime(
        repository=repository,
        snapshot=snapshot,
        signal_registry=_Registry({"kronos", "llm_committee"}),
        venue_registry=_Registry({"paper", "okx", "bybit"}),
        market_registry=_Registry({"default"}),
    )

    assert repository.calls == 0
    assert runtime.snapshot is snapshot
    assert runtime.repository is repository
    assert runtime.cycle is None
    assert runtime.sessions == {}
    await runtime.close()
    await runtime.close()
