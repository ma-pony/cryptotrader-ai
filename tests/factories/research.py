"""Research HTTP fixture: real Store/manager, explicit SQLite, no models or market network."""

from types import SimpleNamespace

import httpx
import pytest

from cryptotrader.backtest.result import BacktestResult
from cryptotrader.backtest.service import BacktestService
from cryptotrader.db import dispose_engine
from cryptotrader.decision.read_service import DecisionReadService
from cryptotrader.tasks import BackgroundTaskManager
from tests.factories.backtest import replay_config


class ResearchRepository:
    def __init__(self, url):
        self.database_url = url
        self.snapshot = replay_config()

    async def get_existing(self):
        return self.snapshot

    async def reveal_token(self, ref):
        raise AssertionError("offline research must never read credentials")


async def no_registries(_snapshot):
    return None, None


class EmptyEngine:
    def __init__(self, **kwargs):
        self.progress = kwargs["progress_callback"]

    async def run(self):
        await self.progress(0.5)
        return BacktestResult()


@pytest.fixture
async def research(monkeypatch, tmp_path, research_offline):
    from api.main import app
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    url = f"sqlite+aiosqlite:///{tmp_path}/research.sqlite"
    await migrate_workbench_schema(url)
    manager = BackgroundTaskManager()
    repository = ResearchRepository(url)
    service = BacktestService(
        repository=repository, task_manager=manager, engine_factory=EmptyEngine, registry_provider=no_registries
    )
    runtime = SimpleNamespace(
        backtest_service=service,
        task_manager=manager,
        repository=repository,
        snapshot=repository.snapshot,
        read_service=DecisionReadService(service.store.journal),
    )
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, service
    await manager.shutdown()
    await dispose_engine(url)


def research_payload(**changes):
    return {
        "pair": "BTC/USDT",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "interval": "1h",
        "initial_equity": "10000",
        "fee_rate": "0.001",
        "slippage_bps": "0",
        "funding_assumption": "available_only",
    } | changes
