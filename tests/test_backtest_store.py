"""Durable research contract, exercised only against explicit temporary SQLite."""

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from cryptotrader.backtest.comparison import compare_runs
from cryptotrader.backtest.models import BacktestParams
from cryptotrader.backtest.result import BacktestResult, EquityPoint
from cryptotrader.backtest.store import BacktestStore
from cryptotrader.db import dispose_engine

PARAMS = BacktestParams(pair="BTC/USDT", start="2024-01-01", end="2024-01-02")


async def _database_url(path) -> str:
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    database_url = f"sqlite+aiosqlite:///{path}"
    await migrate_workbench_schema(database_url)
    return database_url


@pytest.mark.asyncio
async def test_unnamed_result_survives_disposing_database_and_store(tmp_path):
    url = await _database_url(tmp_path / "research.sqlite")
    store = BacktestStore(url)
    result = BacktestResult(equity_curve=[EquityPoint(datetime(2024, 1, 1, tzinfo=UTC), Decimal("10000.0123"))])
    run_id = await store.create(PARAMS, {"revision": 7, "risk": {"position": {"max_single_pct": 0.4}}})
    await store.update(run_id, "completed", 1, result)
    del store
    await dispose_engine(url)
    restored = await BacktestStore(url).get(run_id)
    assert restored.params.name is None
    assert restored.result.equity_curve == result.equity_curve
    assert restored.result.win_rate is None
    assert restored.config_snapshot["risk"]["position"]["max_single_pct"] == 0.4
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_startup_marks_only_queued_running_interrupted(tmp_path):
    url = await _database_url(tmp_path / "recovery.sqlite")
    store = BacktestStore(url)
    ids = [await store.create(PARAMS, {"revision": 1}) for _ in range(3)]
    await store.update(ids[1], "running", 0.4)
    await store.update(ids[2], "canceled", 0.2)
    await BacktestStore(url).recover_interrupted()
    assert [(await store.get(i)).status for i in ids] == ["interrupted", "interrupted", "canceled"]
    assert (await store.get(ids[1])).progress == 0.4
    await store.update(ids[1], "running", 0.9)
    assert (await store.get(ids[1])).status == "interrupted"
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_legacy_empty_backtest_snapshot_requires_explicit_cutover(tmp_path):
    import json
    import sqlite3

    from cryptotrader.migrations.workbench import migrate_backtest_snapshots

    path = tmp_path / "legacy-snapshot.sqlite"
    url = await _database_url(path)
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, {"revision": 1})
    await dispose_engine(url)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE backtest_runs SET config_snapshot = '{}' WHERE run_id = ?", (run_id,))
    with pytest.raises(ValueError, match="migrate_backtest_snapshots"):
        await store.get(run_id)
    backup_path = tmp_path / "backtest-snapshots.json"
    assert await migrate_backtest_snapshots(url, backup_path) == 1
    backup_bytes = backup_path.read_bytes()
    backup = json.loads(backup_bytes)
    assert backup["format"] == "workbench-backtest-snapshots-v1"
    assert backup["rows"][0]["run_id"] == run_id
    assert backup["rows"][0]["config_snapshot"] in ({}, "{}")
    assert (await store.get(run_id)).config_snapshot is None
    assert await migrate_backtest_snapshots(url, backup_path) == 0
    assert backup_path.read_bytes() == backup_bytes
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_backtest_snapshot_backup_fsync_failure_leaves_legacy_row_unchanged(tmp_path, monkeypatch):
    import sqlite3

    from cryptotrader.migrations import workbench

    path = tmp_path / "legacy-snapshot-fsync.sqlite"
    url = await _database_url(path)
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, {"revision": 1})
    await dispose_engine(url)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE backtest_runs SET config_snapshot = '{}' WHERE run_id = ?", (run_id,))

    def fail_fsync(_descriptor):
        raise OSError("fixture fsync failed")

    monkeypatch.setattr(workbench.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fixture fsync failed"):
        await workbench.migrate_backtest_snapshots(url, tmp_path / "failed-backup.json")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT config_snapshot FROM backtest_runs WHERE run_id = ?", (run_id,)
        ).fetchone() == ("{}",)
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_http_never_emits_uncutover_empty_snapshot(tmp_path):
    import sqlite3
    from types import SimpleNamespace

    from fastapi import FastAPI

    from api.routes.backtest import router

    path = tmp_path / "legacy-snapshot-http.sqlite"
    url = await _database_url(path)
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, {"revision": 1})
    await dispose_engine(url)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE backtest_runs SET config_snapshot = '{}' WHERE run_id = ?", (run_id,))
    app = FastAPI()
    app.include_router(router)
    app.state.runtime = SimpleNamespace(backtest_service=SimpleNamespace(store=store))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/backtest/runs/{run_id}")
    assert response.status_code == 409
    assert "尚未显式迁移" in response.json()["detail"]
    assert "config_snapshot" not in response.text
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_comparison_exposes_cost_and_configuration_differences(tmp_path):
    url = await _database_url(tmp_path / "comparison.sqlite")
    store = BacktestStore(url)
    a = await store.create(PARAMS, {"risk": {"max": 0.5}})
    b = await store.create(PARAMS.model_copy(update={"fee_rate": Decimal("0.002")}), {"risk": {"max": 0.3}})
    comparison = compare_runs(await store.get(a), await store.get(b))
    assert comparison.comparable is False
    assert comparison.condition_differences["fee_rate"] == {"left": "0.001", "right": "0.002"}
    assert comparison.configuration_differences["risk"] == {"left": {"max": 0.5}, "right": {"max": 0.3}}
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_canonical_history_entry_is_available_without_running_models(monkeypatch):
    from api.main import app

    monkeypatch.setattr("api.main._get_redis_for_rate_limit", lambda: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/backtest/runs")
    # Before any runtime is available this route should report unavailable, not 404.
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_failed_canonical_publication_rolls_back_terminal_result(tmp_path):
    url = await _database_url(tmp_path / "atomic.sqlite")
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, {"revision": 1})
    result = BacktestResult(decision_ids=["missing-original-record"])
    with pytest.raises(ValueError, match="original journal records"):
        await store.update(run_id, "completed", 1, result)
    restored = await store.get(run_id)
    assert restored.status == "queued"
    assert restored.result is None
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_result_and_evidence_reject_secret_fields_before_persistence(tmp_path):
    url = await _database_url(tmp_path / "secret.sqlite")
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, {"revision": 1})
    with pytest.raises(ValueError, match="secret field"):
        await store.update(run_id, "completed", 1, BacktestResult(data_coverage={"api_key": "do-not-store"}))
    with pytest.raises(ValueError, match="secret field"):
        await store.update(run_id, "failed", 0, model_evidence=[{"token": "do-not-store"}])
    assert (await store.get(run_id)).status == "queued"
    await dispose_engine(url)


@pytest.mark.asyncio
async def test_missing_snapshot_is_null_and_empty_object_is_rejected(tmp_path):
    url = await _database_url(tmp_path / "missing-snapshot.sqlite")
    store = BacktestStore(url)
    run_id = await store.create(PARAMS, None, incomplete_fields=("config_snapshot",))
    assert (await store.get(run_id)).config_snapshot is None
    with pytest.raises(ValueError, match="empty research snapshot"):
        await store.create(PARAMS, {})
    await dispose_engine(url)
