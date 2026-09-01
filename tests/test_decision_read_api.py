"""Decision history remains authenticated and independent of trading activation."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI

from api.dependencies import verify_api_key
from api.routes.decisions import router
from cryptotrader.decision.read_service import DecisionReadService
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
from tests.factories.runtime_config import runtime_document
from tests.test_analysis_isolation import NOW


@pytest.mark.asyncio
async def test_paused_decision_history_is_available_without_active_cycle():
    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(2, runtime_document(), NOW),
        cycle=None,
        read_service=DecisionReadService(MultiVenueCycleStore()),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/decisions")
    assert response.status_code == 200
    assert response.json()["items"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("naive_evidence", [False, True])
async def test_explicit_legacy_migration_preserves_id_and_does_not_invent_curves(tmp_path, naive_evidence):
    import json
    import sqlite3

    from cryptotrader.journal.store import _record_payloads
    from cryptotrader.migrations import workbench
    from tests.test_multi_venue_journal import _record

    path = tmp_path / "legacy.db"
    payloads = _record_payloads(_record())
    old_signals = payloads[0]
    for signal in old_signals["items"]:
        for field in ("blocks", "evaluation_reference", "status", "duration_ms", "usage", "cost"):
            del signal[field]
        if naive_evidence:
            signal["blocks"] = [
                {"kind": "text", "title": "原始说明", "body": "保留"},
                {
                    "kind": "series",
                    "title": "旧曲线",
                    "forecast_start": "2026-08-29T02:00:00",
                    "series": [
                        {"name": "预测", "unit": None, "points": [{"time": "2026-08-29T02:00:00", "value": "100"}]}
                    ],
                },
            ]
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE multi_venue_cycles (
            cycle_id TEXT PRIMARY KEY, config_revision INTEGER NOT NULL, market_data_source_id TEXT NOT NULL,
            component_signals JSON NOT NULL, fused_signal JSON NOT NULL, target_position JSON NOT NULL,
            book_results JSON NOT NULL, cycle_status TEXT NOT NULL, execution_status TEXT NOT NULL,
            requires_attention BOOLEAN NOT NULL, created_at DATETIME NOT NULL)""")
        connection.execute("CREATE INDEX legacy_created_at ON multi_venue_cycles (created_at)")
        connection.execute(
            "INSERT INTO multi_venue_cycles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-id",
                9,
                "market-primary",
                *(json.dumps(value) for value in payloads),
                "partial",
                "partial",
                True,
                "2026-08-29 02:00:00",
            ),
        )
    url = f"sqlite+aiosqlite:///{path}"
    backup_path = tmp_path / "legacy-backup.json"
    migrate = workbench.migrate_decision_records
    assert await migrate(url, backup_path) == 1
    assert backup_path.is_file()
    backup_bytes = backup_path.read_bytes()
    backup = json.loads(backup_bytes)
    assert backup["format"] == "workbench-decision-records-v1"
    assert len(backup["rows"]) == 1
    assert backup["rows"][0]["cycle_id"] == "legacy-id"
    assert json.loads(backup["rows"][0]["component_signals"]) == old_signals
    assert "run_metadata" not in backup["schema"]["table"]
    assert "CREATE TABLE multi_venue_cycles" in backup["schema"]["table"]
    assert backup["schema"]["indexes"] == ["CREATE INDEX legacy_created_at ON multi_venue_cycles (created_at)"]
    with sqlite3.connect(tmp_path / "restored.db") as restored:
        restored.execute(backup["schema"]["table"])
        for statement in backup["schema"]["indexes"]:
            restored.execute(statement)
        restored.executemany(
            "INSERT INTO multi_venue_cycles VALUES ("
            ":cycle_id, :config_revision, :market_data_source_id, :component_signals, :fused_signal, :target_position, "
            ":book_results, :cycle_status, :execution_status, :requires_attention, :created_at)",
            backup["rows"],
        )
        restored.row_factory = sqlite3.Row
        assert dict(restored.execute("SELECT * FROM multi_venue_cycles").fetchone()) == backup["rows"][0]
    with sqlite3.connect(path) as connection:
        columns = {column[1]: column for column in connection.execute("PRAGMA table_info(multi_venue_cycles)")}
        assert columns["run_metadata"][3] == 1
        assert columns["run_metadata"][4] is None
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'legacy_created_at'"
        ).fetchone() == ("legacy_created_at",)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO multi_venue_cycles (
                    cycle_id, config_revision, market_data_source_id, component_signals, fused_signal,
                    target_position, book_results, cycle_status, execution_status, requires_attention, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("missing-run", 1, "market", "{}", "{}", "{}", "{}", "failed", "failed", 1, NOW.isoformat()),
            )
    read = DecisionReadService(MultiVenueCycleStore(url))
    record = await read.get("legacy-id")
    assert record.decision_id == "legacy-id"
    assert record.pair == "BTC/USDT:USDT"
    assert record.origin is None
    assert record.finished_at is None
    assert "config_snapshot" in record.incomplete_fields
    assert "components.blocks" in record.incomplete_fields
    if naive_evidence:
        assert all(len(signal.blocks) == 1 and signal.blocks[0].kind == "text" for signal in record.components)
        assert all(signal.blocks[0].body == "保留" for signal in record.components)
    else:
        assert all(signal.blocks == [] for signal in record.components)
    assert await migrate(url, backup_path) == 0
    assert backup_path.read_bytes() == backup_bytes
    assert (await read.list()).total == 1


@pytest.mark.asyncio
async def test_unified_migration_upgrades_nonempty_execution_without_inventing_new_facts(tmp_path):
    import json
    import sqlite3

    from cryptotrader.journal.store import _record_payloads
    from cryptotrader.migrations import workbench
    from tests.test_multi_venue_journal import _record

    path = tmp_path / "task-before.db"
    url = f"sqlite+aiosqlite:///{path}"
    await workbench.migrate_workbench_schema(url)
    record = _record(cycle_id="legacy-execution")
    await MultiVenueCycleStore(url).save(record)
    payloads = list(_record_payloads(record))
    payloads[0]["items"][0]["blocks"] = [
        {
            "kind": "series",
            "title": "原始预测",
            "forecast_start": "2026-08-29T02:00:00+00:00",
            "series": [
                {
                    "name": "预测",
                    "unit": None,
                    "points": [{"time": "2026-08-29T02:00:00+00:00", "value": "100"}],
                }
            ],
        }
    ]
    for book in payloads[3]["items"]:
        book.pop("reconciliation_required")
        proposals = [book["proposal"], book["execution"]["proposal"]]
        for proposal in proposals:
            proposal["risk"].pop("state")
            for plan in proposal["connection_plans"]:
                plan["capabilities"].pop("account_reads")
                plan["capabilities"].pop("exit_operations")
                plan["capabilities"].pop("history_initial_days")
        for result in book["execution"]["connection_results"]:
            result.pop("quantity_frozen")
            if result["protection"] is not None:
                result["protection"].pop("actual_order_ids")
            if result["final_position"] is not None:
                for protection in result["final_position"]["protections"]:
                    protection.pop("actual_order_ids")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE multi_venue_cycles SET component_signals = ?, book_results = ? WHERE cycle_id = ?",
            (json.dumps(payloads[0]), json.dumps(payloads[3]), record.cycle_id),
        )
    backup_path = tmp_path / "task-before-backup.json"
    assert await workbench.migrate_decision_records(url, backup_path) == 1
    backup = json.loads(backup_path.read_bytes())
    raw_books = json.loads(backup["rows"][0]["book_results"])
    assert "reconciliation_required" not in raw_books["items"][0]

    read = DecisionReadService(MultiVenueCycleStore(url))
    migrated = await read.get(record.cycle_id)
    assert migrated.decision_id == record.cycle_id
    assert [component.component_id for component in migrated.components] == ["kronos", "llm_committee"]
    assert migrated.components[0].blocks[0].evaluation_target is None
    assert migrated.books[0].execution.status == "partial"
    execution = migrated.books[0].connections[0].execution
    assert execution.protection.protection_ids == ["protection-sim-first"]
    assert execution.protection.actual_order_ids == []
    assert migrated.books[0].reconciliation_required is None
    assert execution.quantity_frozen is None
    assert (await MultiVenueCycleStore(url).get(record.cycle_id)).requires_attention is True
    capabilities = migrated.books[0].connections[0].plan.capabilities
    assert capabilities.unknown_fields == ["account_reads", "exit_operations", "history_initial_days"]
    assert "books.execution.protection.actual_order_ids" in migrated.incomplete_fields
    assert "books.reconciliation_required" in migrated.incomplete_fields

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(2, runtime_document(), NOW),
        cycle=None,
        read_service=read,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/decisions/{record.cycle_id}")
    assert response.status_code == 200
    assert response.json()["decision_id"] == record.cycle_id
    assert response.json()["books"][0]["connections"][0]["execution"]["protection"]["protection_ids"] == [
        "protection-sim-first"
    ]
    assert response.json()["books"][0]["reconciliation_required"] is None
    assert response.json()["books"][0]["connections"][0]["execution"]["quantity_frozen"] is None
    assert response.json()["books"][0]["connections"][0]["plan"]["capabilities"]["unknown_fields"] == [
        "account_reads",
        "exit_operations",
        "history_initial_days",
    ]
    assert await workbench.migrate_decision_records(url, backup_path) == 0


@pytest.mark.asyncio
async def test_legacy_unknown_reconciliation_forces_attention_and_remains_readable(tmp_path):
    import json
    import sqlite3

    from cryptotrader.journal.store import _record_payloads
    from cryptotrader.migrations import workbench
    from tests.test_multi_venue_journal import _book_cycle, _record

    path = tmp_path / "legacy-unknown-reconciliation.db"
    url = f"sqlite+aiosqlite:///{path}"
    await workbench.migrate_workbench_schema(url)
    completed_book = _book_cycle(status="completed")
    original = _record(
        cycle_id="legacy-unknown-reconciliation",
        book_results=(completed_book,),
        cycle_status="completed",
        execution_status="completed",
        requires_attention=False,
    )
    store = MultiVenueCycleStore(url)
    await store.save(original)
    books = _record_payloads(original)[3]
    books["items"][0].pop("reconciliation_required")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE multi_venue_cycles SET book_results = ?, requires_attention = ? WHERE cycle_id = ?",
            (json.dumps(books), False, original.cycle_id),
        )

    backup_path = tmp_path / "legacy-unknown-reconciliation-backup.json"
    assert await workbench.migrate_decision_records(url, backup_path) == 1

    stored = await store.get(original.cycle_id)
    assert stored.book_results[0].reconciliation_required is None
    assert stored.requires_attention is True

    read = DecisionReadService(store)
    decision = await read.get(original.cycle_id)
    assert decision.books[0].reconciliation_required is None

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(2, runtime_document(), NOW),
        cycle=None,
        read_service=read,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/decisions/{original.cycle_id}")
    assert response.status_code == 200
    assert response.json()["books"][0]["reconciliation_required"] is None


@pytest.mark.asyncio
async def test_empty_legacy_decision_table_is_backed_up_and_cut_over_once(tmp_path):
    import json
    import sqlite3

    from cryptotrader.migrations import workbench

    path = tmp_path / "empty-legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE multi_venue_cycles (
            cycle_id TEXT PRIMARY KEY, config_revision INTEGER NOT NULL, market_data_source_id TEXT NOT NULL,
            component_signals JSON NOT NULL, fused_signal JSON NOT NULL, target_position JSON NOT NULL,
            book_results JSON NOT NULL, cycle_status TEXT NOT NULL, execution_status TEXT NOT NULL,
            requires_attention BOOLEAN NOT NULL, created_at DATETIME NOT NULL)""")
        connection.execute("CREATE INDEX empty_legacy_created_at ON multi_venue_cycles (created_at)")
    url = f"sqlite+aiosqlite:///{path}"
    backup_path = tmp_path / "empty-backup.json"
    assert await workbench.migrate_decision_records(url, backup_path) == 0
    backup_bytes = backup_path.read_bytes()
    backup = json.loads(backup_bytes)
    assert backup["rows"] == []
    assert "run_metadata" not in backup["schema"]["table"]
    with sqlite3.connect(path) as connection:
        columns = {column[1]: column for column in connection.execute("PRAGMA table_info(multi_venue_cycles)")}
        assert columns["run_metadata"][3] == 1
        assert columns["run_metadata"][4] is None
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'empty_legacy_created_at'"
        ).fetchone() == ("empty_legacy_created_at",)
    assert await workbench.migrate_decision_records(url, backup_path) == 0
    assert backup_path.read_bytes() == backup_bytes


@pytest.mark.asyncio
async def test_decision_cutover_failure_rolls_back_database_but_keeps_fsynced_backup(tmp_path, monkeypatch):
    import json
    import sqlite3

    from cryptotrader.migrations import workbench

    path = tmp_path / "rollback-legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE multi_venue_cycles (
            cycle_id TEXT PRIMARY KEY, config_revision INTEGER NOT NULL, market_data_source_id TEXT NOT NULL,
            component_signals JSON NOT NULL, fused_signal JSON NOT NULL, target_position JSON NOT NULL,
            book_results JSON NOT NULL, cycle_status TEXT NOT NULL, execution_status TEXT NOT NULL,
            requires_attention BOOLEAN NOT NULL, created_at DATETIME NOT NULL)""")

    async def fail_after_rebuild(*_args):
        raise RuntimeError("cutover failed")

    monkeypatch.setattr(workbench, "_create_sqlite_decision_indexes", fail_after_rebuild)
    backup_path = tmp_path / "rollback-backup.json"
    with pytest.raises(RuntimeError, match="cutover failed"):
        await workbench.migrate_decision_records(f"sqlite+aiosqlite:///{path}", backup_path)
    assert json.loads(backup_path.read_bytes())["rows"] == []
    with sqlite3.connect(path) as connection:
        assert "run_metadata" not in {
            column[1] for column in connection.execute("PRAGMA table_info(multi_venue_cycles)")
        }


@pytest.mark.asyncio
async def test_unified_migration_rejects_encoded_secrets_before_creating_backup(tmp_path):
    import json
    import sqlite3

    from cryptotrader.journal.store import _record_payloads
    from cryptotrader.migrations import workbench
    from tests.test_multi_venue_journal import _record

    path = tmp_path / "legacy-secret.db"
    url = f"sqlite+aiosqlite:///{path}"
    await workbench.migrate_workbench_schema(url)
    record = _record(cycle_id="legacy-secret")
    await MultiVenueCycleStore(url).save(record)
    signals = _record_payloads(record)[0]
    signals["items"][0]["details"]["items"].append(["api_key", {"kind": "primitive", "value": "must-not-expand"}])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE multi_venue_cycles SET component_signals = ? WHERE cycle_id = ?",
            (json.dumps(signals), record.cycle_id),
        )
    backup = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="secret field"):
        await workbench.migrate_decision_records(url, backup)
    assert not backup.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["existing", "missing_parent", "fsync"])
async def test_decision_backup_failure_leaves_legacy_schema_and_rows_untouched(tmp_path, monkeypatch, failure):
    import json
    import sqlite3

    from cryptotrader.journal.store import _record_payloads
    from cryptotrader.migrations import workbench
    from tests.test_multi_venue_journal import _record

    path = tmp_path / "backup-failure.db"
    payloads = _record_payloads(_record(cycle_id="unchanged-id"))
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE multi_venue_cycles (
            cycle_id TEXT PRIMARY KEY, config_revision INTEGER NOT NULL, market_data_source_id TEXT NOT NULL,
            component_signals JSON NOT NULL, fused_signal JSON NOT NULL, target_position JSON NOT NULL,
            book_results JSON NOT NULL, cycle_status TEXT NOT NULL, execution_status TEXT NOT NULL,
            requires_attention BOOLEAN NOT NULL, created_at DATETIME NOT NULL)""")
        connection.execute(
            "INSERT INTO multi_venue_cycles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "unchanged-id",
                9,
                "market-primary",
                *(json.dumps(value) for value in payloads),
                "partial",
                "partial",
                True,
                "2026-08-29 02:00:00",
            ),
        )
        before = tuple(connection.iterdump())
    backup = tmp_path / "backup.json"
    if failure == "existing":
        backup.write_bytes(b"do-not-overwrite")
    elif failure == "missing_parent":
        backup = tmp_path / "missing" / "backup.json"
    else:

        def failed_fsync(_descriptor):
            raise OSError("fixture backup fsync failed")

        monkeypatch.setattr(workbench.os, "fsync", failed_fsync)
    with pytest.raises(OSError):
        await workbench.migrate_decision_records(f"sqlite+aiosqlite:///{path}", backup)
    with sqlite3.connect(path) as connection:
        assert tuple(connection.iterdump()) == before
    if failure == "existing":
        assert backup.read_bytes() == b"do-not-overwrite"


@pytest.mark.asyncio
async def test_list_filters_before_pagination_and_preserves_saved_blocks():
    from dataclasses import replace
    from datetime import timedelta

    from tests.test_multi_venue_journal import _record

    journal = MultiVenueCycleStore()
    original = _record()
    for index in range(3):
        record = replace(
            original,
            cycle_id=f"record-{index}",
            created_at=NOW + timedelta(minutes=index),
            run=replace(original.run, origin="scheduled" if index == 1 else "manual"),
        )
        await journal.save(record)
    service = DecisionReadService(journal)
    page = await service.list(
        pair="BTC/USDT:USDT",
        mode="trading",
        origin="manual",
        revision=9,
        started_at=NOW,
        ended_at=NOW + timedelta(minutes=3),
        component_id="kronos",
        limit=1,
        offset=1,
    )
    assert page.total == 2
    assert page.items[0].decision_id == "record-0"
    assert page.has_next is False
    assert (await service.list(component_id="absent")).total == 0


@pytest.mark.parametrize(
    ("dialect_name", "expected_type"),
    [("postgresql", "JSONB"), ("postgresql_asyncpg", "JSONB")],
)
def test_decision_migration_ddl_and_bind_types_match_journal_cas(dialect_name, expected_type):
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.dialects.postgresql import asyncpg

    from cryptotrader.journal.store import _MultiVenueCycleRow
    from cryptotrader.migrations.workbench import _decision_cutover_statements

    dialect = {"postgresql": postgresql.dialect, "postgresql_asyncpg": asyncpg.dialect}[dialect_name]()
    ddl, statement, drop_default, set_not_null = _decision_cutover_statements(dialect)
    add_sql = str(ddl.compile(dialect=dialect))
    assert f"run_metadata {expected_type}" in add_sql
    assert "NOT NULL" not in add_sql
    assert "DEFAULT" not in add_sql
    assert str(drop_default.compile(dialect=dialect)).endswith("run_metadata DROP DEFAULT")
    assert str(set_not_null.compile(dialect=dialect)).endswith("run_metadata SET NOT NULL")
    compiled = statement.compile(dialect=dialect)
    if dialect_name == "postgresql_asyncpg":
        assert str(compiled).count("::JSONB") == 5
    bindings = compiled.binds
    for parameter, column in (("run", "run_metadata"), ("signals", "component_signals")):
        bound_type = bindings[parameter].type.dialect_impl(dialect)
        column_type = _MultiVenueCycleRow.__table__.c[column].type.dialect_impl(dialect)
        assert bound_type.compile(dialect=dialect) == column_type.compile(dialect=dialect) == expected_type
        processor = bound_type.bind_processor(dialect)
        assert processor({"version": 1}) == '{"version": 1}'


@pytest.mark.asyncio
async def test_inactive_decision_history_still_requires_configured_api_authentication():
    from unittest.mock import AsyncMock

    from cryptotrader.runtime_config.models import SecurityConfig

    app = FastAPI()
    app.include_router(router, dependencies=[Depends(verify_api_key)])
    document = runtime_document(security=SecurityConfig(enabled=True))
    app.state.runtime = SimpleNamespace(
        snapshot=RuntimeConfigSnapshot(2, document, NOW),
        cycle=None,
        read_service=DecisionReadService(MultiVenueCycleStore()),
        repository=SimpleNamespace(reveal_token=AsyncMock(return_value=SimpleNamespace(token="fixture-access"))),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/decisions")).status_code == 401
        assert (await client.get("/api/decisions", headers={"X-API-Key": "fixture-access"})).status_code == 200
