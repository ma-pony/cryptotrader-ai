"""One-time venue credential cutover with an explicit encrypted backup destination."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from cryptotrader.db import get_async_session
from cryptotrader.runtime_config.repository import (
    API_ACCESS_CREDENTIAL_REF,
    LLM_GATEWAY_CREDENTIAL_REF,
    NEWS_PROVIDER_CREDENTIAL_REF,
    _RuntimeConfigRow,
    _RuntimeCredentialRow,
)
from cryptotrader.runtime_config.secrets import CredentialPayload, CredentialVault


def workbench_metadata():
    """Return every SQLAlchemy schema owned by the Workbench runtime."""

    from cryptotrader.accounts.store import Base as AccountBase
    from cryptotrader.alerts.store import Base as AlertBase
    from cryptotrader.backtest.store import BacktestBase
    from cryptotrader.hitl.store import _BookBase
    from cryptotrader.journal.store import _MultiVenueBase
    from cryptotrader.portfolio.manager import _pm_models
    from cryptotrader.risk.book_state import BookRiskRow  # noqa: F401 - registers the shared account table.
    from cryptotrader.runtime_config.repository import _Base as RuntimeBase
    from cryptotrader.signals.evaluation_store import EvaluationBase
    from cryptotrader.triggers.models import Base as TriggerBase

    return (
        RuntimeBase.metadata,
        AccountBase.metadata,
        _MultiVenueBase.metadata,
        _BookBase.metadata,
        EvaluationBase.metadata,
        BacktestBase.metadata,
        AlertBase.metadata,
        TriggerBase.metadata,
        _pm_models()[0].metadata,
    )


async def require_workbench_schema(database_url: str) -> None:
    """Read-only preflight for the complete explicitly migrated runtime schema."""

    from cryptotrader.migrations.schema import require_tables

    table_names = {name for metadata in workbench_metadata() for name in metadata.tables}
    await require_tables(database_url, table_names)


async def migrate_workbench_schema(database_url: str) -> None:
    """Explicitly install the current Workbench schema without loading Runtime."""

    from cryptotrader.db import get_engine

    if not database_url:
        raise ValueError("explicit database URL is required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        for schema in workbench_metadata():
            await connection.run_sync(schema.create_all)


async def migrate_venue_credentials(database_url: str, vault: CredentialVault, backup_path: Path) -> int:
    """Re-encrypt linked legacy payloads; preserve config, references, timestamps and tokens.

    Both the database and exclusive backup file must be supplied explicitly.
    No schema creation, runtime initialization or environment defaults occur here.
    """
    session = await get_async_session(database_url)
    async with session, session.begin():
        config = await session.get(_RuntimeConfigRow, "global")
        if config is None:
            raise ValueError("runtime configuration is missing")
        references = {item.get("credential_ref") for item in config.document["execution"]["connections"]}
        references -= {None, API_ACCESS_CREDENTIAL_REF, LLM_GATEWAY_CREDENTIAL_REF, NEWS_PROVIDER_CREDENTIAL_REF}
        rows = (
            await session.scalars(
                select(_RuntimeCredentialRow).where(_RuntimeCredentialRow.credential_ref.in_(references))
            )
        ).all()
        converted = []
        backups = []
        for row in rows:
            envelope = bytes(row.encrypted_payload)
            if envelope[:1] == vault.VERSION:
                vault.open(row.credential_ref, envelope)
                continue
            if envelope[:1] != b"\x01" or len(envelope) < 30:
                raise ValueError("unsupported legacy credential envelope")
            plaintext = vault._cipher.decrypt(envelope[1:13], envelope[13:], row.credential_ref.encode())
            try:
                legacy = json.loads(plaintext)
                if (
                    not isinstance(legacy, dict)
                    or not {"api_key", "secret"} <= legacy.keys()
                    or not legacy.keys() <= {"api_key", "secret", "passphrase"}
                ):
                    raise ValueError
                values = {key: value for key, value in legacy.items() if value is not None}
                if not all(isinstance(value, str) for value in values.values()):
                    raise ValueError
                payload = CredentialPayload(values=values)
            except (ValueError, TypeError):
                raise ValueError("invalid legacy credential payload") from None
            converted.append((row, vault.seal(row.credential_ref, payload)))
            backups.append(
                {
                    "credential_ref": row.credential_ref,
                    "encrypted_payload": base64.b64encode(envelope).decode(),
                    "updated_at": row.updated_at.isoformat(),
                }
            )
        if not converted:
            return 0
        descriptor = os.open(Path(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as backup:
            json.dump({"format": "workbench-venue-credentials-v1", "credentials": backups}, backup)
            backup.flush()
            os.fsync(backup.fileno())
        for row, encrypted in converted:
            row.encrypted_payload = encrypted
        return len(converted)


async def migrate_decision_records(database_url: str, backup_path: Path) -> int:
    """Explicit one-time journal cutover. Never invoked by Runtime or HTTP.

    Only the old multi-venue producer's known facts are retained. In particular
    no current configuration, current model, curve, origin or end time is used
    to reconstruct historical facts.
    """
    from sqlalchemy import inspect, text

    from cryptotrader.db import get_engine

    if not database_url or backup_path is None:
        raise ValueError("explicit database URL and backup path are required")

    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        columns = await connection.run_sync(lambda sync: inspect(sync).get_columns("multi_venue_cycles"))
        run_column = next((column for column in columns if column["name"] == "run_metadata"), None)
        has_run_metadata = run_column is not None
        schema_requires_cutover = run_column is None or run_column["nullable"] or run_column["default"] is not None
        rows = (await connection.execute(text("SELECT * FROM multi_venue_cycles"))).mappings().all()
        converted = []
        for row in rows:
            upgraded = _upgrade_decision_record(row, has_run_metadata)
            if upgraded is not None:
                converted.append(upgraded)
        if not converted and not schema_requires_cutover:
            return 0
        schema = await connection.run_sync(_decision_backup_schema)
        # Serialize the untouched rows before conversion can mutate any native
        # JSON values returned by PostgreSQL. No DDL/UPDATE precedes fsync.
        backup_payload = json.dumps(
            {
                "format": "workbench-decision-records-v1",
                "dialect": engine.dialect.name,
                "schema": schema,
                "rows": [dict(row) for row in rows],
            },
            default=_decision_backup_value,
        )
        descriptor = os.open(Path(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as backup:
            backup.write(backup_payload)
            backup.flush()
            os.fsync(backup.fileno())
        if engine.dialect.name == "sqlite":
            # Python's sqlite driver does not begin a transaction for DDL;
            # make the table rebuild atomically rollbackable before CREATE/DROP.
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            await _rebuild_sqlite_decision_table(connection, rows, converted, schema)
            return len(converted)
        add_column, statement, drop_default, set_not_null = _decision_cutover_statements(engine.dialect)
        if not has_run_metadata:
            await connection.execute(add_column)
        for decision_id, signals, fused, target, books, run, requires_attention in converted:
            await connection.execute(
                statement,
                {
                    "id": decision_id,
                    "signals": signals,
                    "fused": fused,
                    "target": target,
                    "books": books,
                    "run": run,
                    "requires_attention": requires_attention,
                },
            )
        await connection.execute(drop_default)
        await connection.execute(set_not_null)
        return len(converted)


async def migrate_backtest_snapshots(database_url: str, backup_path: Path) -> int:
    """Explicitly replace legacy JSON `{}` snapshots with SQL NULL after backup."""
    from sqlalchemy import text

    from cryptotrader.db import get_engine

    if not database_url or backup_path is None:
        raise ValueError("explicit database URL and backup path are required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        rows = (await connection.execute(text("SELECT * FROM backtest_runs"))).mappings().all()
        affected = [row for row in rows if _decision_json(row["config_snapshot"]) == {}]
        if not affected:
            return 0
        backup_payload = json.dumps(
            {
                "format": "workbench-backtest-snapshots-v1",
                "dialect": engine.dialect.name,
                "rows": [dict(row) for row in affected],
            },
            default=_decision_backup_value,
        )
        descriptor = os.open(Path(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as backup:
            backup.write(backup_payload)
            backup.flush()
            os.fsync(backup.fileno())
        for row in affected:
            await connection.execute(
                text("UPDATE backtest_runs SET config_snapshot = NULL WHERE run_id = :run_id"),
                {"run_id": row["run_id"]},
            )
        return len(affected)


def _upgrade_decision_record(row, has_run_metadata):
    from copy import deepcopy
    from types import SimpleNamespace

    from cryptotrader.journal.models import DecisionRun
    from cryptotrader.journal.store import _contains_journal_secret, _multi_venue_record, _run_payload

    payloads = tuple(
        _decision_json(row[name]) for name in ("component_signals", "fused_signal", "target_position", "book_results")
    )
    run_payload = _decision_json(row["run_metadata"]) if has_run_metadata else None
    if _contains_journal_secret(payloads) or any(_encoded_detail_contains_secret(value) for value in payloads):
        raise ValueError("secret field in legacy decision record")
    if run_payload is not None and _encoded_detail_contains_secret(run_payload):
        raise ValueError("secret field in legacy decision record")
    strict_run = False
    existing_requires_attention = bool(row["requires_attention"])
    try:
        if run_payload is not None:
            _multi_venue_record(
                _decision_row(
                    row,
                    payloads,
                    run_payload,
                    SimpleNamespace,
                    requires_attention=existing_requires_attention,
                )
            )
            strict_run = True
    except ValueError as error:
        if str(error) == "secret field":
            raise ValueError("secret field in legacy decision record") from None
    if strict_run:
        return None
    signals, fused, target, books = (deepcopy(value) for value in payloads)
    missing: set[str] = set()
    _upgrade_component_evidence(signals, missing)
    unknown_reconciliation = _upgrade_book_results(books, missing)
    if (
        run_payload is None
        or not isinstance(run_payload, dict)
        or not {
            "version",
            "pair",
            "mode",
            "origin",
            "config_snapshot",
            "finished_at",
            "failure",
            "incomplete_fields",
        }
        <= run_payload.keys()
    ):
        pairs = {book["pair"] for book in books["items"]}
        pair = next(iter(pairs)) if len(pairs) == 1 else None
        missing.update({"origin", "config_snapshot", "finished_at", "failure"})
        if pair is None:
            missing.add("pair")
        run_payload = _run_payload(DecisionRun(pair, "trading", None, {}, None, None, tuple(sorted(missing))))
    else:
        run_payload = deepcopy(run_payload)
        existing_missing = run_payload.setdefault("incomplete_fields", [])
        if type(existing_missing) is not list:
            raise ValueError("invalid legacy decision incomplete_fields")
        existing_missing.extend(item for item in sorted(missing) if item not in existing_missing)
    upgraded = (signals, fused, target, books)
    safe_requires_attention = existing_requires_attention or unknown_reconciliation
    _multi_venue_record(
        _decision_row(
            row,
            upgraded,
            run_payload,
            SimpleNamespace,
            requires_attention=safe_requires_attention,
        )
    )
    return (row["cycle_id"], *upgraded, run_payload, safe_requires_attention)


def _decision_json(value):
    return json.loads(value) if isinstance(value, str) else value


def _encoded_detail_contains_secret(value) -> bool:
    """Recognize keys hidden inside the strict detail mapping's pair encoding."""
    from cryptotrader.journal.store import _is_secret_field

    if isinstance(value, dict):
        if value.get("kind") == "mapping" and isinstance(value.get("items"), list):
            for item in value["items"]:
                if isinstance(item, list) and len(item) == 2:
                    if isinstance(item[0], str) and _is_secret_field(item[0]):
                        return True
                    if _encoded_detail_contains_secret(item[1]):
                        return True
        return any(_encoded_detail_contains_secret(item) for item in value.values())
    if isinstance(value, list):
        return any(_encoded_detail_contains_secret(item) for item in value)
    return False


def _decision_row(row, payloads, run_payload, namespace, *, requires_attention):
    created_at = row["created_at"]
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    signals, fused, target, books = payloads
    return namespace(
        cycle_id=row["cycle_id"],
        config_revision=row["config_revision"],
        market_data_source_id=row["market_data_source_id"],
        component_signals=signals,
        fused_signal=fused,
        target_position=target,
        book_results=books,
        cycle_status=row["cycle_status"],
        execution_status=row["execution_status"],
        requires_attention=requires_attention,
        created_at=created_at,
        run_metadata=run_payload,
    )


def _upgrade_component_evidence(signals, missing):
    for signal in signals["items"]:
        for block in signal.get("blocks", []):
            if isinstance(block, dict) and block.get("kind") == "series" and "evaluation_target" not in block:
                block["evaluation_target"] = None
                missing.add("components.blocks.evaluation_target")
    _migrate_component_evidence(signals, missing)


def _upgrade_book_results(books, missing):
    unknown_reconciliation = False
    for book in books["items"]:
        if "reconciliation_required" not in book:
            book["reconciliation_required"] = None
            unknown_reconciliation = True
            missing.add("books.reconciliation_required")
        for portfolio_name in ("portfolio_before", "portfolio_after"):
            _upgrade_portfolio(book.get(portfolio_name), missing)
        _upgrade_proposal(book.get("proposal"), missing)
        execution = book.get("execution")
        if execution is None:
            continue
        _upgrade_proposal(execution.get("proposal"), missing)
        for result in execution.get("connection_results", []):
            if "quantity_frozen" not in result:
                result["quantity_frozen"] = None
                missing.add("books.execution.quantity_frozen")
            for order in result.get("orders", []):
                _upgrade_order(order, missing)
            _upgrade_protection(result.get("protection"), missing)
            compensation = result.get("compensation") or {}
            _upgrade_order(compensation.get("order"), missing)
            _upgrade_protection(compensation.get("required_protection"), missing)
            final_position = result.get("final_position")
            if final_position is not None:
                for protection in final_position.get("protections", []):
                    _upgrade_protection(protection, missing)
    return unknown_reconciliation


def _upgrade_proposal(proposal, missing):
    if proposal is None:
        return
    risk = proposal.get("risk")
    if isinstance(risk, dict) and "state" not in risk:
        risk["state"] = None
        missing.add("books.risk.state")
    for plan in proposal.get("connection_plans", []):
        capabilities = plan.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        unknown_fields = capabilities.setdefault("unknown_fields", [])
        if type(unknown_fields) is not list:
            raise ValueError("invalid legacy unknown capability fields")
        for key, unknown in (("account_reads", []), ("exit_operations", []), ("history_initial_days", None)):
            if key not in capabilities:
                capabilities[key] = unknown
                if key not in unknown_fields:
                    unknown_fields.append(key)
                missing.add(f"books.execution.capabilities.{key}")


def _upgrade_portfolio(portfolio, missing):
    if portfolio is None:
        return
    for connection in portfolio.get("connections", []):
        snapshot = connection.get("account_snapshot")
        if not isinstance(snapshot, dict):
            continue
        if "valuation_notes" not in snapshot:
            snapshot["valuation_notes"] = ["历史记录未保存估值说明"]
            missing.add("books.portfolio.account_snapshot.valuation_notes")
        currency = (snapshot.get("equity") or {}).get("currency") or "UNKNOWN"
        for order in snapshot.get("orders", []):
            if "client_order_id" not in order:
                order["client_order_id"] = None
                missing.add("books.portfolio.account_snapshot.orders.client_order_id")
            if "remaining_notional" not in order:
                order["remaining_notional"] = {
                    "amount": None,
                    "currency": currency,
                    "unavailable_reason": "历史记录未保存剩余名义价值",
                }
                missing.add("books.portfolio.account_snapshot.orders.remaining_notional")


def _upgrade_order(order, missing):
    if isinstance(order, dict) and "client_order_id" not in order:
        order["client_order_id"] = None
        missing.add("books.execution.orders.client_order_id")


def _upgrade_protection(protection, missing):
    if isinstance(protection, dict) and "actual_order_ids" not in protection:
        # A routing/cancel reference is not proof of a platform order identity.
        protection["actual_order_ids"] = []
        missing.add("books.execution.protection.actual_order_ids")


def _decision_cutover_statements(dialect):
    from sqlalchemy import bindparam, text

    from cryptotrader.journal.store import _MultiVenueCycleRow

    columns = _MultiVenueCycleRow.__table__.c
    run_type = columns.run_metadata.type
    sql_type = run_type.compile(dialect=dialect)
    ddl = text(f"ALTER TABLE multi_venue_cycles ADD COLUMN run_metadata {sql_type}")
    statement = text("""UPDATE multi_venue_cycles SET
        component_signals = :signals,
        fused_signal = :fused,
        target_position = :target,
        book_results = :books,
        run_metadata = :run,
        requires_attention = :requires_attention
        WHERE cycle_id = :id""").bindparams(
        bindparam("signals", type_=columns.component_signals.type),
        bindparam("fused", type_=columns.fused_signal.type),
        bindparam("target", type_=columns.target_position.type),
        bindparam("books", type_=columns.book_results.type),
        bindparam("run", type_=run_type),
        bindparam("id", type_=columns.cycle_id.type),
        bindparam("requires_attention", type_=columns.requires_attention.type),
    )
    drop_default = text("ALTER TABLE multi_venue_cycles ALTER COLUMN run_metadata DROP DEFAULT")
    set_not_null = text("ALTER TABLE multi_venue_cycles ALTER COLUMN run_metadata SET NOT NULL")
    return ddl, statement, drop_default, set_not_null


async def _rebuild_sqlite_decision_table(connection, rows, converted, schema):
    from sqlalchemy import Column, MetaData, Table, text
    from sqlalchemy.schema import CreateTable

    from cryptotrader.journal.store import _MultiVenueCycleRow

    source = await connection.run_sync(
        lambda sync_connection: Table("multi_venue_cycles", MetaData(), autoload_with=sync_connection)
    )
    temporary = source.to_metadata(MetaData(), name="multi_venue_cycles__canonical")
    if "run_metadata" not in temporary.c:
        temporary.append_column(
            Column("run_metadata", _MultiVenueCycleRow.__table__.c.run_metadata.type, nullable=False)
        )
    else:
        temporary.c.run_metadata.nullable = False
        temporary.c.run_metadata.default = None
        temporary.c.run_metadata.server_default = None
    await connection.execute(text(str(CreateTable(temporary).compile(dialect=connection.dialect))))
    upgrades = {item[0]: item for item in converted}
    rebuilt_rows = []
    for raw in rows:
        values = dict(raw)
        if isinstance(values.get("created_at"), str):
            values["created_at"] = datetime.fromisoformat(values["created_at"])
        upgraded = upgrades.get(raw["cycle_id"])
        if upgraded is not None:
            _, signals, fused, target, books, run, requires_attention = upgraded
            values.update(
                component_signals=signals,
                fused_signal=fused,
                target_position=target,
                book_results=books,
                run_metadata=run,
                requires_attention=requires_attention,
            )
        rebuilt_rows.append(values)
    if rebuilt_rows:
        await connection.execute(temporary.insert(), rebuilt_rows)
    await connection.execute(text("DROP TABLE multi_venue_cycles"))
    await connection.execute(text("ALTER TABLE multi_venue_cycles__canonical RENAME TO multi_venue_cycles"))
    for statement in schema["indexes"]:
        await connection.execute(text(statement))
    await _create_sqlite_decision_indexes(connection)


async def _create_sqlite_decision_indexes(connection):
    from sqlalchemy import Index, MetaData, Table

    from cryptotrader.journal.store import _MultiVenueCycleRow

    def create_canonical_indexes(sync_connection):
        final = Table("multi_venue_cycles", MetaData(), autoload_with=sync_connection)
        for source_index in _MultiVenueCycleRow.__table__.indexes:
            if source_index.name in {index.name for index in final.indexes}:
                continue
            Index(source_index.name, *(final.c[column.name] for column in source_index.columns)).create(sync_connection)

    await connection.run_sync(create_canonical_indexes)


def _decision_backup_schema(connection):
    from sqlalchemy import MetaData, Table
    from sqlalchemy.schema import CreateIndex, CreateTable

    table = Table("multi_venue_cycles", MetaData(), autoload_with=connection)
    return {
        "table": str(CreateTable(table).compile(dialect=connection.dialect)),
        "indexes": [str(CreateIndex(index).compile(dialect=connection.dialect)) for index in table.indexes],
    }


def _decision_backup_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported decision backup value type: {type(value).__name__}")


async def migrate_run_controls(database_url: str, backup_path: Path) -> int:
    """Explicit document cutover. Always pause automation; never open a runtime."""
    from copy import deepcopy

    from cryptotrader.runtime_config.models import RuntimeConfigDocument

    if not database_url or not database_url.strip() or backup_path is None:
        raise ValueError("explicit database_url and backup_path are required")
    session = await get_async_session(database_url)
    async with session, session.begin():
        row = await session.get(_RuntimeConfigRow, "global", with_for_update=True)
        if row is None:
            raise ValueError("runtime configuration is missing")
        original = deepcopy(row.document)
        if (
            "system" not in original
            and "pairs" not in original.get("scheduler", {})
            and "hitl_required" not in original.get("signals", {})
        ):
            RuntimeConfigDocument.model_validate(original)
            return 0
        payload = {
            "format": "workbench-run-controls-v1",
            "row": {column.name: getattr(row, column.name) for column in _RuntimeConfigRow.__table__.columns},
        }
        serialized = json.dumps(payload, default=_decision_backup_value)
        descriptor = os.open(Path(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as backup:
            backup.write(serialized)
            backup.flush()
            os.fsync(backup.fileno())
        document = deepcopy(original)
        active = document.pop("system", {}).get("active", False)
        scheduler = document.setdefault("scheduler", {})
        document["execution"]["pairs"] = scheduler.pop("pairs", document["execution"].get("pairs", []))
        scheduler["automation_enabled"] = False
        document["signals"].pop("hitl_required", None)
        if not active:
            for book in document["execution"]["books"]:
                book["enabled"] = False
        converted = RuntimeConfigDocument.model_validate(document)
        row.document = converted.model_dump(mode="json")
        row.revision += 1
        row.apply_status = "pending"
        row.apply_error = None
        return 1


def _migrate_component_evidence(signals, missing):
    from cryptotrader.signals.presentation import RESULT_BLOCKS, EvaluationReference

    for signal in signals["items"]:
        for key, value in {
            "blocks": [],
            "evaluation_reference": None,
            "status": "completed",
            "duration_ms": None,
            "usage": None,
            "cost": None,
        }.items():
            if key not in signal:
                signal[key] = value
                missing.add(f"components.{key}")
        valid_blocks = []
        for block in signal["blocks"]:
            try:
                RESULT_BLOCKS.validate_python((block,))
                valid_blocks.append(block)
            except ValueError:
                # Naive historical timestamps cannot be assigned a timezone.
                missing.add("components.blocks")
        signal["blocks"] = valid_blocks
        if signal["evaluation_reference"] is not None:
            try:
                EvaluationReference.model_validate(signal["evaluation_reference"])
            except ValueError:
                signal["evaluation_reference"] = None
                missing.add("components.evaluation_reference")


async def migrate_account_ledger(database_url: str, backup_path: Path) -> int:
    """Explicit account-ledger cutover. Old membership before migration stays unknown."""
    from datetime import UTC

    from cryptotrader.accounts.store import AccountStore, update_memberships
    from cryptotrader.runtime_config.models import RuntimeConfigDocument

    if not database_url or backup_path is None:
        raise ValueError("explicit database URL and backup path are required")
    store = AccountStore(database_url)
    session = await get_async_session(database_url)
    async with session, session.begin():
        row = await session.get(_RuntimeConfigRow, "global", with_for_update=True)
        if row is None:
            raise ValueError("runtime configuration is missing")
        if "accounts" in row.document:
            return 0
        descriptor = os.open(Path(backup_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as backup:
            json.dump(
                {"format": "workbench-account-ledger-v1", "document": row.document, "revision": row.revision}, backup
            )
            backup.flush()
            os.fsync(backup.fileno())
        await store.ensure_tables()
        document = RuntimeConfigDocument.model_validate(row.document)
        row.document = document.model_dump(mode="json")
        row.revision += 1
        row.apply_status = "pending"
        row.apply_error = None
        await update_memberships(session, document, datetime.now(UTC))
        return 1


async def migrate_book_risk_state(database_url: str) -> None:
    """Explicit new-table migration; no runtime config, peak reset or account access."""
    from cryptotrader.db import get_engine
    from cryptotrader.risk.book_state import BookRiskRow

    if not database_url:
        raise ValueError("explicit database URL is required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: BookRiskRow.__table__.create(sync, checkfirst=True))


async def migrate_account_operations(database_url: str) -> None:
    """Explicit new-table migration only; never open configured accounts or change configuration."""
    from cryptotrader.accounts.store import AccountOperationRow, ArchivedAccountRow
    from cryptotrader.db import get_engine

    if not database_url:
        raise ValueError("explicit database URL is required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        for table in (AccountOperationRow.__table__, ArchivedAccountRow.__table__):
            await connection.run_sync(lambda sync, table=table: table.create(sync, checkfirst=True))


async def migrate_component_evaluations(database_url: str) -> None:
    """Explicit additive migration; no prediction rewrite, source call or configuration read."""
    from cryptotrader.db import get_engine
    from cryptotrader.signals.evaluation_store import EvaluationRow

    if not database_url:
        raise ValueError("explicit database URL is required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: EvaluationRow.__table__.create(sync, checkfirst=True))


async def migrate_alerts(database_url: str) -> None:
    """Explicit additive schema only; never send, scan facts or open venue accounts."""
    from cryptotrader.alerts.store import Base
    from cryptotrader.db import get_engine

    if not database_url:
        raise ValueError("explicit database URL is required")
    engine = await get_engine(database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def import_backtest_session(database_url: str, source: Path) -> str:
    """Explicit one-directory import. Never scan the user's historical home or alter source files."""
    import asyncio
    import hashlib

    from cryptotrader.backtest.models import BacktestParams
    from cryptotrader.backtest.result import BacktestResult
    from cryptotrader.backtest.store import BacktestStore, result_from_payload, result_payload

    source = await asyncio.to_thread(source.resolve, strict=True)
    params_bytes = await asyncio.to_thread((source / "params.json").read_bytes)
    result_bytes = await asyncio.to_thread((source / "result.json").read_bytes)
    identity = "import_" + hashlib.sha256(params_bytes + b"\0" + result_bytes).hexdigest()[:32]
    store = BacktestStore(database_url)
    if await store.get(identity) is not None:
        return identity
    old, result = json.loads(params_bytes), json.loads(result_bytes)
    params = BacktestParams(
        pair=old["pair"],
        start=old["start"],
        end=old["end"],
        name=source.name[:120],
        interval=old.get("interval"),
        initial_equity=old.get("initial_equity", old.get("initial_capital")),
        fee_rate=old.get("fee_rate"),
        slippage_bps=old.get("slippage_bps"),
        funding_assumption=old.get("funding_assumption"),
    )
    missing = [
        f"{key}: 旧文件未保存此实验条件"
        for key in ("interval", "initial_equity", "fee_rate", "slippage_bps", "funding_assumption")
        if getattr(params, key) is None
    ]
    missing.append("config_snapshot: 旧文件没有完整安全配置, 不能复用")
    missing.append("decisions: 旧文件未保存可校验的统一决策编码, 保留源文件, 不重新生成")
    values = result_payload(BacktestResult())
    for key in (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "fees",
        "funding",
        "fills",
        "closed_trades",
        "funding_entries",
        "cost_assumptions",
        "unmodeled_costs",
        "data_coverage",
    ):
        if key in result:
            values[key] = result[key]
    curve = result.get("equity_curve")
    if curve and all(isinstance(point, dict) and "time" in point and "equity" in point for point in curve):
        values["equity_curve"] = curve
    else:
        missing.append("equity_curve: 旧文件未保存带历史时间的权益曲线, 不能重建")
    if "fills" not in result:
        missing.append("fills: 旧文件没有实际成交流水")
    restored = result_from_payload(values)
    await store.create(params, None, incomplete_fields=missing, run_id=identity)
    await store.update(identity, "completed", 1, restored)
    return identity
