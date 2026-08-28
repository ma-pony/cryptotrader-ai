"""多平台 Journal 只持久化严格、脱敏的领域 DTO。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.models import BookExecutionResult, ConnectionExecutionResult
from cryptotrader.pair import Pair
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal
from tests.test_execution_coordinator import _proposal, _result


def _book_result(*, requires_attention: bool = True) -> BookExecutionResult:
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")
    failed = ConnectionExecutionResult.failed(
        proposal.connection_plans[1],
        "place_order",
        execution_quote=proposal.connection_plans[1].quote,
        requires_attention=requires_attention,
        trace=("pre_read", "place_order"),
    )
    return BookExecutionResult(proposal, (completed, failed), "partial", requires_attention)


def _record(*, cycle_id: str = "cycle-1", details=None):
    from cryptotrader.journal.models import MultiVenueCycleRecord

    return MultiVenueCycleRecord(
        cycle_id=cycle_id,
        config_revision=9,
        market_data_source_id="market-primary",
        component_signals=(
            ComponentSignal(
                "kronos",
                "long",
                0.8,
                "model evidence",
                details
                if details is not None
                else {
                    "window": 512,
                    "threshold": Decimal("0.10"),
                    "as_of": datetime(2026, 8, 29, 1, tzinfo=UTC),
                    "pair": Pair.parse("BTC/USDT:USDT"),
                    "direction": "long",
                },
            ),
            ComponentSignal("llm_committee", "short", 0.2, "committee evidence"),
        ),
        fused_signal=FusedSignal(
            0.3,
            (
                ComponentContribution("kronos", 0.5, 0.8, 0.4),
                ComponentContribution("llm_committee", 0.5, -0.2, -0.1),
            ),
            "weighted evidence",
        ),
        target_position=TargetPosition("long", 0.3),
        book_results=(_book_result(),),
        cycle_status="completed",
        execution_status="partial",
        requires_attention=True,
        created_at=datetime(2026, 8, 29, 2, tzinfo=UTC),
    )


def test_multi_venue_cycle_record_is_immutable_and_rejects_invalid_identity():
    record = _record()
    with pytest.raises(FrozenInstanceError):
        record.cycle_status = "failed"
    with pytest.raises(ValueError, match="cycle_id"):
        replace(record, cycle_id="")
    with pytest.raises(ValueError, match="UTC"):
        replace(record, created_at=record.created_at.replace(tzinfo=None))
    with pytest.raises(ValueError, match="requires_attention"):
        replace(record, requires_attention=False)


def test_multi_venue_cycle_record_rejects_loose_signal_and_target_enum_values():
    record = _record()

    with pytest.raises(ValueError, match="confidence"):
        replace(record, component_signals=(ComponentSignal("loose", "long", 1, "integer confidence"),))
    with pytest.raises(ValueError, match="target_position"):
        replace(record, target_position=TargetPosition("up", 0.3))


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_multi_venue_cycle_round_trip_preserves_book_and_connection_results(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}" if database else None)
    record = _record(cycle_id=f"cycle-{database}")

    await store.save(record)
    loaded = await store.get(record.cycle_id)

    assert loaded == record
    assert loaded is not None
    assert loaded.book_results[0].status == "partial"
    assert tuple(item.status for item in loaded.book_results[0].connection_results) == ("completed", "failed")
    assert loaded.book_results[0].target_weights == (Decimal("0.4"), Decimal("0.6"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "ApiSecret",
        "SECRET",
        "pass-phrase",
        "password",
        "accessToken",
        "credential_ref",
        "privateKey",
    ],
)
async def test_journal_serialization_recursively_rejects_secret_shaped_keys(secret_key):
    from cryptotrader.journal.store import MultiVenueCycleStore

    record = _record(details={"outer": [{"safe": {secret_key: "RAW_SECRET_MARKER"}}]})

    with pytest.raises(ValueError, match=r"^secret field$") as captured:
        await MultiVenueCycleStore().save(record)
    assert "RAW_SECRET_MARKER" not in repr(captured.value)
    assert secret_key not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_multi_venue_store_schema_is_exact_and_ignores_legacy_trading_cycles(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "coexist.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE trading_cycles (cycle_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO trading_cycles VALUES ('legacy-cycle', 'RAW_LEGACY_PAYLOAD')")

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    record = _record()
    await store.save(record)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(multi_venue_cycles)")}
        legacy = connection.execute("SELECT payload FROM trading_cycles").fetchone()[0]

    assert columns == {
        "cycle_id",
        "config_revision",
        "market_data_source_id",
        "component_signals",
        "fused_signal",
        "target_position",
        "book_results",
        "cycle_status",
        "execution_status",
        "requires_attention",
        "created_at",
    }
    assert legacy == "RAW_LEGACY_PAYLOAD"
    assert await store.get("legacy-cycle") is None
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_multi_venue_store_lists_newest_first_and_counts_only_new_records():
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore()
    first = _record(cycle_id="first")
    second = replace(_record(cycle_id="second"), created_at=first.created_at.replace(microsecond=1))
    await store.save(first)
    await store.save(second)

    assert [record.cycle_id for record in await store.list(limit=1)] == ["second"]
    assert [record.cycle_id for record in await store.list(limit=1, offset=1)] == ["first"]
    assert await store.count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_duplicate_cycle_fails_without_database_cause_or_context(tmp_path, database):
    from cryptotrader.journal.store import MultiVenueCycleStore

    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{tmp_path / 'duplicate.db'}" if database else None)
    record = _record()
    await store.save(record)

    with pytest.raises(ValueError, match="cycle already exists") as captured:
        await store.save(record)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_corrupt_multi_venue_payload_fails_closed_without_raw_context(tmp_path):
    from cryptotrader.journal.store import MultiVenueCycleStore

    path = tmp_path / "corrupt.db"
    store = MultiVenueCycleStore(f"sqlite+aiosqlite:///{path}")
    record = _record()
    await store.save(record)

    with sqlite3.connect(path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT book_results FROM multi_venue_cycles WHERE cycle_id = ?",
                (record.cycle_id,),
            ).fetchone()[0]
        )
        payload["items"][0]["proposal"]["book_id"] = "RAW_SECRET_PAYLOAD"
        connection.execute(
            "UPDATE multi_venue_cycles SET book_results = ? WHERE cycle_id = ?",
            (json.dumps(payload), record.cycle_id),
        )

    with pytest.raises(ValueError, match="stored cycle payload") as captured:
        await store.get(record.cycle_id)
    assert "RAW_SECRET_PAYLOAD" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
