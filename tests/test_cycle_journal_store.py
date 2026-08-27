"""交易周期 Journal 的持久化契约。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.factories.signal_fusion import cycle_record


@pytest.mark.asyncio
async def test_cycle_record_round_trips_component_contributions(tmp_path):
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}")
    record = cycle_record(
        status="completed",
        profile_revision=3,
        component_signals=(
            {
                "component_id": "kronos",
                "direction": "long",
                "confidence": 0.8,
            },
        ),
        fused_signal={
            "score": 0.48,
            "contributions": [
                {"component_id": "kronos", "weighted_score": 0.48},
            ],
        },
        target_position={"side": "long", "size_ratio": 0.48},
    )

    await store.append(record)

    assert await store.get(record.cycle_id) == record


@pytest.mark.asyncio
async def test_component_failure_is_journaled_without_trade_plan(tmp_path):
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}")
    record = cycle_record(
        status="component_failed",
        component_error={"llm_committee": "RuntimeError: timeout"},
    )

    await store.append(record)
    loaded = await store.get(record.cycle_id)

    assert loaded is not None
    assert loaded.component_error == {"llm_committee": "RuntimeError: timeout"}
    assert loaded.trade_plan is None


@pytest.mark.asyncio
async def test_list_filters_pair_and_returns_newest_first(tmp_path):
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}")
    first = cycle_record(cycle_id="first", pair="BTC/USDT:USDT")
    second = replace(
        cycle_record(cycle_id="second", pair="ETH/USDT:USDT"),
        created_at=first.created_at.replace(microsecond=first.created_at.microsecond + 1),
    )
    third = replace(
        cycle_record(cycle_id="third", pair="BTC/USDT:USDT"),
        created_at=first.created_at.replace(microsecond=first.created_at.microsecond + 2),
    )
    for record in (first, second, third):
        await store.append(record)

    records = await store.list(pair="BTC/USDT:USDT", limit=10)

    assert [record.cycle_id for record in records] == ["third", "first"]


@pytest.mark.asyncio
async def test_memory_store_is_instance_local():
    from cryptotrader.journal.store import CycleJournalStore

    first = CycleJournalStore()
    second = CycleJournalStore()
    record = cycle_record()

    await first.append(record)

    assert await first.get(record.cycle_id) == record
    assert await second.get(record.cycle_id) is None


@pytest.mark.asyncio
async def test_append_rejects_duplicate_cycle_id():
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore()
    record = cycle_record()
    await store.append(record)

    with pytest.raises(ValueError, match="already exists"):
        await store.append(record)


@pytest.mark.asyncio
async def test_replace_moves_pending_cycle_to_its_terminal_state():
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore()
    pending = cycle_record(status="awaiting_approval")
    completed = replace(pending, status="completed", hitl_result={"status": "approved"})
    await store.append(pending)

    await store.replace(completed)

    assert store.records == [completed]


@pytest.mark.asyncio
async def test_database_replace_moves_pending_cycle_to_its_terminal_state(tmp_path):
    from cryptotrader.journal.store import CycleJournalStore

    store = CycleJournalStore(f"sqlite+aiosqlite:///{tmp_path / 'cycles.db'}")
    pending = cycle_record(status="awaiting_approval")
    completed = replace(pending, status="completed", hitl_result={"status": "approved"})
    await store.append(pending)

    await store.replace(completed)

    assert await store.get(pending.cycle_id) == completed
