"""Test PostgreSQL journal store (using in-memory fallback)."""

from datetime import UTC, datetime

import pytest

from cryptotrader.journal.store import JournalStore
from cryptotrader.models import DecisionCommit, GateResult, TradeVerdict


@pytest.fixture
def store():
    return JournalStore(None)


@pytest.fixture
def sample_commit():
    return DecisionCommit(
        hash="abc12345",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={"funding_rate": 0.001, "volatility": 0.02},
        analyses={},
        debate_rounds=2,
        divergence=0.15,
        verdict=TradeVerdict(action="long", confidence=0.7, position_scale=0.8),
        risk_gate=GateResult(passed=True),
    )


@pytest.mark.asyncio
async def test_commit_and_log(store, sample_commit):
    await store.commit(sample_commit)
    logs = await store.log(limit=10)
    assert len(logs) == 1
    assert logs[0].hash == "abc12345"
    assert logs[0].pair == "BTC/USDT"


@pytest.mark.asyncio
async def test_show(store, sample_commit):
    await store.commit(sample_commit)
    dc = await store.show("abc12345")
    assert dc is not None
    assert dc.verdict.action == "long"


@pytest.mark.asyncio
async def test_show_not_found(store):
    dc = await store.show("nonexist")
    assert dc is None


@pytest.mark.asyncio
async def test_update_pnl(store, sample_commit):
    await store.commit(sample_commit)
    await store.update_pnl("abc12345", 0.05, "Good trade")
    dc = await store.show("abc12345")
    assert dc.pnl == 0.05
    assert dc.retrospective == "Good trade"


@pytest.mark.asyncio
async def test_log_filter_by_pair(store, sample_commit):
    await store.commit(sample_commit)
    c2 = DecisionCommit(
        hash="def67890",
        parent_hash="abc12345",
        timestamp=datetime.now(UTC),
        pair="ETH/USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=1,
    )
    await store.commit(c2)
    btc = await store.log(limit=10, pair="BTC/USDT")
    assert len(btc) == 1
    assert btc[0].pair == "BTC/USDT"


@pytest.mark.asyncio
async def test_log_ordering(store):
    """In-memory log() returns newest-first, matching DB behavior."""
    from datetime import timedelta

    base = datetime.now(UTC)
    for i in range(3):
        c = DecisionCommit(
            hash=f"order_{i}",
            parent_hash=None,
            timestamp=base + timedelta(seconds=i),
            pair="BTC/USDT",
            snapshot_summary={},
            analyses={},
            debate_rounds=0,
        )
        await store.commit(c)

    logs = await store.log(limit=10)
    assert len(logs) == 3
    # newest-first: order_2, order_1, order_0
    assert logs[0].hash == "order_2"
    assert logs[1].hash == "order_1"
    assert logs[2].hash == "order_0"


@pytest.mark.asyncio
async def test_commit_dedup(store):
    """Committing the same hash twice should only store one record."""
    c = DecisionCommit(
        hash="dedup_test",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=0,
    )
    await store.commit(c)
    await store.commit(c)

    logs = await store.log(limit=10)
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_use_db_false_without_url():
    store = JournalStore(None)
    assert not store._use_db
