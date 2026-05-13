"""Phase 2C — DecisionCommit + JournalStore round-trip for stop_loss_price /
take_profit_price / algo_id audit fields.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from cryptotrader._compat import UTC
from cryptotrader.journal.commit import build_commit
from cryptotrader.journal.store import JournalStore
from cryptotrader.models import DecisionCommit, GateResult, TradeVerdict


@pytest.fixture
def store():
    return JournalStore(None)


def _make_dc_with_sl_tp(**overrides) -> DecisionCommit:
    base = {
        "hash": "phase2c01",
        "parent_hash": None,
        "timestamp": datetime.now(UTC),
        "pair": "DOGE/USDT:USDT",
        "snapshot_summary": {"price": 0.1139},
        "analyses": {},
        "debate_rounds": 2,
        "divergence": 0.0,
        "verdict": TradeVerdict(
            action="long",
            confidence=0.62,
            position_scale=0.22,
            stop_loss=0.1080,
            take_profit=0.1200,
        ),
        "risk_gate": GateResult(passed=True),
        "stop_loss_price": 0.1080,
        "take_profit_price": 0.1200,
        "algo_id": "okx-algo-xyz789",
    }
    base.update(overrides)
    return DecisionCommit(**base)


# ── DecisionCommit dataclass ────────────────────────────────────────────


def test_decision_commit_defaults_sl_tp_none():
    dc = DecisionCommit(
        hash="h0",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=0,
    )
    assert dc.stop_loss_price is None
    assert dc.take_profit_price is None
    assert dc.algo_id is None


def test_decision_commit_carries_sl_tp_when_provided():
    dc = _make_dc_with_sl_tp()
    assert dc.stop_loss_price == 0.1080
    assert dc.take_profit_price == 0.1200
    assert dc.algo_id == "okx-algo-xyz789"


# ── build_commit ────────────────────────────────────────────────────────


def test_build_commit_accepts_sl_tp_kwargs():
    dc = build_commit(
        pair="DOGE/USDT:USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=2,
        divergence=0.0,
        verdict=TradeVerdict(
            action="long",
            confidence=0.62,
            position_scale=0.22,
            stop_loss=0.1080,
            take_profit=0.1200,
        ),
        risk_gate=GateResult(passed=True),
        order=None,
        parent_hash=None,
        stop_loss_price=0.1080,
        take_profit_price=0.1200,
        algo_id="okx-algo-xyz789",
    )
    assert dc.stop_loss_price == 0.1080
    assert dc.take_profit_price == 0.1200
    assert dc.algo_id == "okx-algo-xyz789"


def test_build_commit_sl_tp_defaults_to_none():
    dc = build_commit(
        pair="BTC/USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=0,
        divergence=0.0,
        verdict=TradeVerdict(action="hold"),
        risk_gate=GateResult(passed=True),
        order=None,
        parent_hash=None,
    )
    assert dc.stop_loss_price is None
    assert dc.take_profit_price is None
    assert dc.algo_id is None


# ── In-memory store round-trip ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_inmemory_store_roundtrip_preserves_sl_tp(store):
    dc = _make_dc_with_sl_tp()
    await store.commit(dc)
    [restored] = await store.log(limit=10)
    assert restored.stop_loss_price == 0.1080
    assert restored.take_profit_price == 0.1200
    assert restored.algo_id == "okx-algo-xyz789"


@pytest.mark.asyncio
async def test_inmemory_store_none_sl_tp_stays_none(store):
    dc = DecisionCommit(
        hash="hN",
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={},
        analyses={},
        debate_rounds=0,
        verdict=TradeVerdict(action="hold"),
    )
    await store.commit(dc)
    [restored] = await store.log(limit=10)
    assert restored.stop_loss_price is None
    assert restored.take_profit_price is None
    assert restored.algo_id is None


# ── SQLite migration smoke test ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_migration_adds_sl_tp_columns(tmp_path):
    """Verify that _OBSERVABILITY_COLUMNS migration adds the 3 new cols
    to a fresh SQLite database and round-trips a commit through them.
    """
    db_file = tmp_path / "test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    store = JournalStore(db_url)

    dc = _make_dc_with_sl_tp()
    await store.commit(dc)
    logs = await store.log(limit=10)
    assert len(logs) == 1
    assert logs[0].stop_loss_price == 0.1080
    assert logs[0].take_profit_price == 0.1200
    assert logs[0].algo_id == "okx-algo-xyz789"
