"""实时与历史行情来源共享相同的信号到目标仓位映射。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.decision.models import CycleRequest
from tests.test_multi_book_cycle import PAIR, _book, _cycle, _snapshot


@pytest.mark.asyncio
async def test_live_and_historical_sources_produce_the_same_platform_neutral_target():
    book = _book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False)
    snapshot = _snapshot(book)
    live, *_ = _cycle(snapshot)
    historical, *_ = _cycle(snapshot)

    live_outcome = await live.run(CycleRequest(PAIR))
    historical_outcome = await historical.run(CycleRequest(PAIR))

    assert live_outcome.target_position == historical_outcome.target_position
    assert live_outcome.target_position.signed_ratio == 1.0


def test_backtest_snapshot_replaces_every_configured_book_with_one_hundred_percent_paper_target():
    source = _snapshot(_book("simulation", "simulated", ("sim-first", "sim-second"), hitl=False))
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", initial_capital=10_000.0)

    snapshot = engine._backtest_snapshot(source)

    book = snapshot.document.execution.books[0]
    assert snapshot.revision == source.revision
    assert book.id == "backtest"
    assert tuple(item.connection_id for item in book.allocations) == ("backtest-paper",)
    assert tuple(item.weight for item in book.allocations) == (Decimal("1.0"),)
