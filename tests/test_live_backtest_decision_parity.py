"""实时与历史行情来源共享相同的信号到目标仓位映射。"""

from __future__ import annotations

import pytest

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
