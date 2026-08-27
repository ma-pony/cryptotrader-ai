from __future__ import annotations

from datetime import UTC, datetime

from cryptotrader.backtest.engine import BacktestEngine


def test_future_candle_changes_cannot_change_point_in_time_snapshot():
    prefix = [
        [1_704_067_200_000 + index * 3_600_000, 100 + index, 102 + index, 99 + index, 101 + index, 10]
        for index in range(4)
    ]
    bullish_future = [prefix[-1][0] + 3_600_000, 104, 150, 103, 145, 100]
    bearish_future = [prefix[-1][0] + 3_600_000, 104, 105, 50, 55, 100]
    left = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    right = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    left._candles_by_timeframe = {"1h": [*prefix, bullish_future]}
    right._candles_by_timeframe = {"1h": [*prefix, bearish_future]}
    as_of = datetime.fromtimestamp(prefix[-1][0] / 1000, UTC)

    left_snapshot = left._snapshot_at("1h", as_of)
    right_snapshot = right._snapshot_at("1h", as_of)

    assert left_snapshot.market.ohlcv.equals(right_snapshot.market.ohlcv)
    assert left_snapshot.news == right_snapshot.news
    assert left_snapshot.macro == right_snapshot.macro
    assert len(left_snapshot.market.ohlcv) == len(prefix)
