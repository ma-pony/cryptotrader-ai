"""Offline tests for Kronos auxiliary market data."""

from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.data.kronos_aux import _closed_interval_rows, fetch_premium_close_5d


def test_closed_interval_rows_excludes_forming_period():
    interval_ms = 4 * 60 * 60 * 1000
    now_ms = 3 * interval_ms + 60_000
    rows = [[i * interval_ms, i] for i in range(4)]

    assert _closed_interval_rows(rows, interval_ms, now_ms) == rows[:3]


@pytest.mark.asyncio
async def test_premium_average_uses_closed_klines_only():
    interval_ms = 4 * 60 * 60 * 1000
    rows = [
        [0, "0", "0", "0", "1.0"],
        [interval_ms, "0", "0", "0", "2.0"],
        [2 * interval_ms, "0", "0", "0", "3.0"],
        [3 * interval_ms, "0", "0", "0", "99.0"],
    ]

    with (
        patch("cryptotrader.data.kronos_aux._get_json", new=AsyncMock(return_value=rows)),
        patch("time.time", return_value=(3 * interval_ms + 60_000) / 1000),
    ):
        result = await fetch_premium_close_5d()

    assert result == pytest.approx(2.0)
