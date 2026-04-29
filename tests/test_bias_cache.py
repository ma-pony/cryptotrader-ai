"""Tests for the _build_bias 60s cache in api.routes.decisions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from api.routes.decisions import _bias_cache, _build_bias


@pytest.mark.asyncio
async def test_build_bias_cache_hits_skip_detect_biases() -> None:
    """Second call within TTL must not re-run detect_biases."""
    _bias_cache.clear()

    class FakeStore:
        pass

    store = FakeStore()

    stats = {
        "tech_agent": {
            "accuracy": 0.7,
            "neutral_rate": 0.1,
            "bullish_rate": 0.7,
            "bearish_rate": 0.2,
            "avg_conf_when_right": 0.75,
            "avg_conf_when_wrong": 0.45,
            "sample_size": 30,
        },
    }

    with patch("cryptotrader.journal.calibrate.detect_biases", new=AsyncMock(return_value=stats)) as mock:
        first = await _build_bias(store)
        second = await _build_bias(store)

    assert first is second  # same object returned from cache
    assert mock.call_count == 1  # second call cached


@pytest.mark.asyncio
async def test_build_bias_cache_empty_result_cached() -> None:
    """When detect_biases returns empty, None is also cached (avoids repeated scans)."""
    _bias_cache.clear()

    class FakeStore:
        pass

    store = FakeStore()

    with patch("cryptotrader.journal.calibrate.detect_biases", new=AsyncMock(return_value={})) as mock:
        first = await _build_bias(store)
        second = await _build_bias(store)

    assert first is None
    assert second is None
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_build_bias_cache_exception_cached() -> None:
    """detect_biases exception path also cached as None to avoid hammering."""
    _bias_cache.clear()

    class FakeStore:
        pass

    store = FakeStore()

    with patch(
        "cryptotrader.journal.calibrate.detect_biases",
        new=AsyncMock(side_effect=RuntimeError("db down")),
    ) as mock:
        first = await _build_bias(store)
        second = await _build_bias(store)

    assert first is None
    assert second is None
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_build_bias_different_stores_separate_cache_slots() -> None:
    """Cache key is id(store), so two different stores must not share entries."""
    _bias_cache.clear()

    class FakeStore:
        pass

    store_a = FakeStore()
    store_b = FakeStore()

    stats = {
        "chain_agent": {
            "accuracy": 0.5,
            "neutral_rate": 0.2,
            "bullish_rate": 0.4,
            "bearish_rate": 0.4,
            "avg_conf_when_right": 0.6,
            "avg_conf_when_wrong": 0.5,
            "sample_size": 10,
        },
    }

    with patch("cryptotrader.journal.calibrate.detect_biases", new=AsyncMock(return_value=stats)) as mock:
        await _build_bias(store_a)
        await _build_bias(store_b)

    assert mock.call_count == 2  # different stores → different cache keys


@pytest.mark.asyncio
async def test_build_bias_severity_escalates_with_warnings() -> None:
    """Severity heuristic: 0 warnings → low; 1-2 → medium; 3+ → high."""
    _bias_cache.clear()

    class FakeStore:
        pass

    store = FakeStore()

    # 30% neutral (no warning) + 80% bullish (directional warning)
    stats = {
        f"agent_{i}": {
            "accuracy": 0.2,  # triggers low-accuracy warning
            "neutral_rate": 0.5,  # high neutral rate warning
            "bullish_rate": 0.9,  # directional bias warning
            "bearish_rate": 0.05,
            "avg_conf_when_right": 0.8,
            "avg_conf_when_wrong": 0.9,  # overconfidence warning
            "sample_size": 50,
        }
        for i in range(1)
    }

    with patch("cryptotrader.journal.calibrate.detect_biases", new=AsyncMock(return_value=stats)):
        result = await _build_bias(store)

    assert result is not None
    # At least 3 warnings for a single agent = "high" severity
    assert result.severity == "high"
