"""Tests for data/store.py — cache helpers, store/retrieve, forward-fill."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from cryptotrader.data.store import (
    _forward_fill,
    _record_fetch,
    _should_fetch,
    cache_result,
    count_records,
    get_cached_or_none,
    get_data,
    get_latest,
    get_range,
    store_batch,
    store_data,
)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Each test gets its own fresh SQLite database."""
    db_path = tmp_path / "test_store.db"
    monkeypatch.setattr("cryptotrader.data.store.DATA_DB", db_path)
    monkeypatch.setattr("cryptotrader.data.store._conn", None)
    return


class TestStoreData:
    def test_store_and_get(self):
        store_data("test_src", "2025-01-01", {"price": 100})
        result = get_data("test_src", "2025-01-01")
        assert result == {"price": 100}

    def test_get_nonexistent(self):
        assert get_data("nope", "2025-01-01") is None

    def test_store_overwrites(self):
        store_data("src", "2025-01-01", {"v": 1})
        store_data("src", "2025-01-01", {"v": 2})
        assert get_data("src", "2025-01-01") == {"v": 2}

    def test_store_float(self):
        store_data("src", "2025-01-01", 42.5)
        assert get_data("src", "2025-01-01") == 42.5


class TestGetLatest:
    def test_single_record(self):
        store_data("src", "2025-01-01", 10)
        result = get_latest("src", limit=1)
        assert len(result) == 1
        assert result[0] == ("2025-01-01", 10)

    def test_multiple_records_ordered(self):
        store_data("src", "2025-01-01", 10)
        store_data("src", "2025-01-03", 30)
        store_data("src", "2025-01-02", 20)
        result = get_latest("src", limit=2)
        assert len(result) == 2
        assert result[0][0] == "2025-01-03"
        assert result[1][0] == "2025-01-02"

    def test_empty_source(self):
        assert get_latest("nope") == []


class TestGetRange:
    def test_range(self):
        for i in range(1, 6):
            store_data("src", f"2025-01-0{i}", i * 10)
        result = get_range("src", "2025-01-02", "2025-01-04")
        assert len(result) == 3
        assert result["2025-01-02"] == 20
        assert result["2025-01-04"] == 40

    def test_empty_range(self):
        assert get_range("src", "2025-01-01", "2025-01-31") == {}


class TestCountRecords:
    def test_count(self):
        assert count_records("src") == 0
        store_data("src", "2025-01-01", 1)
        store_data("src", "2025-01-02", 2)
        assert count_records("src") == 2


class TestStoreBatch:
    def test_batch_insert(self):
        records = [("2025-01-01", 10), ("2025-01-02", 20), ("2025-01-03", 30)]
        store_batch("src", records)
        assert count_records("src") == 3

    def test_empty_batch(self):
        store_batch("src", [])
        assert count_records("src") == 0

    def test_forward_fill(self):
        records = [("2025-01-01", 10), ("2025-01-04", 40)]
        store_batch("src", records, forward_fill=True)
        assert count_records("src") == 4
        assert get_data("src", "2025-01-02") == 10
        assert get_data("src", "2025-01-03") == 10


class TestForwardFill:
    def test_single_record(self):
        assert _forward_fill([("2025-01-01", 10)]) == [("2025-01-01", 10)]

    def test_no_gap(self):
        records = [("2025-01-01", 10), ("2025-01-02", 20)]
        result = _forward_fill(records)
        assert len(result) == 2

    def test_fills_gap(self):
        records = [("2025-01-01", 10), ("2025-01-04", 40)]
        result = _forward_fill(records)
        assert len(result) == 4
        assert result[1] == ("2025-01-02", 10)
        assert result[2] == ("2025-01-03", 10)

    def test_unsorted_input(self):
        records = [("2025-01-03", 30), ("2025-01-01", 10)]
        result = _forward_fill(records)
        assert result[0][0] == "2025-01-01"
        assert len(result) == 3


class TestShouldFetchAndCache:
    def test_never_fetched(self):
        assert _should_fetch("new_source") is True

    def test_recently_fetched(self):
        _record_fetch("src")
        assert _should_fetch("src") is False

    def test_stale_fetch(self):
        _record_fetch("src")
        with patch("cryptotrader.data.store.time") as mock_time:
            mock_time.time.return_value = time.time() + 600
            assert _should_fetch("src") is True


class TestCacheResult:
    def test_cache_with_date(self):
        cache_result("src", 42.0, date="2025-03-15")
        assert get_data("src", "2025-03-15") == 42.0

    def test_cache_without_date(self):
        cache_result("src", {"val": 1})
        result = get_latest("src", limit=1)
        assert len(result) == 1


class TestGetCachedOrNone:
    def test_backtest_mode_hit(self):
        store_data("src", "2025-01-15", 99.0)
        assert get_cached_or_none("src", date="2025-01-15") == 99.0

    def test_backtest_mode_miss(self):
        assert get_cached_or_none("src", date="2025-01-15") is None

    def test_live_mode_stale(self):
        assert get_cached_or_none("new_source") is None

    def test_live_mode_fresh(self):
        store_data("src", "2025-01-01", 42.0)
        _record_fetch("src")
        result = get_cached_or_none("src")
        assert result == 42.0
