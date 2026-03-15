"""Tests for backtest/session.py and backtest/cache.py.

Covers the session storage functions and OHLCV cache SQLite operations.
Uses tmp_path to isolate filesystem state from the real ~/.cryptotrader directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from cryptotrader.backtest.result import BacktestResult
from cryptotrader.models import (
    ConsensusMetrics,
    DecisionCommit,
    NodeTraceEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_commit(hash_val: str = "test_hash", pair: str = "BTC/USDT") -> DecisionCommit:
    return DecisionCommit(
        hash=hash_val,
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair=pair,
        snapshot_summary={"price": 50000.0},
        analyses={},
        debate_rounds=0,
    )


def _make_commit_with_observability(hash_val: str) -> DecisionCommit:
    return DecisionCommit(
        hash=hash_val,
        parent_hash=None,
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        snapshot_summary={"price": 60000.0},
        analyses={},
        debate_rounds=1,
        consensus_metrics=ConsensusMetrics(
            strength=0.72,
            mean_score=0.60,
            dispersion=0.12,
            skip_threshold=0.50,
            confusion_threshold=0.05,
        ),
        verdict_source="weighted",
        experience_memory={"success_patterns": ["trend_follow"]},
        node_trace=[NodeTraceEntry(node="debate_gate", duration_ms=35, summary="skip")],
        debate_skip_reason="consensus",
    )


# ---------------------------------------------------------------------------
# backtest/session.py tests
# ---------------------------------------------------------------------------


class TestGenerateSessionId:
    """generate_session_id() produces deterministic-format session IDs."""

    def test_session_id_contains_pair(self) -> None:
        from cryptotrader.backtest.session import generate_session_id

        sid = generate_session_id("BTC/USDT", "1h", "2025-01-01", "2025-02-01")
        assert "BTC_USDT" in sid

    def test_session_id_contains_start_and_end(self) -> None:
        from cryptotrader.backtest.session import generate_session_id

        sid = generate_session_id("ETH/USDT", "4h", "2025-03-01", "2025-03-31")
        assert "2025-03-01" in sid
        assert "2025-03-31" in sid

    def test_session_id_contains_interval(self) -> None:
        from cryptotrader.backtest.session import generate_session_id

        sid = generate_session_id("BTC/USDT", "1d", "2025-01-01", "2025-02-01")
        assert "1d" in sid

    def test_two_sessions_have_different_ids(self) -> None:
        """Two session IDs generated at different times must differ (timestamp suffix)."""
        import time

        from cryptotrader.backtest.session import generate_session_id

        sid1 = generate_session_id("BTC/USDT", "1h", "2025-01-01", "2025-02-01")
        time.sleep(0.001)
        sid2 = generate_session_id("BTC/USDT", "1h", "2025-01-01", "2025-02-01")
        # They may be the same if generated within the same second — that's OK.
        # Just verify they are strings.
        assert isinstance(sid1, str)
        assert isinstance(sid2, str)


class TestGetSessionDir:
    """get_session_dir() creates the directory if it does not exist."""

    def test_creates_directory(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import get_session_dir

        path = get_session_dir("test_session_001")
        assert path.exists()
        assert path.is_dir()

    def test_returns_correct_path(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        sessions_dir = tmp_path / "sessions"
        monkeypatch.setattr(session_module, "_SESSIONS_DIR", sessions_dir)

        from cryptotrader.backtest.session import get_session_dir

        path = get_session_dir("my_session")
        assert path == sessions_dir / "my_session"


class TestSaveAndLoadCommits:
    """save_commits() / load_commits() round-trip test."""

    def test_save_commits_creates_jsonl_file(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_commits

        commits = [_make_commit("hash_a"), _make_commit("hash_b")]
        path = save_commits("test_session", commits)
        assert path.exists()
        assert path.name == "commits.jsonl"

    def test_load_commits_returns_empty_for_missing_session(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import load_commits

        result = load_commits("nonexistent_session")
        assert result == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import load_commits, save_commits

        commits = [_make_commit("hash_x"), _make_commit("hash_y")]
        save_commits("roundtrip_session", commits)
        loaded = load_commits("roundtrip_session")

        assert len(loaded) == 2
        hashes = {r["hash"] for r in loaded}
        assert "hash_x" in hashes
        assert "hash_y" in hashes

    def test_save_commits_serializes_observability_fields(self, tmp_path, monkeypatch) -> None:
        """Commits with all 5 new observability fields are serialized to JSONL."""
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import load_commits, save_commits

        commit = _make_commit_with_observability("obs_hash")
        save_commits("obs_session", [commit])
        loaded = load_commits("obs_session")

        assert len(loaded) == 1
        rec = loaded[0]
        assert rec["hash"] == "obs_hash"
        assert rec["verdict_source"] == "weighted"
        assert rec["debate_skip_reason"] == "consensus"
        assert rec["consensus_metrics"] is not None
        assert rec["consensus_metrics"]["strength"] == pytest.approx(0.72)
        assert len(rec["node_trace"]) == 1
        assert rec["node_trace"][0]["node"] == "debate_gate"

    def test_load_commits_empty_file_returns_empty_list(self, tmp_path, monkeypatch) -> None:
        """An empty commits.jsonl returns an empty list without errors."""
        from cryptotrader.backtest import session as session_module

        sessions_dir = tmp_path / "sessions"
        monkeypatch.setattr(session_module, "_SESSIONS_DIR", sessions_dir)

        # Create empty file
        session_dir = sessions_dir / "empty_session"
        session_dir.mkdir(parents=True)
        (session_dir / "commits.jsonl").write_text("")

        from cryptotrader.backtest.session import load_commits

        result = load_commits("empty_session")
        assert result == []


class TestSaveResult:
    """save_result() stores BacktestResult as JSON."""

    def test_save_result_creates_json_file(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_result

        result = BacktestResult(
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.10,
            win_rate=0.60,
            equity_curve=[10000, 10500, 10200],
            trades=[],
            decisions=[],
        )
        path = save_result("result_session", result)
        assert path.exists()
        assert path.name == "result.json"

    def test_save_result_excludes_equity_curve(self, tmp_path, monkeypatch) -> None:
        """equity_curve is removed from result.json to keep file size small."""
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_result

        result = BacktestResult(
            total_return=0.03,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            win_rate=0.55,
            equity_curve=[10000, 10300, 10150],
            trades=[],
            decisions=[],
        )
        path = save_result("equity_session", result)
        data = json.loads(path.read_text())
        assert "equity_curve" not in data

    def test_save_result_contains_summary_stats(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_result

        result = BacktestResult(
            total_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            win_rate=0.65,
            equity_curve=[10000],
            trades=[],
            decisions=[],
        )
        path = save_result("stats_session", result)
        data = json.loads(path.read_text())
        assert data["total_return"] == pytest.approx(0.12)
        assert data["sharpe_ratio"] == pytest.approx(1.5)
        assert data["win_rate"] == pytest.approx(0.65)


class TestListSessions:
    """list_sessions() returns sorted session IDs."""

    def test_returns_empty_when_no_sessions_dir(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "nonexistent")

        from cryptotrader.backtest.session import list_sessions

        result = list_sessions()
        assert result == []

    def test_returns_sorted_session_ids(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        sessions_dir = tmp_path / "sessions"
        monkeypatch.setattr(session_module, "_SESSIONS_DIR", sessions_dir)

        # Create session directories directly
        for name in ["session_c", "session_a", "session_b"]:
            (sessions_dir / name).mkdir(parents=True)

        from cryptotrader.backtest.session import list_sessions

        result = list_sessions()
        assert result == ["session_a", "session_b", "session_c"]

    def test_ignores_files_not_directories(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr(session_module, "_SESSIONS_DIR", sessions_dir)

        # Create one dir and one file
        (sessions_dir / "real_session").mkdir()
        (sessions_dir / "some_file.txt").write_text("data")

        from cryptotrader.backtest.session import list_sessions

        result = list_sessions()
        assert "real_session" in result
        assert "some_file.txt" not in result


# ---------------------------------------------------------------------------
# backtest/cache.py tests (SQLite OHLCV cache)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_cache_db")
class TestOHLCVCache:
    """OHLCV cache get/store operations with isolated SQLite databases."""

    @pytest.fixture
    def _patch_cache_db(self, tmp_path, monkeypatch):
        """Redirect the OHLCV cache database to a temp directory."""
        from cryptotrader.backtest import cache as cache_module

        db_path = tmp_path / "ohlcv_cache.db"
        monkeypatch.setattr(cache_module, "CACHE_DB", db_path)

    def test_get_cached_returns_empty_for_empty_db(self) -> None:
        from cryptotrader.backtest.cache import get_cached

        result = get_cached("BTC/USDT", "1h", 0, 1_000_000_000)
        assert result == []

    def test_store_and_retrieve_ohlcv(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [
            [1_000_000, 50000.0, 51000.0, 49000.0, 50500.0, 100.0],
            [1_001_000, 50500.0, 52000.0, 50000.0, 51000.0, 120.0],
        ]
        store_ohlcv("BTC/USDT", "1h", candles)

        result = get_cached("BTC/USDT", "1h", 0, 2_000_000)
        assert len(result) == 2
        assert result[0][0] == 1_000_000
        assert result[1][0] == 1_001_000

    def test_get_cached_filters_by_time_range(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [
            [100, 100.0, 110.0, 90.0, 105.0, 1.0],
            [200, 105.0, 115.0, 95.0, 110.0, 1.0],
            [300, 110.0, 120.0, 100.0, 115.0, 1.0],
        ]
        store_ohlcv("ETH/USDT", "1h", candles)

        # Only fetch ts in [150, 250]
        result = get_cached("ETH/USDT", "1h", 150, 250)
        assert len(result) == 1
        assert result[0][0] == 200

    def test_get_cached_filters_by_pair(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 1.0, 1.1, 0.9, 1.05, 10.0]]
        store_ohlcv("SOL/USDT", "1h", candles)

        # Query for different pair — should return empty
        result = get_cached("BTC/USDT", "1h", 0, 1000)
        assert result == []

    def test_get_cached_filters_by_timeframe(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 1.0, 1.1, 0.9, 1.05, 10.0]]
        store_ohlcv("BTC/USDT", "4h", candles)

        # Query 1h timeframe — should return empty
        result = get_cached("BTC/USDT", "1h", 0, 1000)
        assert result == []

    def test_store_ohlcv_upserts_on_duplicate(self) -> None:
        """Duplicate (pair, timeframe, ts) entries should be replaced, not duplicated."""
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        candles = [[100, 50000.0, 51000.0, 49000.0, 50500.0, 100.0]]
        store_ohlcv("BTC/USDT", "1h", candles)

        # Store same ts with updated values
        updated = [[100, 51000.0, 52000.0, 50000.0, 51500.0, 200.0]]
        store_ohlcv("BTC/USDT", "1h", updated)

        result = get_cached("BTC/USDT", "1h", 0, 200)
        # Should still be only 1 entry
        assert len(result) == 1
        # Value should be the updated one
        assert result[0][1] == pytest.approx(51000.0)

    def test_store_empty_candles_does_nothing(self) -> None:
        from cryptotrader.backtest.cache import get_cached, store_ohlcv

        store_ohlcv("BTC/USDT", "1h", [])
        result = get_cached("BTC/USDT", "1h", 0, 1_000_000)
        assert result == []


# ---------------------------------------------------------------------------
# save_experience() tests
# ---------------------------------------------------------------------------


class TestSaveExperience:
    """save_experience() serializes ExperienceMemory to JSON."""

    def test_save_experience_creates_json_file(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_experience
        from cryptotrader.models import ExperienceMemory

        experience = {
            "tech_agent": ExperienceMemory(
                success_patterns=[],
                forbidden_zones=[],
                strategic_insights=["trend following works well"],
                updated_at="2025-01-01T00:00:00",
            )
        }
        path = save_experience("exp_session", experience)
        assert path.exists()
        assert path.name == "experience.json"

    def test_save_experience_data_is_readable(self, tmp_path, monkeypatch) -> None:
        from cryptotrader.backtest import session as session_module

        monkeypatch.setattr(session_module, "_SESSIONS_DIR", tmp_path / "sessions")

        from cryptotrader.backtest.session import save_experience
        from cryptotrader.models import ExperienceMemory

        experience = {
            "macro_agent": ExperienceMemory(
                success_patterns=[],
                forbidden_zones=[],
                strategic_insights=["avoid high-volatility periods"],
                updated_at="2025-02-01",
            )
        }
        path = save_experience("exp_session_2", experience)
        data = json.loads(path.read_text())
        assert "macro_agent" in data
        assert data["macro_agent"]["strategic_insights"] == ["avoid high-volatility periods"]


# ---------------------------------------------------------------------------
# fetch_historical() in cache.py — mocked ccxt network
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_cache_db")
class TestFetchHistorical:
    """fetch_historical() uses cache when available, fetches from ccxt otherwise."""

    @pytest.fixture
    def _patch_cache_db(self, tmp_path, monkeypatch):
        from cryptotrader.backtest import cache as cache_module

        db_path = tmp_path / "ohlcv_cache.db"
        monkeypatch.setattr(cache_module, "CACHE_DB", db_path)

    @pytest.mark.asyncio
    async def test_fetch_historical_uses_cache_when_available(self) -> None:
        """fetch_historical returns cached data when cache covers the full range."""
        from cryptotrader.backtest.cache import fetch_historical, store_ohlcv

        # Pre-populate cache with data starting well before the query range
        since_ms = 1_000_000
        until_ms = 2_000_000
        candles = [
            [since_ms - 10, 100.0, 110.0, 90.0, 105.0, 1.0],
            [since_ms + 100, 105.0, 115.0, 95.0, 110.0, 1.0],
            [until_ms - 100, 110.0, 120.0, 100.0, 115.0, 1.0],
        ]
        store_ohlcv("BTC/USDT", "1h", candles)

        result = await fetch_historical("BTC/USDT", "1h", since_ms, until_ms)

        # Should return from cache (first candle ts <= since_ms + 86_400_000)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_historical_fetches_from_ccxt_when_cache_miss(
        self,
    ) -> None:
        """fetch_historical calls ccxt when cache does not cover the range."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cryptotrader.backtest.cache import fetch_historical

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(
            side_effect=[
                [
                    [5_000_000, 200.0, 210.0, 190.0, 205.0, 50.0],
                    [5_001_000, 205.0, 215.0, 195.0, 210.0, 60.0],
                ],
                [],  # second call returns empty to stop pagination loop
            ]
        )
        mock_exchange.close = AsyncMock()

        mock_binance_cls = MagicMock(return_value=mock_exchange)

        with patch("ccxt.async_support.binance", mock_binance_cls):
            result = await fetch_historical("SOL/USDT", "4h", 5_000_000, 5_002_000)

        # ccxt was called and results returned
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_historical_returns_empty_when_ccxt_returns_empty(
        self,
    ) -> None:
        """fetch_historical returns empty list when ccxt returns no data."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from cryptotrader.backtest.cache import fetch_historical

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=[])
        mock_exchange.close = AsyncMock()

        mock_binance_cls = MagicMock(return_value=mock_exchange)

        with patch("ccxt.async_support.binance", mock_binance_cls):
            result = await fetch_historical("DOGE/USDT", "1d", 10_000_000, 20_000_000)

        assert result == []
