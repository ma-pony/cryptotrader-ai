"""Regression tests for Phase 1 deep-review fixes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.routes.market import _fetch_long_short


class TestFetchLongShortGuards:
    """Regression: I-C4 — _fetch_long_short must guard against non-200 HTTP status."""

    @pytest.mark.asyncio
    async def test_non_binance_exchange_returns_none(self) -> None:
        assert await _fetch_long_short("BTC/USDT", "okx") == (None, None)

    @pytest.mark.asyncio
    async def test_timeout_returns_none_not_raises(self) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_long_short("BTC/USDT", "binance")
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_non_200_response_skipped(self) -> None:
        """429 rate-limit must not crash json() parsing."""
        bad_resp = MagicMock()
        bad_resp.status_code = 429
        bad_resp.json = MagicMock(side_effect=ValueError("Rate limit HTML body"))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=bad_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_long_short("BTC/USDT", "binance")
        # Non-200 → guard triggers, no parse, None returned, no raise
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_malformed_json_caught(self) -> None:
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json = MagicMock(side_effect=ValueError("not json"))

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=ok_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _fetch_long_short("BTC/USDT", "binance")
        assert result == (None, None)

    @pytest.mark.asyncio
    async def test_success_path_returns_both_ratios(self) -> None:
        g_resp = MagicMock()
        g_resp.status_code = 200
        g_resp.json = MagicMock(return_value=[{"longShortRatio": "1.42"}])
        t_resp = MagicMock()
        t_resp.status_code = 200
        t_resp.json = MagicMock(return_value=[{"longAccount": "0.58"}])

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=[g_resp, t_resp])

        with patch("httpx.AsyncClient", return_value=mock_client):
            ls, top = await _fetch_long_short("BTC/USDT", "binance")
        assert ls == pytest.approx(1.42)
        assert top == pytest.approx(0.58)


class TestCacheHitRateClamp:
    """Regression: I-C3 — cache_hit_rate must never exceed 1.0.

    Verified indirectly by re-importing the metrics endpoint and inspecting the
    computed value. Anthropic prompt caching can emit multi-segment cache_read
    per call, so raw cache_hits > calls is possible.
    """

    @pytest.mark.asyncio
    async def test_multi_segment_cache_clamped(self) -> None:
        from datetime import datetime, timedelta

        from api.routes.metrics import _llm_accounting_last_24h
        from cryptotrader._compat import UTC

        now = datetime.now(UTC)
        fake_commit = MagicMock()
        fake_commit.timestamp = now
        fake_commit.token_usage = {"calls": 2, "cost_usd": 0.05, "cache_hits": 5}
        fake_commit.pair = "BTC/USDT"

        older_commit = MagicMock()
        older_commit.timestamp = now - timedelta(days=5)
        older_commit.token_usage = {"calls": 1, "cost_usd": 0.01, "cache_hits": 0}
        older_commit.pair = "BTC/USDT"

        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.log = AsyncMock(return_value=[fake_commit, older_commit])
            calls, _cost, cache_hit_rate, _dpd = await _llm_accounting_last_24h(None)

        # cache_hits(5) / calls(2) = 2.5 raw — must be clamped to 1.0
        assert calls == 2
        assert cache_hit_rate <= 1.0
        assert cache_hit_rate == pytest.approx(1.0)


class TestSqliteMigration:
    """Regression: I-C2 — SQLite DBs must receive new columns via PRAGMA+ALTER."""

    @pytest.mark.asyncio
    async def test_sqlite_adds_missing_columns(self, tmp_path) -> None:
        """Create a SQLite DB with the old schema, ensure _ensure_tables adds new columns."""
        db_path = tmp_path / "journal.sqlite"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        # First: create the DB with the current schema (all cols). Then drop the two new
        # columns manually to simulate a pre-Phase-1 DB.
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """CREATE TABLE decision_commits (
                        hash VARCHAR(16) PRIMARY KEY,
                        parent_hash VARCHAR(16),
                        timestamp DATETIME NOT NULL,
                        pair VARCHAR(20) NOT NULL,
                        snapshot_summary TEXT,
                        analyses TEXT,
                        debate_rounds INTEGER DEFAULT 0,
                        challenges TEXT,
                        divergence FLOAT DEFAULT 0.0,
                        verdict TEXT,
                        risk_gate TEXT,
                        order_data TEXT,
                        fill_price FLOAT,
                        slippage FLOAT,
                        portfolio_after TEXT,
                        pnl FLOAT,
                        retrospective TEXT,
                        trace_id VARCHAR(36),
                        consensus_metrics TEXT,
                        verdict_source VARCHAR(20) NOT NULL DEFAULT 'ai',
                        experience_memory TEXT,
                        node_trace TEXT NOT NULL DEFAULT '[]',
                        debate_skip_reason VARCHAR(500) NOT NULL DEFAULT ''
                    )"""
                )
            )
        await engine.dispose()

        # Now run _ensure_tables via JournalStore — it should add the 2 missing columns.
        from cryptotrader.journal.store import _ensure_tables, _table_ready

        _table_ready.discard(db_url)  # force re-migration
        await _ensure_tables(db_url)

        # Verify columns exist
        engine2 = create_async_engine(db_url)
        async with engine2.begin() as conn:
            result = await conn.execute(text("PRAGMA table_info(decision_commits)"))
            cols = {row[1] for row in result.fetchall()}
        await engine2.dispose()

        assert "latency_breakdown" in cols
        assert "token_usage" in cols

    @pytest.mark.asyncio
    async def test_sqlite_migration_idempotent(self, tmp_path) -> None:
        """Running _ensure_tables twice must not error — PRAGMA guards prevent dup ALTER.

        We pre-create the table with TEXT columns (SQLite can't compile JSONB), then
        verify the migration handles both:
        - First call: adds missing latency_breakdown + token_usage via ALTER
        - Second call: PRAGMA shows cols present, ALTER loop skips (no duplicate-column error)
        """
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        from cryptotrader.journal.store import _ensure_tables, _table_ready

        db_path = tmp_path / "journal2.sqlite"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        # Pre-create table with core columns (no latency/token columns) to simulate
        # an upgrade-from-old-schema scenario.
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """CREATE TABLE decision_commits (
                        hash VARCHAR(16) PRIMARY KEY,
                        parent_hash VARCHAR(16),
                        timestamp DATETIME NOT NULL,
                        pair VARCHAR(20) NOT NULL,
                        snapshot_summary TEXT,
                        analyses TEXT,
                        debate_rounds INTEGER DEFAULT 0,
                        challenges TEXT,
                        divergence FLOAT DEFAULT 0.0,
                        verdict TEXT,
                        risk_gate TEXT,
                        order_data TEXT,
                        fill_price FLOAT,
                        slippage FLOAT,
                        portfolio_after TEXT,
                        pnl FLOAT,
                        retrospective TEXT,
                        trace_id VARCHAR(36),
                        consensus_metrics TEXT,
                        verdict_source VARCHAR(20) NOT NULL DEFAULT 'ai',
                        experience_memory TEXT,
                        node_trace TEXT NOT NULL DEFAULT '[]',
                        debate_skip_reason VARCHAR(500) NOT NULL DEFAULT ''
                    )"""
                )
            )
        await engine.dispose()

        _table_ready.discard(db_url)
        await _ensure_tables(db_url)  # first run: adds 2 new cols
        _table_ready.discard(db_url)
        await _ensure_tables(db_url)  # second run: must not raise "duplicate column"

        # Confirm columns still present and exactly once
        engine2 = create_async_engine(db_url)
        async with engine2.begin() as conn:
            result = await conn.execute(text("PRAGMA table_info(decision_commits)"))
            cols = [row[1] for row in result.fetchall()]
        await engine2.dispose()
        assert cols.count("latency_breakdown") == 1
        assert cols.count("token_usage") == 1
