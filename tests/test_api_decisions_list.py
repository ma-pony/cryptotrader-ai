"""Tests for GET /api/decisions — FR-803.

Paginated list of decision commits with optional pair / from / to filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.infrastructure.database_url = None
    return cfg


@dataclass
class _StubVerdict:
    action: str = "long"
    confidence: float = 0.7
    position_scale: float = 0.5
    reasoning: str = "stub"
    thesis: str = ""
    invalidation: str = ""
    verdict_source: str = "ai"
    divergence: float = 0.0


def _make_commit(idx: int, pair: str = "BTC/USDT", price: float = 65000.0):
    """Stub DecisionCommit with the minimum fields the list endpoint needs."""
    from cryptotrader.models import DecisionCommit, GateResult

    return DecisionCommit(
        hash=f"hash{idx:04d}",
        parent_hash=None,
        timestamp=datetime.now(UTC) - timedelta(minutes=idx),
        pair=pair,
        snapshot_summary={"price": price},
        analyses={},
        debate_rounds=0,
        verdict=_StubVerdict(),  # type: ignore[arg-type]
        risk_gate=GateResult(passed=True),
        order=None,
        trace_id=f"trace-{idx}",
    )


class TestDecisionsListShape:
    def test_returns_200_with_pagination_envelope(self, client: TestClient) -> None:
        commits = [_make_commit(i) for i in range(3)]
        mock_store = MagicMock()
        mock_store.log = AsyncMock(return_value=commits)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions?page=1&size=20")

        assert resp.status_code == 200
        body = resp.json()
        for key in ("items", "total", "page", "size", "has_next"):
            assert key in body, f"missing key: {key}"
        assert body["page"] == 1
        assert body["size"] == 20

    def test_items_have_required_subset_fields(self, client: TestClient) -> None:
        """List response items include only the slim DecisionCommit subset (data-model §2)."""
        commits = [_make_commit(0)]
        mock_store = MagicMock()
        mock_store.log = AsyncMock(return_value=commits)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions")

        body = resp.json()
        assert len(body["items"]) == 1
        item = body["items"][0]
        for key in ("commit_hash", "ts", "pair", "price", "verdict", "is_filled"):
            assert key in item, f"missing key in list item: {key}"
        # Verdict must include action+source per data-model §2 Verdict
        assert item["verdict"]["action"] == "long"
        assert "source" in item["verdict"]

    def test_default_page_and_size(self, client: TestClient) -> None:
        mock_store = MagicMock()
        mock_store.log = AsyncMock(return_value=[])

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions")

        body = resp.json()
        assert body["page"] == 1
        assert body["size"] == 20

    def test_pair_filter_passed_to_store(self, client: TestClient) -> None:
        mock_store = MagicMock()
        mock_store.log = AsyncMock(return_value=[])

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            client.get("/api/decisions?pair=ETH/USDT")

        # JournalStore.log called with pair filter
        kwargs = mock_store.log.call_args.kwargs
        args = mock_store.log.call_args.args
        assert kwargs.get("pair") == "ETH/USDT" or "ETH/USDT" in args


class TestDecisionsListPagination:
    def test_size_capped(self, client: TestClient) -> None:
        """size > 100 must be rejected (per contract sanity)."""
        resp = client.get("/api/decisions?size=10000")
        assert resp.status_code in (400, 422)

    def test_page_must_be_positive(self, client: TestClient) -> None:
        resp = client.get("/api/decisions?page=0")
        assert resp.status_code in (400, 422)

    def test_has_next_true_when_more_pages(self, client: TestClient) -> None:
        commits = [_make_commit(i) for i in range(20)]
        mock_store = MagicMock()
        # Returning exactly `size` items is the cue for has_next=True
        mock_store.log = AsyncMock(return_value=commits)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions?page=1&size=20")

        body = resp.json()
        assert body["has_next"] is True

    def test_has_next_false_when_short_page(self, client: TestClient) -> None:
        commits = [_make_commit(i) for i in range(5)]
        mock_store = MagicMock()
        mock_store.log = AsyncMock(return_value=commits)

        with (
            patch("cryptotrader.config.load_config", return_value=_mock_config()),
            patch("cryptotrader.journal.store.JournalStore", return_value=mock_store),
        ):
            resp = client.get("/api/decisions?page=1&size=20")

        body = resp.json()
        assert body["has_next"] is False
