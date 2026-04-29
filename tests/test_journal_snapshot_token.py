"""Tests for _snapshot_token_usage in nodes/journal.py."""

from __future__ import annotations

from cryptotrader.llm.token_tracker import set_ledger, start_ledger
from cryptotrader.nodes.journal import _snapshot_token_usage


class TestSnapshotTokenUsage:
    def setup_method(self) -> None:
        set_ledger(None)

    def test_no_active_ledger_returns_empty(self) -> None:
        assert _snapshot_token_usage() == {}

    def test_active_ledger_yields_dict(self) -> None:
        ledger = start_ledger()
        ledger.record(model="gpt-4o", input_tokens=100, output_tokens=50)
        out = _snapshot_token_usage()
        assert out["calls"] == 1.0
        assert out["input_tokens"] == 100.0
        assert out["output_tokens"] == 50.0
        assert "by_model" in out

    def test_empty_ledger_returns_zero_stats(self) -> None:
        start_ledger()
        out = _snapshot_token_usage()
        assert out["calls"] == 0.0
        assert out["cost_usd"] == 0.0
