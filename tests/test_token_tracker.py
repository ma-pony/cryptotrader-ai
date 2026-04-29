"""Tests for cryptotrader.llm.token_tracker — cost accounting + ContextVar semantics."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cryptotrader.llm.token_tracker import (
    MODEL_COSTS,
    TokenLedger,
    TokenTrackerCallback,
    _match_cost,
    current_ledger,
    default_callback,
    set_ledger,
    start_ledger,
)


class TestMatchCost:
    def test_exact_match_returns_table_value(self) -> None:
        cost = _match_cost("claude-sonnet-4-6")
        assert cost == MODEL_COSTS["claude-sonnet-4-6"]

    def test_prefix_match_strips_suffix(self) -> None:
        # Models often get date/version suffixes.
        cost = _match_cost("claude-sonnet-4-5-20250929-foo")
        assert cost == MODEL_COSTS["claude-sonnet-4-5-20250929"]

    def test_unknown_model_returns_zero(self) -> None:
        assert _match_cost("totally-made-up-model-xyz") == (0.0, 0.0)

    def test_longest_prefix_wins(self) -> None:
        """Regression: gpt-4o-mini-YYYY must NOT match gpt-4o (17x cheaper).

        Dict insertion order previously caused the shorter 'gpt-4o' prefix to match
        first because it appears before 'gpt-4o-mini' in MODEL_COSTS. Fix sorts by
        length descending so the longest matching prefix always wins.
        """
        mini_cost = MODEL_COSTS["gpt-4o-mini"]
        full_cost = MODEL_COSTS["gpt-4o"]
        assert mini_cost != full_cost  # safety for the test premise
        # Suffixed model name — must resolve to mini, not full
        resolved = _match_cost("gpt-4o-mini-2025-04-16")
        assert resolved == mini_cost, f"Expected {mini_cost}, got {resolved} (matched wrong prefix)"

    def test_cached_after_first_resolution(self) -> None:
        """Second lookup of same unknown model should short-circuit via _RESOLVED_COSTS."""
        from cryptotrader.llm import token_tracker as tt

        # Clear cache for determinism
        tt._RESOLVED_COSTS.clear()
        model = "custom-suffixed-gpt-4o-20260101"
        first = _match_cost(model)
        assert model in tt._RESOLVED_COSTS
        second = _match_cost(model)
        assert first == second


class TestTokenLedger:
    def test_record_accumulates_counts(self) -> None:
        ledger = TokenLedger()
        ledger.record(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)
        ledger.record(model="gpt-4o-mini", input_tokens=2000, output_tokens=800)
        assert ledger.input_tokens == 3000
        assert ledger.output_tokens == 1300
        assert ledger.calls == 2
        assert ledger.cache_hits == 0

    def test_record_computes_cost_per_model(self) -> None:
        ledger = TokenLedger()
        # gpt-4o-mini = $0.15 / $0.60 per 1M
        ledger.record(model="gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        assert ledger.cost_usd == pytest.approx(0.75)

    def test_record_tracks_per_model_breakdown(self) -> None:
        ledger = TokenLedger()
        ledger.record(model="gpt-4o", input_tokens=500, output_tokens=250)
        ledger.record(model="gpt-4o-mini", input_tokens=2000, output_tokens=1000)
        assert set(ledger.by_model.keys()) == {"gpt-4o", "gpt-4o-mini"}
        assert ledger.by_model["gpt-4o"]["input"] == 500
        assert ledger.by_model["gpt-4o-mini"]["calls"] == 1

    def test_cache_hit_increments_counter(self) -> None:
        ledger = TokenLedger()
        ledger.record(model="claude-sonnet-4-6", input_tokens=100, output_tokens=50, cache_hit=True)
        assert ledger.cache_hits == 1

    def test_to_dict_serializes_fully(self) -> None:
        ledger = TokenLedger()
        ledger.record(model="gpt-4o", input_tokens=100, output_tokens=50)
        out = ledger.to_dict()
        assert set(out.keys()) == {
            "input_tokens",
            "output_tokens",
            "cache_hits",
            "calls",
            "cost_usd",
            "by_model",
        }
        assert out["by_model"]["gpt-4o"]["calls"] == 1.0

    def test_unknown_model_yields_zero_cost(self) -> None:
        ledger = TokenLedger()
        ledger.record(model="mystery-model-1.0", input_tokens=10_000, output_tokens=5_000)
        assert ledger.cost_usd == 0.0
        assert ledger.input_tokens == 10_000


class TestLedgerContextVar:
    def setup_method(self) -> None:
        set_ledger(None)

    def test_start_ledger_binds_fresh(self) -> None:
        ledger = start_ledger()
        assert current_ledger() is ledger
        assert ledger.calls == 0

    def test_start_ledger_replaces_existing(self) -> None:
        first = start_ledger()
        first.record(model="gpt-4o", input_tokens=1, output_tokens=1)
        second = start_ledger()
        assert current_ledger() is second
        assert second.calls == 0
        assert first.calls == 1  # old ledger still retains its data


class TestTokenTrackerCallback:
    def setup_method(self) -> None:
        set_ledger(None)

    def test_records_usage_from_llm_result(self) -> None:
        ledger = start_ledger()
        cb = TokenTrackerCallback()
        msg = AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        cb.on_llm_end(result)
        assert ledger.input_tokens == 500
        assert ledger.output_tokens == 200
        assert ledger.calls == 1

    def test_no_ledger_silently_skips(self) -> None:
        cb = TokenTrackerCallback()
        msg = AIMessage(
            content="x",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        # Must not raise when ledger is unset.
        cb.on_llm_end(result)

    def test_zero_tokens_no_record(self) -> None:
        ledger = start_ledger()
        cb = TokenTrackerCallback()
        msg = AIMessage(
            content="",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        result = LLMResult(generations=[[ChatGeneration(message=msg)]])
        cb.on_llm_end(result)
        assert ledger.calls == 0

    def test_default_callback_returns_singleton(self) -> None:
        cb1 = default_callback()
        cb2 = default_callback()
        assert cb1 is cb2
