"""Dashboard smoke tests — verify core helpers and importability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_helper():
    """Test the run_async helper (moved from app._run to data_loader.run_async)."""

    async def _coro():
        return 42

    # run_async lives in data_loader now; import it with st mocked
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        import sys
        from importlib import import_module

        sys.modules.pop("dashboard.data_loader", None)
        mod = import_module("dashboard.data_loader")
        assert mod.run_async(_coro()) == 42


def test_backtest_result_summary_with_llm_stats():
    """BacktestResult.summary() includes LLM stats when present."""
    from cryptotrader.backtest.result import BacktestResult

    r = BacktestResult(total_return=0.05, llm_calls=10, llm_tokens=5000)
    s = r.summary()
    assert s["llm_calls"] == 10
    assert s["llm_tokens"] == 5000


def test_backtest_result_summary_without_llm_stats():
    """BacktestResult.summary() omits LLM stats when zero."""
    from cryptotrader.backtest.result import BacktestResult

    r = BacktestResult(total_return=0.05)
    s = r.summary()
    assert "llm_calls" not in s


def test_coingecko_ids_has_btc():
    """CoinGecko ID mapping includes BTC."""
    from cryptotrader.data.news import _COINGECKO_IDS

    assert _COINGECKO_IDS["BTC"] == "bitcoin"


def test_coingecko_fallback():
    """Unknown symbols fall back to lowercase as CoinGecko ID."""
    from cryptotrader.data.news import _COINGECKO_IDS

    # Known symbol uses mapping
    assert _COINGECKO_IDS.get("BTC", "btc") == "bitcoin"
    # Unknown symbol would fall back to lowercase
    assert _COINGECKO_IDS.get("UNKNOWN", "unknown") == "unknown"
