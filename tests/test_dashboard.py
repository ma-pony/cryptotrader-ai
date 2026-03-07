"""Dashboard smoke tests — verify core helpers and importability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_run_helper():
    """Test the _run async helper executes coroutines correctly."""

    async def _coro():
        return 42

    # Import the helper by importing the module with st mocked
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        from importlib import import_module

        mod = import_module("dashboard.app")
        assert mod._run(_coro()) == 42


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


def test_prewarm_finbert():
    """prewarm_finbert returns bool."""
    from cryptotrader.data.news import prewarm_finbert

    result = prewarm_finbert()
    assert isinstance(result, bool)


def test_coingecko_fallback():
    """Unknown symbols fall back to lowercase as CoinGecko ID."""
    from cryptotrader.data.news import _COINGECKO_IDS

    # Known symbol uses mapping
    assert _COINGECKO_IDS.get("BTC", "btc") == "bitcoin"
    # Unknown symbol would fall back to lowercase
    assert _COINGECKO_IDS.get("UNKNOWN", "unknown") == "unknown"
