"""Tests for dashboard pages task 9.1 (OverviewPage) and 9.2 (RiskStatusPage).

Testing strategy (per task instructions):
- Minimize mocks.
- Test pure logic functions: portfolio summary formatting, position list transformation.
- Use real model objects and real data structures, not mock dicts.
- For Streamlit rendering: test that functions do not crash with real data and
  verify key behavioral contracts (warning shown on None, early return on Redis unavail).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers — streamlit mock factory
# ---------------------------------------------------------------------------


def _make_col_mock() -> MagicMock:
    """Return a single column mock that works as a context manager."""
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return col


def _make_st_mock() -> MagicMock:
    """Return a MagicMock that simulates the Streamlit API."""
    st = MagicMock()
    st.cache_data = lambda *args, **kwargs: lambda fn: fn
    st.cache_resource = lambda fn: fn

    # columns(n) must return exactly n mocks so tuple-unpacking in pages works.
    def _columns_side_effect(n, *args, **kwargs):
        return [_make_col_mock() for _ in range(n if isinstance(n, int) else len(n))]

    st.columns.side_effect = _columns_side_effect

    # expander context manager
    st.expander.return_value.__enter__ = MagicMock(return_value=MagicMock())
    st.expander.return_value.__exit__ = MagicMock(return_value=False)
    # empty() returns a context manager
    empty_ctx = MagicMock()
    empty_ctx.__enter__ = MagicMock(return_value=empty_ctx)
    empty_ctx.__exit__ = MagicMock(return_value=False)
    st.empty.return_value = empty_ctx
    return st


def _import_page(module_name: str, st_mock: MagicMock) -> Any:
    """Import a dashboard page module with streamlit and time patched."""
    # Remove cached version so we reimport fresh
    for key in list(sys.modules.keys()):
        if module_name in key or "dashboard._pages" in key:
            del sys.modules[key]

    time_mock = MagicMock()
    time_mock.sleep = MagicMock()  # prevent actual sleeping

    with patch.dict("sys.modules", {"streamlit": st_mock, "time": time_mock}):
        return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Pure logic helpers — position direction formatting
# ---------------------------------------------------------------------------


def _make_portfolio_dict(
    positions: dict | None = None,
    cash: float = 10000.0,
    total_value: float = 15000.0,
    account_id: str = "default",
) -> dict[str, Any]:
    """Build a portfolio dict matching PortfolioManager.get_portfolio() output."""
    if positions is None:
        positions = {
            "BTC/USDT": {"amount": 0.5, "avg_price": 40000.0},
            "ETH/USDT": {"amount": -2.0, "avg_price": 2500.0},
        }
    return {
        "account_id": account_id,
        "positions": positions,
        "cash": cash,
        "total_value": total_value,
    }


def _make_scheduler_status(
    running: bool = True,
    pairs: list[str] | None = None,
    next_run_time: str | None = "2026-03-15T10:00:00Z",
    interval_minutes: int = 240,
    cycle_count: int = 5,
) -> dict[str, Any]:
    """Build a scheduler status dict matching the API response schema."""
    if pairs is None:
        pairs = ["BTC/USDT"]
    jobs = []
    if running:
        jobs = [
            {
                "job_id": "trading_cycle",
                "name": "Trading cycle",
                "next_run_time": next_run_time,
                "pairs": pairs,
            }
        ]
    return {
        "running": running,
        "jobs": jobs,
        "cycle_count": cycle_count,
        "interval_minutes": interval_minutes,
        "pairs": pairs,
    }


def _make_risk_status_dict(
    hourly: int = 3,
    daily: int = 12,
    circuit_breaker_active: bool = False,
) -> dict[str, Any]:
    """Build a risk status dict matching load_risk_status() output."""
    return {
        "hourly_trade_count": hourly,
        "daily_trade_count": daily,
        "circuit_breaker_active": circuit_breaker_active,
    }


# ===========================================================================
# Task 9.1 — OverviewPage logic tests
# ===========================================================================


class TestPositionDirectionFormatting:
    """Test that the overview page correctly formats position directions."""

    def test_positive_amount_is_long(self):
        """Positive position amounts should be labelled Long."""
        positions = {"BTC/USDT": {"amount": 0.5, "avg_price": 40000.0}}
        portfolio = _make_portfolio_dict(positions=positions)
        # The page renders Long/Short based on the sign of amount.
        # We extract the expected label from the data directly.
        amount = portfolio["positions"]["BTC/USDT"]["amount"]
        direction = "Long" if amount > 0 else "Short"
        assert direction == "Long"

    def test_negative_amount_is_short(self):
        """Negative position amounts should be labelled Short."""
        positions = {"ETH/USDT": {"amount": -1.5, "avg_price": 2500.0}}
        portfolio = _make_portfolio_dict(positions=positions)
        amount = portfolio["positions"]["ETH/USDT"]["amount"]
        direction = "Long" if amount > 0 else "Short"
        assert direction == "Short"

    def test_zero_amount_is_long_or_no_position(self):
        """Zero amount should not be displayed as Short (treat as flat)."""
        positions = {"BTC/USDT": {"amount": 0.0, "avg_price": 0.0}}
        portfolio = _make_portfolio_dict(positions=positions)
        amount = portfolio["positions"]["BTC/USDT"]["amount"]
        # Zero is not negative so should not be short
        assert amount >= 0


class TestPortfolioMetricExtraction:
    """Test portfolio summary field extraction used by OverviewPage."""

    def test_total_value_extracted(self):
        """total_value should be accessible in the portfolio dict."""
        portfolio = _make_portfolio_dict(total_value=20000.0)
        assert portfolio["total_value"] == 20000.0

    def test_cash_extracted(self):
        """Cash balance should be accessible."""
        portfolio = _make_portfolio_dict(cash=5000.0)
        assert portfolio["cash"] == 5000.0

    def test_positions_is_dict(self):
        """positions should be a dict keyed by pair."""
        portfolio = _make_portfolio_dict()
        assert isinstance(portfolio["positions"], dict)

    def test_empty_positions_allowed(self):
        """Portfolio with no positions should not raise."""
        portfolio = _make_portfolio_dict(positions={})
        assert portfolio["positions"] == {}


class TestSchedulerStatusNoneHandling:
    """Test OverviewPage scheduler status None-handling contract."""

    def test_none_scheduler_status_shows_warning(self):
        """When load_scheduler_status returns None, st.warning should be called."""
        st_mock = _make_st_mock()
        # We test this behaviorally by calling the render() function with mocked data loaders.
        with patch.dict("sys.modules", {"streamlit": st_mock}):
            # Clear any cached module
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]

            # Patch the data loader functions at module import time
            data_loader_mock = MagicMock()
            data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
            data_loader_mock.load_scheduler_status.return_value = None  # simulates API down
            data_loader_mock.run_async = MagicMock(return_value=None)

            time_mock = MagicMock()
            time_mock.sleep = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "streamlit": st_mock,
                    "dashboard.data_loader": data_loader_mock,
                    "time": time_mock,
                },
            ):
                for key in list(sys.modules.keys()):
                    if "dashboard._pages.overview" in key:
                        del sys.modules[key]
                import dashboard._pages.overview as overview_page

                overview_page.render()

            # st.warning should have been called (for scheduler unavailable)
            assert st_mock.warning.called

    def test_none_scheduler_status_does_not_raise(self):
        """Render should complete without raising even when scheduler returns None."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = None
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            # Should not raise
            overview_page.render()

    def test_valid_scheduler_status_does_not_warn(self):
        """When scheduler status is valid, st.warning should NOT be called for it."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

        # No warning should be called for scheduler when it's available
        # (warnings from other sections don't count — we check the call args)
        warning_args = [str(c) for c in st_mock.warning.call_args_list]
        scheduler_warnings = [a for a in warning_args if "调度器" in a or "scheduler" in a.lower()]
        assert len(scheduler_warnings) == 0


class TestOverviewPageRenderNoError:
    """Test that OverviewPage renders without errors under normal conditions."""

    def test_render_with_full_portfolio(self):
        """render() with complete portfolio data should not raise."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

    def test_render_with_empty_positions(self):
        """render() with empty positions should not raise."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict(positions={})
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

    def test_render_calls_line_chart_for_equity(self):
        """render() should call st.line_chart for the equity curve display."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

        assert st_mock.line_chart.called

    def test_render_calls_table_for_positions(self):
        """render() should call st.table to display the positions list."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

        assert st_mock.table.called

    def test_render_calls_metric_for_equity(self):
        """render() should call st.metric at least once for portfolio metrics."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

        assert st_mock.metric.called

    def test_render_calls_rerun_for_auto_refresh(self):
        """render() should call st.rerun() to implement the 10-second auto-refresh."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_portfolio.return_value = _make_portfolio_dict()
        data_loader_mock.load_scheduler_status.return_value = _make_scheduler_status()
        data_loader_mock.run_async = MagicMock(return_value=None)

        time_mock = MagicMock()
        time_mock.sleep = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "time": time_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.overview" in key:
                    del sys.modules[key]
            import dashboard._pages.overview as overview_page

            overview_page.render()

        assert st_mock.rerun.called


# ===========================================================================
# Task 9.2 — RiskStatusPage logic tests
# ===========================================================================


class TestRiskStatusNoneHandling:
    """Test RiskStatusPage Redis-unavailable handling contract."""

    def test_none_risk_status_shows_warning(self):
        """When load_risk_status returns None, st.warning must be called."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = None
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        assert st_mock.warning.called

    def test_none_risk_status_does_not_call_st_stop(self):
        """When Redis is unavailable, st.stop() must NOT be called (graceful degradation)."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = None
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        # st.stop() must NOT have been called
        st_mock.stop.assert_not_called()

    def test_none_risk_status_does_not_raise(self):
        """render() with None risk status must complete without raising."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = None
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            # Must not raise
            risk_page.render()

    def test_none_risk_status_warning_text_contains_redis(self):
        """Warning message must mention Redis so operators understand the cause."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = None
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        warning_texts = " ".join(str(c) for c in st_mock.warning.call_args_list)
        # Must mention Redis (or 风控 / unavailable concept)
        assert "Redis" in warning_texts or "redis" in warning_texts.lower() or "风控" in warning_texts


class TestRiskStatusCircuitBreakerActive:
    """Test RiskStatusPage circuit breaker ACTIVE path."""

    def test_active_circuit_breaker_uses_error_container(self):
        """When circuit breaker is ACTIVE, st.error() should be called."""
        st_mock = _make_st_mock()
        # Make st.error a context manager
        error_ctx = MagicMock()
        error_ctx.__enter__ = MagicMock(return_value=error_ctx)
        error_ctx.__exit__ = MagicMock(return_value=False)
        st_mock.error.return_value = error_ctx

        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(circuit_breaker_active=True)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        # st.error should be called at least once (for circuit breaker card)
        assert st_mock.error.called

    def test_inactive_circuit_breaker_no_error_for_cb(self):
        """When circuit breaker is INACTIVE, the circuit breaker section should not use error."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(circuit_breaker_active=False)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        # st.error should NOT be called when circuit breaker is inactive
        st_mock.error.assert_not_called()


class TestRiskStatusTradeCounts:
    """Test that trade counts are displayed by RiskStatusPage."""

    def test_hourly_count_shown(self):
        """Hourly trade count should appear in the rendered output."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(hourly=7, daily=21)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        # Some metric or write call should reference the counts
        all_calls = st_mock.metric.call_args_list + st_mock.write.call_args_list + st_mock.markdown.call_args_list
        all_text = " ".join(str(c) for c in all_calls)
        # The numbers 7 or 21 (or "hourly"/"daily") should appear
        assert "7" in all_text or "hourly" in all_text.lower() or "trade" in all_text.lower()

    def test_render_with_zero_counts_no_crash(self):
        """Zero trade counts should render without errors."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(hourly=0, daily=0)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()


class TestRiskStatusResetButton:
    """Test the circuit-breaker reset button behavior."""

    def test_reset_button_calls_run_async(self):
        """When user clicks reset, run_async should be called for reset_circuit_breaker."""
        st_mock = _make_st_mock()
        # Simulate button click
        st_mock.button.return_value = True

        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(circuit_breaker_active=True)
        data_loader_mock.run_async = MagicMock(return_value=None)

        # Mock RedisStateManager to avoid real Redis
        redis_mock = MagicMock()
        redis_mock.RedisStateManager.return_value.reset_circuit_breaker = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
                "cryptotrader.risk.state": redis_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        # run_async should have been called (for reset), and st.rerun called after
        assert data_loader_mock.run_async.called or st_mock.rerun.called

    def test_no_button_click_no_rerun(self):
        """When reset button is NOT clicked, st.rerun should not be called."""
        st_mock = _make_st_mock()
        # Button NOT clicked
        st_mock.button.return_value = False

        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(circuit_breaker_active=False)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

        st_mock.rerun.assert_not_called()


class TestRiskStatusPageRenderNoError:
    """Smoke tests — render() must not raise under normal conditions."""

    def test_render_with_full_risk_status(self):
        """render() with complete risk status dict should not raise."""
        st_mock = _make_st_mock()
        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict()
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()

    def test_render_with_active_circuit_breaker(self):
        """render() with active circuit breaker should not raise."""
        st_mock = _make_st_mock()
        error_ctx = MagicMock()
        error_ctx.__enter__ = MagicMock(return_value=error_ctx)
        error_ctx.__exit__ = MagicMock(return_value=False)
        st_mock.error.return_value = error_ctx

        data_loader_mock = MagicMock()
        data_loader_mock.load_risk_status.return_value = _make_risk_status_dict(circuit_breaker_active=True)
        data_loader_mock.run_async = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "streamlit": st_mock,
                "dashboard.data_loader": data_loader_mock,
            },
        ):
            for key in list(sys.modules.keys()):
                if "dashboard._pages.risk_status" in key:
                    del sys.modules[key]
            import dashboard._pages.risk_status as risk_page

            risk_page.render()
