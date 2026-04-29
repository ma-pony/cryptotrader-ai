"""Tests for debate/verdict.py helper functions."""

from __future__ import annotations

import pytest

from cryptotrader.debate.verdict import (
    _extract_json,
    _format_constraints,
    _format_funding_rate,
    _format_position_context,
    _format_trend_context,
    _normalize_action,
)


class TestNormalizeAction:
    def test_valid_actions_pass_through(self):
        for action in ("long", "short", "hold", "close"):
            assert _normalize_action(action) == action

    def test_buy_maps_to_long(self):
        assert _normalize_action("buy") == "long"
        assert _normalize_action("bullish") == "long"

    def test_sell_maps_to_short(self):
        assert _normalize_action("sell") == "short"
        assert _normalize_action("bearish") == "short"

    def test_exit_maps_to_close(self):
        assert _normalize_action("exit") == "close"
        assert _normalize_action("flatten") == "close"
        assert _normalize_action("close_position") == "close"

    def test_unknown_maps_to_hold(self):
        assert _normalize_action("unknown") == "hold"
        assert _normalize_action("") == "hold"

    def test_case_insensitive(self):
        assert _normalize_action("LONG") == "long"
        assert _normalize_action("BUY") == "long"

    def test_strips_whitespace(self):
        assert _normalize_action("  short  ") == "short"


class TestExtractJson:
    def test_simple_json(self):
        result = _extract_json('{"action": "long"}')
        assert result == {"action": "long"}

    def test_json_with_surrounding_text(self):
        result = _extract_json('The analysis shows: {"action": "hold"} based on data')
        assert result == {"action": "hold"}

    def test_nested_json(self):
        result = _extract_json('{"a": {"b": 1}, "c": 2}')
        assert result == {"a": {"b": 1}, "c": 2}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json("no json here")

    def test_unbalanced_braces_raises(self):
        with pytest.raises(ValueError, match="Unbalanced braces"):
            _extract_json('{"a": 1')


class TestFormatFundingRate:
    def test_normal_rate(self):
        result = _format_funding_rate(0.0001)
        assert "0.000100" in result
        assert "ELEVATED" not in result
        assert "NEGATIVE" not in result

    def test_elevated_rate(self):
        result = _format_funding_rate(0.001)
        assert "ELEVATED" in result
        assert "crowded long" in result

    def test_negative_rate(self):
        result = _format_funding_rate(-0.001)
        assert "NEGATIVE" in result
        assert "crowded short" in result


class TestFormatConstraints:
    def test_empty_constraints(self):
        assert _format_constraints({}) == "No risk constraints available."

    def test_max_position(self):
        result = _format_constraints({"max_position_pct": 0.25})
        assert "25%" in result

    def test_drawdown(self):
        result = _format_constraints({"drawdown_current": 0.05})
        assert "5.0%" in result

    def test_daily_loss_remaining(self):
        result = _format_constraints({"daily_loss_remaining_pct": 0.02})
        assert "2.0%" in result
        assert "EXHAUSTED" not in result

    def test_daily_loss_exhausted(self):
        result = _format_constraints({"daily_loss_remaining_pct": 0.0})
        assert "EXHAUSTED" in result

    def test_cooldown_pairs(self):
        result = _format_constraints({"cooldown_pairs": ["BTC/USDT", "ETH/USDT"]})
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result

    def test_circuit_breaker(self):
        result = _format_constraints({"circuit_breaker_active": True})
        assert "CIRCUIT BREAKER ACTIVE" in result

    def test_funding_rate_in_constraints(self):
        result = _format_constraints({"funding_rate": 0.001})
        assert "ELEVATED" in result

    def test_volatility(self):
        result = _format_constraints({"volatility": 0.0234})
        assert "0.0234" in result

    def test_remaining_exposure(self):
        result = _format_constraints({"remaining_exposure_pct": 0.5})
        assert "50%" in result


class TestFormatPositionContext:
    def test_none_position(self):
        result = _format_position_context(None)
        assert "FLAT" in result

    def test_empty_dict(self):
        result = _format_position_context({})
        assert "FLAT" in result

    def test_flat_with_last_action(self):
        ctx = {"side": "flat", "last_action_context": "Closed at profit"}
        result = _format_position_context(ctx)
        assert "FLAT" in result
        assert "Closed at profit" in result

    def test_long_position(self):
        ctx = {"side": "long", "entry_price": 50000, "current_price": 55000, "days_held": 3}
        result = _format_position_context(ctx)
        assert "LONG" in result
        assert "$50,000" in result
        assert "$55,000" in result
        assert "+10.0%" in result
        assert "3" in result

    def test_short_position(self):
        ctx = {"side": "short", "entry_price": 50000, "current_price": 45000, "days_held": 2}
        result = _format_position_context(ctx)
        assert "SHORT" in result
        assert "+10.0%" in result

    def test_position_with_last_action(self):
        ctx = {
            "side": "long",
            "entry_price": 50000,
            "current_price": 52000,
            "days_held": 1,
            "last_action_context": "Strong trend entry",
        }
        result = _format_position_context(ctx)
        assert "Strong trend entry" in result

    def test_zero_prices(self):
        ctx = {"side": "long", "entry_price": 0, "current_price": 0, "days_held": 0}
        result = _format_position_context(ctx)
        assert "LONG" in result


class TestFormatTrendContext:
    def test_none(self):
        result = _format_trend_context(None)
        assert "unavailable" in result

    def test_with_changes(self):
        ctx = {"change_7d": 0.05, "change_14d": 0.12, "change_30d": -0.03}
        result = _format_trend_context(ctx)
        assert "+5.0%" in result
        assert "+12.0%" in result
        assert "-3.0%" in result

    def test_with_range(self):
        ctx = {"high_30d": 70000, "low_30d": 50000, "current_price": 60000}
        result = _format_trend_context(ctx)
        assert "$50,000" in result
        assert "$70,000" in result
        assert "50%" in result

    def test_partial_data(self):
        ctx = {"change_7d": 0.03}
        result = _format_trend_context(ctx)
        assert "+3.0%" in result
