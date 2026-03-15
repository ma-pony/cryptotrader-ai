"""Tests for RiskGate per-check exception isolation (task 2.2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.models import CheckResult, TradeVerdict
from cryptotrader.risk.state import RedisStateManager


def _make_gate(checks: list):
    """Build a RiskGate instance with injected checks, bypassing __init__."""
    from cryptotrader.risk.gate import RiskGate

    gate = object.__new__(RiskGate)
    gate._checks = checks
    gate.redis_state = RedisStateManager(None)
    gate._redis_was_configured = False
    return gate


def _passing_check(name: str = "mock_pass") -> MagicMock:
    """Return a mock check that always passes."""
    c = MagicMock()
    c.name = name
    c.evaluate = AsyncMock(return_value=CheckResult(passed=True))
    return c


def _failing_check(name: str = "mock_fail", reason: str = "mock failure") -> MagicMock:
    """Return a mock check that always rejects."""
    c = MagicMock()
    c.name = name
    c.evaluate = AsyncMock(return_value=CheckResult(passed=False, reason=reason))
    return c


def _exploding_check(name: str = "mock_explode", exc: Exception | None = None) -> MagicMock:
    """Return a mock check whose evaluate() raises an exception."""
    c = MagicMock()
    c.name = name
    c.evaluate = AsyncMock(side_effect=exc or RuntimeError("check crashed"))
    return c


# -- Normal pass path --


@pytest.mark.asyncio
async def test_all_checks_pass_returns_passed():
    """All checks pass -> GateResult(passed=True)."""
    gate = _make_gate([_passing_check("a"), _passing_check("b")])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    result = await gate.check(verdict, {})
    assert result.passed


# -- Exception isolation --


@pytest.mark.asyncio
async def test_single_check_exception_does_not_stop_other_checks():
    """When one check raises, subsequent checks must still execute."""
    executed = []

    async def _track_eval(verdict, portfolio):
        executed.append("second")
        return CheckResult(passed=True)

    exploding = _exploding_check("check_a")
    passing = _passing_check("check_b")
    passing.evaluate = AsyncMock(side_effect=_track_eval)

    gate = _make_gate([exploding, passing])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    await gate.check(verdict, {})

    assert "second" in executed, "second check must run after first raises"


@pytest.mark.asyncio
async def test_exception_in_check_causes_gate_to_reject():
    """A check that raises is treated as failed; gate rejects the trade."""
    gate = _make_gate([_exploding_check("volatile_check")])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    result = await gate.check(verdict, {})
    assert not result.passed
    assert result.rejected_by == "volatile_check"


@pytest.mark.asyncio
async def test_exception_check_rejected_by_uses_check_error_reason():
    """Reason field for an exception check must contain 'check_error'."""
    gate = _make_gate([_exploding_check("some_check")])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    result = await gate.check(verdict, {})
    assert "check_error" in result.reason


@pytest.mark.asyncio
async def test_exception_is_logged_as_warning():
    """Exception must be logged with logger.warning(exc_info=True)."""
    gate = _make_gate([_exploding_check("log_check")])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    with patch("cryptotrader.risk.gate.logger") as mock_logger:
        await gate.check(verdict, {})
        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert call_kwargs.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_all_checks_run_even_when_first_throws():
    """All subsequent checks run even when the first one raises."""
    call_log: list[str] = []

    async def _make_eval(name):
        async def _eval(verdict, portfolio):
            call_log.append(name)
            return CheckResult(passed=True)

        return _eval

    c1 = _exploding_check("first")
    c2 = _passing_check("second")
    c2.evaluate = AsyncMock(side_effect=await _make_eval("second"))
    c3 = _passing_check("third")
    c3.evaluate = AsyncMock(side_effect=await _make_eval("third"))

    gate = _make_gate([c1, c2, c3])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    await gate.check(verdict, {})

    assert "second" in call_log
    assert "third" in call_log


@pytest.mark.asyncio
async def test_mixed_exception_and_failure_both_cause_rejection():
    """Exploding check + normal failing check: gate still rejects."""
    gate = _make_gate(
        [
            _exploding_check("crash_check"),
            _failing_check("fail_check", "explicit failure"),
            _passing_check("pass_check"),
        ]
    )
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    result = await gate.check(verdict, {})
    assert not result.passed


@pytest.mark.asyncio
async def test_passing_checks_after_exception_still_run():
    """Checks on both sides of a raising check must execute."""
    executed: list[str] = []

    async def _track(name):
        async def _eval(verdict, portfolio):
            executed.append(name)
            return CheckResult(passed=True)

        return _eval

    c_pass_before = _passing_check("before")
    c_pass_before.evaluate = AsyncMock(side_effect=await _track("before"))

    c_explode = _exploding_check("middle")

    c_pass_after = _passing_check("after")
    c_pass_after.evaluate = AsyncMock(side_effect=await _track("after"))

    gate = _make_gate([c_pass_before, c_explode, c_pass_after])
    verdict = TradeVerdict(action="long", confidence=0.7, position_scale=0.05)
    await gate.check(verdict, {})

    assert "before" in executed
    assert "after" in executed


# -- close action still bypasses checks (not affected by isolation) --


@pytest.mark.asyncio
async def test_close_action_bypasses_all_checks():
    """action=close returns pass immediately without running any checks."""
    gate = _make_gate([_exploding_check("should_not_run")])
    verdict = TradeVerdict(action="close", confidence=0.7, position_scale=0.0)
    result = await gate.check(verdict, {})
    assert result.passed
