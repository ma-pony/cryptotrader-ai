"""RiskGate 必须隔离单个检查异常, 同时完整执行检查链。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.config import RiskConfig
from cryptotrader.risk.models import RiskCheckResult
from cryptotrader.risk.state import RedisStateManager
from tests.factories.signal_fusion import risk_request


def _check(name: str, result=None, error: Exception | None = None) -> MagicMock:
    check = MagicMock()
    check.name = name
    check.evaluate = AsyncMock(
        return_value=result or RiskCheckResult(passed=True),
        side_effect=error,
    )
    return check


def _gate(checks: list):
    from cryptotrader.risk.gate import RiskGate

    return RiskGate(RiskConfig(), RedisStateManager(None), checks=checks)


@pytest.mark.asyncio
async def test_all_checks_pass():
    result = await _gate([_check("a"), _check("b")]).check(risk_request(), {})

    assert result.passed


@pytest.mark.asyncio
async def test_exception_rejects_but_does_not_stop_later_checks():
    exploding = _check("volatile_check", error=RuntimeError("check crashed"))
    later = _check("later")

    result = await _gate([exploding, later]).check(risk_request(), {})

    assert not result.passed
    assert result.rejected_by == "volatile_check"
    assert result.reason.startswith("check_error:")
    later.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_exception_is_logged_with_traceback():
    gate = _gate([_check("log_check", error=RuntimeError("check crashed"))])

    with patch("cryptotrader.risk.gate.logger") as logger:
        await gate.check(risk_request(), {})

    logger.warning.assert_called_once()
    assert logger.warning.call_args.kwargs["exc_info"] is True


@pytest.mark.asyncio
async def test_flat_target_bypasses_checks():
    from cryptotrader.decision.models import TargetPosition

    check = _check("should_not_run", error=RuntimeError("must not run"))

    result = await _gate([check]).check(risk_request(target=TargetPosition("flat", 0.0)), {})

    assert result.passed
    check.evaluate.assert_not_awaited()
