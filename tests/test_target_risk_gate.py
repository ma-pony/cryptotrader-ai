"""TargetPosition 风控 Gate 的聚合与失败语义。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cryptotrader.config import RiskConfig
from cryptotrader.decision.models import TargetPosition
from cryptotrader.risk.state import RedisStateManager
from tests.factories.signal_fusion import position, risk_request


@dataclass
class FakeCheck:
    name: str
    result: object | None = None
    error: Exception | None = None
    calls: int = 0

    async def evaluate(self, request, portfolio):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class RedisDown:
    _redis = object()
    _last_ping_error = {"type": "ConnectionError", "msg": "offline"}

    async def ping(self) -> bool:
        return False


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (position("long", 1.0, 0.5), TargetPosition("flat", 0.0), True),
        (position("long", 1.0, 0.5), TargetPosition("long", 0.3), True),
        (position("short", 1.0, 0.5), TargetPosition("short", 0.5), True),
        (position("flat", 0.0, 0.0), TargetPosition("long", 0.3), False),
        (position("long", 1.0, 0.5), TargetPosition("long", 0.8), False),
        (position("long", 1.0, 0.5), TargetPosition("short", 0.3), False),
    ],
)
def test_risk_request_identifies_exposure_reduction(current, target, expected):
    request = risk_request(current=current, target=target)

    assert request.reduces_exposure is expected


@pytest.mark.asyncio
async def test_flat_target_bypasses_unavailable_redis_and_checks():
    from cryptotrader.risk.gate import RiskGate

    exploding = FakeCheck("must_not_run", error=RuntimeError("boom"))
    gate = RiskGate(RiskConfig(), RedisDown(), checks=[exploding])
    request = risk_request(
        current=position("long", 1.0, 0.5),
        target=TargetPosition("flat", 0.0),
    )

    result = await gate.check(request, {"total_value": 10_000.0})

    assert result.passed is True
    assert result.plan.target == TargetPosition("flat", 0.0)
    assert exploding.calls == 0


@pytest.mark.asyncio
async def test_gate_applies_strictest_size_ratio_cap_without_mutating_request():
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.models import RiskCheckResult

    gate = RiskGate(
        RiskConfig(),
        RedisStateManager(None),
        checks=[
            FakeCheck("cap_40", RiskCheckResult(True, size_ratio_cap=0.4)),
            FakeCheck("cap_25", RiskCheckResult(True, size_ratio_cap=0.25)),
        ],
    )
    request = risk_request(
        current=position("flat", 0.0, 0.0),
        target=TargetPosition("long", 0.8),
    )

    result = await gate.check(request, {"total_value": 10_000.0})

    assert result.passed is True
    assert result.plan.target == TargetPosition("long", 0.25)
    assert request.plan.target == TargetPosition("long", 0.8)


@pytest.mark.asyncio
async def test_zero_cap_flattens_target_and_records_cap_provenance():
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.models import RiskCheckResult

    gate = RiskGate(
        RiskConfig(),
        RedisStateManager(None),
        checks=[FakeCheck("max_position", RiskCheckResult(True, reason="no exposure allowed", size_ratio_cap=0.0))],
    )

    result = await gate.check(risk_request(target=TargetPosition("short", 0.8)), {"total_value": 10_000.0})

    assert result.passed is True
    assert result.plan.target == TargetPosition("flat", 0.0)
    assert result.cap_source == "max_position"
    assert result.reason == "no exposure allowed"


@pytest.mark.asyncio
async def test_gate_returns_first_rejection_but_runs_all_checks():
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.models import RiskCheckResult

    first = FakeCheck("first", RiskCheckResult(False, reason="blocked first"))
    second = FakeCheck("second", RiskCheckResult(False, reason="blocked second"))
    gate = RiskGate(RiskConfig(), RedisStateManager(None), checks=[first, second])

    result = await gate.check(risk_request(), {"total_value": 10_000.0})

    assert result.passed is False
    assert result.rejected_by == "first"
    assert result.reason == "blocked first"
    assert (first.calls, second.calls) == (1, 1)


@pytest.mark.asyncio
async def test_gate_turns_check_exception_into_rejection_and_continues():
    from cryptotrader.risk.gate import RiskGate
    from cryptotrader.risk.models import RiskCheckResult

    exploding = FakeCheck("explode", error=RuntimeError("offline"))
    passing = FakeCheck("passing", RiskCheckResult(True))
    gate = RiskGate(RiskConfig(), RedisStateManager(None), checks=[exploding, passing])

    result = await gate.check(risk_request(), {"total_value": 10_000.0})

    assert result.passed is False
    assert result.rejected_by == "explode"
    assert result.reason == "check_error: explode raised an unexpected exception"
    assert passing.calls == 1


@pytest.mark.asyncio
async def test_gate_rejects_new_exposure_when_configured_redis_is_down():
    from cryptotrader.risk.gate import RiskGate

    gate = RiskGate(RiskConfig(), RedisDown(), checks=[])

    result = await gate.check(risk_request(), {"total_value": 10_000.0})

    assert result.passed is False
    assert result.rejected_by == "redis_unavailable"
    assert "ConnectionError: offline" in result.reason
