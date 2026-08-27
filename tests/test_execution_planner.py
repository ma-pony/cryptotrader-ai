"""目标仓位到订单差值的纯规划契约。"""

from __future__ import annotations

import pytest

from cryptotrader.decision.models import TargetPosition
from tests.factories.signal_fusion import context, position, trade_plan


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (position("flat", 0.0, 0.0), TargetPosition("long", 0.5), [("buy", 5.0, False)]),
        (position("long", 3.0, 0.3), TargetPosition("long", 0.8), [("buy", 5.0, False)]),
        (position("long", 8.0, 0.8), TargetPosition("long", 0.3), [("sell", 5.0, True)]),
        (position("short", 8.0, 0.8), TargetPosition("short", 0.3), [("buy", 5.0, True)]),
        (position("long", 8.0, 0.8), TargetPosition("flat", 0.0), [("sell", 8.0, True)]),
        (
            position("short", 4.0, 0.4),
            TargetPosition("long", 0.3),
            [("buy", 4.0, True), ("buy", 3.0, False)],
        ),
    ],
)
def test_planner_creates_position_delta(current, target, expected):
    from cryptotrader.execution.planner import ExecutionPlanner

    result = ExecutionPlanner(max_single_pct=0.1).plan(
        context=context(price=100.0, equity=10_000.0, position=current, market_type="swap"),
        trade_plan=trade_plan(target),
    )

    assert [(item.side, item.amount, item.reduce_only) for item in result.intents] == expected
    assert all(item.pair == "BTC/USDT:USDT" for item in result.intents)


def test_planner_returns_no_intents_when_target_already_matches_position():
    from cryptotrader.execution.planner import ExecutionPlanner

    result = ExecutionPlanner(max_single_pct=0.1).plan(
        context=context(
            price=100.0,
            equity=10_000.0,
            position=position("long", 5.0, 0.5),
            market_type="swap",
        ),
        trade_plan=trade_plan(TargetPosition("long", 0.5)),
    )

    assert result.intents == ()


def test_planner_preserves_exit_prices():
    from cryptotrader.execution.planner import ExecutionPlanner

    plan = trade_plan(TargetPosition("long", 0.5), stop_loss=90.0, take_profit=120.0)
    result = ExecutionPlanner(max_single_pct=0.1).plan(context=context(), trade_plan=plan)

    assert result.stop_loss == 90.0
    assert result.take_profit == 120.0


def test_spot_short_is_rejected():
    from cryptotrader.execution.planner import ExecutionPlanner, ExecutionPlanningError

    with pytest.raises(ExecutionPlanningError, match="spot"):
        ExecutionPlanner(max_single_pct=0.1).plan(
            context=context(
                price=100.0,
                equity=10_000.0,
                position=position("flat", 0, 0),
                market_type="spot",
            ),
            trade_plan=trade_plan(TargetPosition("short", 0.5)),
        )
