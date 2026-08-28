"""Book-level risk caps preserve the configured allocation policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot, ConnectionPortfolioSnapshot
from cryptotrader.venues.models import ConnectionPosition


def _book(*weights: float) -> ExecutionBook:
    return ExecutionBook(
        "simulation",
        "Simulation",
        "simulated",
        True,
        False,
        tuple(ConnectionAllocation(f"connection-{index}", True, weight) for index, weight in enumerate(weights)),
    )


def _portfolio(*, equity: str = "100", notionals: tuple[str, ...] = ("0", "0")) -> BookPortfolioSnapshot:
    pair = Pair.parse("BTC/USDT:USDT")
    per_connection_equity = Decimal(equity) / len(notionals)
    connections = tuple(
        ConnectionPortfolioSnapshot(
            f"connection-{index}",
            per_connection_equity,
            {"USDT": per_connection_equity},
            ConnectionPosition(pair, Decimal("0"), Decimal(notional), None),
        )
        for index, notional in enumerate(notionals)
    )
    return BookPortfolioSnapshot(
        "simulation",
        "simulated",
        Decimal(equity),
        sum((Decimal(value) for value in notionals), Decimal("0")),
        connections,
    )


def _request(*, target: str = "0.7", equity: str = "100", peak: str = "100"):
    from cryptotrader.risk.models import BookRiskRequest

    return BookRiskRequest(_book(0.4, 0.6), _portfolio(equity=equity), Decimal(target), Decimal(peak))


def _limits(**overrides):
    from cryptotrader.risk.models import BookRiskLimits

    values = {
        "max_net_exposure": Decimal("1"),
        "max_gross_exposure": Decimal("1"),
        "max_drawdown": Decimal("0.2"),
        "max_connection_concentration": Decimal("1"),
    }
    values.update(overrides)
    return BookRiskLimits(**values)


def test_book_cap_scales_whole_target_without_reweighting_connections():
    from cryptotrader.risk.gate import BookRiskGate

    result = BookRiskGate(_limits(max_net_exposure=Decimal("0.4"))).evaluate(_request())

    assert result.passed is True
    assert result.requested_target_exposure == Decimal("0.7")
    assert result.capped_target_exposure == Decimal("0.4")
    assert result.connection_weights == (Decimal("0.4"), Decimal("0.6"))
    assert tuple(target.target_signed_notional for target in result.connection_targets) == (
        Decimal("16.00"),
        Decimal("24.00"),
    )
    assert result.cap_source == "max_net_exposure"


@pytest.mark.parametrize(
    ("limits", "expected", "source"),
    [
        ({"max_gross_exposure": Decimal("0.3")}, Decimal("0.3"), "max_gross_exposure"),
        (
            {"max_connection_concentration": Decimal("0.24")},
            Decimal("0.4"),
            "max_connection_concentration",
        ),
    ],
)
def test_book_gate_caps_net_gross_and_concentration_at_the_book_target(limits, expected, source):
    from cryptotrader.risk.gate import BookRiskGate

    result = BookRiskGate(_limits(**limits)).evaluate(_request())

    assert result.capped_target_exposure == expected
    assert result.cap_source == source
    assert tuple(item.weight for item in result.connection_targets) == (Decimal("0.4"), Decimal("0.6"))


def test_drawdown_breach_caps_the_whole_book_to_flat_instead_of_blocking_de_risking():
    from cryptotrader.risk.gate import BookRiskGate

    result = BookRiskGate(_limits(max_drawdown=Decimal("0.1"))).evaluate(
        _request(target="0.7", equity="80", peak="100")
    )

    assert result.passed is True
    assert result.capped_target_exposure == Decimal("0")
    assert result.cap_source == "max_drawdown"
    assert tuple(item.target_signed_notional for item in result.connection_targets) == (Decimal("0.0"),) * 2


def test_book_risk_models_are_frozen_and_reject_binary_float_inputs():
    from cryptotrader.risk.models import BookRiskRequest

    request = _request()
    with pytest.raises(FrozenInstanceError):
        request.target_exposure = Decimal("0")
    with pytest.raises(ValueError, match="Decimal"):
        BookRiskRequest(request.book, request.portfolio, 0.5, request.peak_equity)
