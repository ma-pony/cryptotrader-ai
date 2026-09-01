"""Connection preflight rules distinguish risk increase from safe reduction."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cryptotrader.execution.models import ConnectionTarget
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.venues.models import (
    ConnectionPosition,
    OpenVenueState,
    VenueCapabilities,
    VenueQuote,
)


def _capabilities(
    *,
    market_types=frozenset({"swap"}),
    native_protection: bool = True,
    reduce_only: bool = True,
) -> VenueCapabilities:
    return VenueCapabilities(market_types, native_protection, False, reduce_only, frozenset({"market"}))


def _request(
    *,
    pair: Pair | None = None,
    current: str = "0",
    target: str = "5000",
    equity: str = "10000",
    available: bool = True,
    capabilities: VenueCapabilities | None = None,
):
    from cryptotrader.risk.models import ConnectionRiskRequest

    pair = pair or Pair.parse("BTC/USDT:USDT")
    position = ConnectionPosition(pair, Decimal("0"), Decimal(current), None)
    portfolio = ConnectionPortfolioSnapshot("connection", Decimal(equity), {pair.quote: Decimal(equity)}, position)
    from dataclasses import replace

    from tests.fakes.account_session import account_from_portfolio

    portfolio = replace(portfolio, account_snapshot=account_from_portfolio(portfolio))
    connection_target = ConnectionTarget(
        "book",
        "connection",
        Decimal("1"),
        Decimal("10000"),
        Decimal(target) / Decimal("10000"),
        Decimal(target),
    )
    if not available:
        return ConnectionRiskRequest(connection_target, portfolio, False, None, None, None)
    quote = VenueQuote(pair, Decimal("99"), Decimal("101"), Decimal("100"))
    state = OpenVenueState(position, (), ())
    return ConnectionRiskRequest(
        connection_target,
        portfolio,
        True,
        quote,
        state,
        capabilities or _capabilities(market_types=frozenset({pair.market_type})),
    )


def _gate(max_margin: str = "0.6"):
    from cryptotrader.risk.gate import ConnectionRiskGate
    from cryptotrader.risk.models import ConnectionRiskLimits

    return ConnectionRiskGate(ConnectionRiskLimits(max_margin_fraction=Decimal(max_margin)))


def test_derivative_risk_increase_requires_native_protection():
    result = _gate().evaluate(_request(capabilities=_capabilities(native_protection=False)))

    assert result.passed is False
    assert result.reason == "native protection required for derivative risk increase"
    assert result.risk_increase is True


def test_derivative_reduction_ignores_equity_margin_and_native_protection_but_requires_reduce_only():
    reducing = _request(
        current="5000",
        target="0",
        equity="0",
        capabilities=_capabilities(native_protection=False, reduce_only=True),
    )

    assert _gate(max_margin="0.01").evaluate(reducing).passed is True
    rejected = _gate().evaluate(
        _request(
            current="5000",
            target="0",
            equity="0",
            capabilities=_capabilities(native_protection=False, reduce_only=False),
        )
    )
    assert rejected.reason == "reduce-only capability required for derivative risk reduction"


def test_connection_gate_checks_availability_market_equity_and_margin_in_deterministic_order():
    assert _gate().evaluate(_request(available=False)).reason == "connection unavailable"
    assert _gate().evaluate(_request(capabilities=_capabilities(market_types=frozenset({"spot"})))).reason == (
        "market type unsupported by connection"
    )
    assert _gate().evaluate(_request(equity="0")).reason == "positive equity required for risk increase"
    assert _gate(max_margin="0.4").evaluate(_request()).reason == "insufficient margin for risk increase"


def test_spot_short_increase_is_rejected_but_selling_an_existing_spot_position_is_safe():
    spot = Pair.parse("BTC/USDT")

    short = _gate().evaluate(_request(pair=spot, target="-1000"))
    reduction = _gate().evaluate(_request(pair=spot, current="1000", target="0"))

    assert short.reason == "spot markets do not support short exposure"
    assert reduction.passed is True


def test_connection_request_rejects_stale_open_state_position():
    from cryptotrader.risk.models import ConnectionRiskRequest

    request = _request()
    stale = OpenVenueState(
        ConnectionPosition(request.quote.pair, Decimal("0"), Decimal("1"), None),
        (),
        (),
    )

    with pytest.raises(ValueError, match="open state position"):
        ConnectionRiskRequest(
            request.target,
            request.portfolio,
            True,
            request.quote,
            stale,
            request.capabilities,
        )
