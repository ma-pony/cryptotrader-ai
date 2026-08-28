"""New book/connection risk requests replace the legacy target-risk wrapper."""

from __future__ import annotations

from decimal import Decimal

from tests.test_book_risk_gate import _limits as book_limits
from tests.test_book_risk_gate import _request as book_request
from tests.test_connection_risk_gate import _gate as connection_gate
from tests.test_connection_risk_gate import _request as connection_request


def test_target_risk_gate_caps_the_book_then_evaluates_connection_targets():
    from cryptotrader.risk.gate import BookRiskGate

    book_decision = BookRiskGate(book_limits(max_net_exposure=Decimal("0.4"))).evaluate(book_request())
    connection_decision = connection_gate().evaluate(connection_request(target="4000"))

    assert book_decision.passed is True
    assert book_decision.capped_target_exposure == Decimal("0.4")
    assert tuple(target.weight for target in book_decision.connection_targets) == (
        Decimal("0.4"),
        Decimal("0.6"),
    )
    assert connection_decision.passed is True


def test_sign_flip_is_classified_as_risk_increase_for_connection_preflight():
    decision = connection_gate().evaluate(connection_request(current="2000", target="-1000"))

    assert decision.passed is True
    assert decision.risk_increase is True
