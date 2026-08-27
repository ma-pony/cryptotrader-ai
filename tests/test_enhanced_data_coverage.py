"""Tests for risk/checks/correlation.py and risk/checks/token_security.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.decision.models import TargetPosition
from cryptotrader.pair import Pair
from cryptotrader.risk.checks.correlation import CorrelationCheck, _find_group
from tests.factories.signal_fusion import risk_request

# ── _find_group ──


class TestFindGroup:
    def test_btc_group(self):
        g = _find_group("BTC")
        assert g is not None
        assert "BTC" in g
        assert "WBTC" in g

    def test_eth_group(self):
        g = _find_group("ETH")
        assert g is not None
        assert "STETH" in g

    def test_unknown(self):
        assert _find_group("UNKNOWN_TOKEN_XYZ") is None

    def test_case_insensitive(self):
        g = _find_group("btc")
        assert g is not None


# ── CorrelationCheck ──


class TestCorrelationCheck:
    def _make_check(self, max_correlated: int = 2):
        cfg = MagicMock()
        cfg.max_correlated_positions = max_correlated
        return CorrelationCheck(cfg)

    @pytest.mark.asyncio
    async def test_hold_always_passes(self):
        check = self._make_check()
        request = risk_request(target=TargetPosition("flat", 0.0))
        result = await check.evaluate(request, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unknown_symbol_passes(self):
        check = self._make_check()
        request = risk_request(pair=Pair.parse("UNKNOWN/USDT"), market_type="spot")
        result = await check.evaluate(request, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_correlated_positions(self):
        check = self._make_check()
        result = await check.evaluate(risk_request(), {"positions": {}})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_too_many_correlated(self):
        check = self._make_check(max_correlated=1)
        portfolio = {"positions": {"WBTC/USDT": {"amount": 1.0}}}
        result = await check.evaluate(risk_request(), portfolio)
        assert result.passed is False
        assert "correlated" in result.reason

    @pytest.mark.asyncio
    async def test_zero_amount_ignored(self):
        check = self._make_check(max_correlated=1)
        portfolio = {"positions": {"WBTC/USDT": {"amount": 0}}}
        result = await check.evaluate(risk_request(), portfolio)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_numeric_position_value(self):
        check = self._make_check(max_correlated=1)
        portfolio = {"positions": {"STETH/USDT": 5.0}}
        request = risk_request(pair=Pair.parse("ETH/USDT"), market_type="spot")
        result = await check.evaluate(request, portfolio)
        assert result.passed is False


# ── TokenSecurityCheck ──


class TestTokenSecurityCheck:
    @pytest.mark.asyncio
    async def test_no_contract_address(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        result = await check.evaluate(risk_request(), {})
        assert result.passed is True
        assert "No contract" in result.reason

    @pytest.mark.asyncio
    async def test_audit_api_error(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(side_effect=Exception("network"))
        portfolio = {"contract_address": "0xabc", "symbol": "TOKEN", "chain": "BSC"}
        result = await check.evaluate(risk_request(), portfolio)
        assert result.passed is True
        assert "unavailable" in result.reason

    @pytest.mark.asyncio
    async def test_high_risk(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(return_value={"risk_level": "HIGH", "issues": ["honeypot"]})
        portfolio = {"contract_address": "0xabc", "symbol": "TOKEN", "chain": "BSC"}
        result = await check.evaluate(risk_request(), portfolio)
        assert result.passed is False
        assert "honeypot" in result.reason

    @pytest.mark.asyncio
    async def test_low_risk(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(return_value={"risk_level": "LOW", "issues": []})
        portfolio = {"contract_address": "0xabc", "symbol": "TOKEN", "chain": "BSC"}
        result = await check.evaluate(risk_request(), portfolio)
        assert result.passed is True
