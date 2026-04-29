"""Tests for risk/checks/correlation.py and risk/checks/token_security.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.models import TradeVerdict
from cryptotrader.risk.checks.correlation import CorrelationCheck, _find_group

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
        v = MagicMock(spec=TradeVerdict)
        v.action = "hold"
        result = await check.evaluate(v, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unknown_symbol_passes(self):
        check = self._make_check()
        v = MagicMock(spec=TradeVerdict)
        v.action = "long"
        v.pair = "UNKNOWN/USDT"
        result = await check.evaluate(v, {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_correlated_positions(self):
        check = self._make_check()
        v = MagicMock(spec=TradeVerdict)
        v.action = "long"
        v.pair = "BTC/USDT"
        result = await check.evaluate(v, {"positions": {}})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_too_many_correlated(self):
        check = self._make_check(max_correlated=1)
        v = MagicMock(spec=TradeVerdict)
        v.action = "long"
        v.pair = "BTC/USDT"
        portfolio = {"positions": {"WBTC/USDT": {"amount": 1.0}}}
        result = await check.evaluate(v, portfolio)
        assert result.passed is False
        assert "correlated" in result.reason

    @pytest.mark.asyncio
    async def test_zero_amount_ignored(self):
        check = self._make_check(max_correlated=1)
        v = MagicMock(spec=TradeVerdict)
        v.action = "long"
        v.pair = "BTC/USDT"
        portfolio = {"positions": {"WBTC/USDT": {"amount": 0}}}
        result = await check.evaluate(v, portfolio)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_numeric_position_value(self):
        check = self._make_check(max_correlated=1)
        v = MagicMock(spec=TradeVerdict)
        v.action = "long"
        v.pair = "ETH/USDT"
        portfolio = {"positions": {"STETH/USDT": 5.0}}
        result = await check.evaluate(v, portfolio)
        assert result.passed is False


# ── TokenSecurityCheck ──


class TestTokenSecurityCheck:
    @pytest.mark.asyncio
    async def test_no_contract_address(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        v = MagicMock(spec=TradeVerdict)
        v.contract_address = None  # spec=TradeVerdict won't have this
        # Use delattr to ensure getattr returns None
        del v.contract_address
        result = await check.evaluate(v, {})
        assert result.passed is True
        assert "No contract" in result.reason

    @pytest.mark.asyncio
    async def test_audit_api_error(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(side_effect=Exception("network"))
        v = MagicMock()
        v.contract_address = "0xabc"
        v.symbol = "TOKEN"
        v.chain = "BSC"
        result = await check.evaluate(v, {})
        assert result.passed is True
        assert "unavailable" in result.reason

    @pytest.mark.asyncio
    async def test_high_risk(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(return_value={"risk_level": "HIGH", "issues": ["honeypot"]})
        v = MagicMock()
        v.contract_address = "0xabc"
        v.symbol = "TOKEN"
        v.chain = "BSC"
        result = await check.evaluate(v, {})
        assert result.passed is False
        assert "honeypot" in result.reason

    @pytest.mark.asyncio
    async def test_low_risk(self):
        from cryptotrader.risk.checks.token_security import TokenSecurityCheck

        check = TokenSecurityCheck()
        check.audit = MagicMock()
        check.audit.audit_token = AsyncMock(return_value={"risk_level": "LOW", "issues": []})
        v = MagicMock()
        v.contract_address = "0xabc"
        v.symbol = "TOKEN"
        v.chain = "BSC"
        result = await check.evaluate(v, {})
        assert result.passed is True
