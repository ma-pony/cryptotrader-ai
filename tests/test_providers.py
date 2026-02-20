"""Tests for on-chain data providers with mocked HTTP."""

from __future__ import annotations

import json
import pytest
import httpx

from cryptotrader.data.providers import defillama, coinglass, cryptoquant, whale_alert


class MockResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data

    @property
    def text(self):
        return json.dumps(self._data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


# ── DefiLlama ──

@pytest.mark.asyncio
async def test_defillama_tvl(monkeypatch):
    tvl_data = [{"tvl": 100.0}] * 7 + [{"tvl": 110.0}]

    async def mock_get(self, url, **kw):
        return MockResponse(tvl_data)

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await defillama.fetch_tvl("Ethereum")
    assert result["defi_tvl"] == 110.0
    assert result["defi_tvl_change_7d"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_defillama_failure(monkeypatch):
    async def mock_get(self, url, **kw):
        raise httpx.ConnectError("fail")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await defillama.fetch_tvl()
    assert result["defi_tvl"] == 0.0


# ── CoinGlass ──

@pytest.mark.asyncio
async def test_coinglass_no_key():
    result = await coinglass.fetch_derivatives(api_key=None)
    assert result["open_interest"] == 0.0


@pytest.mark.asyncio
async def test_coinglass_with_key(monkeypatch):
    call_count = 0

    async def mock_get(self, url, **kw):
        nonlocal call_count
        call_count += 1
        if "open_interest" in url:
            return MockResponse({"data": [{"openInterest": 5000.0}]})
        return MockResponse({"data": [{"longLiquidationUsd": 100, "shortLiquidationUsd": 200}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await coinglass.fetch_derivatives("test-key", "BTC")
    assert result["open_interest"] == 5000.0
    assert result["liquidations_24h"]["long"] == 100


# ── CryptoQuant ──

@pytest.mark.asyncio
async def test_cryptoquant_no_key():
    result = await cryptoquant.fetch_exchange_netflow(api_key=None)
    assert result == 0.0


@pytest.mark.asyncio
async def test_cryptoquant_with_key(monkeypatch):
    async def mock_get(self, url, **kw):
        return MockResponse({"result": {"data": [{"netflow": -500.0}]}})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await cryptoquant.fetch_exchange_netflow("test-key")
    assert result == -500.0


# ── Whale Alert ──

@pytest.mark.asyncio
async def test_whale_alert_no_key():
    result = await whale_alert.fetch_whale_transfers(api_key=None)
    assert result == []


@pytest.mark.asyncio
async def test_whale_alert_with_key(monkeypatch):
    async def mock_get(self, url, **kw):
        return MockResponse({"transactions": [
            {"hash": "abc", "from": {"owner": "binance"}, "to": {"owner": "unknown"}, "amount_usd": 1000000}
        ]})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await whale_alert.fetch_whale_transfers("test-key")
    assert len(result) == 1
    assert result[0]["amount_usd"] == 1000000
