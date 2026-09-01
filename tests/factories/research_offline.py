"""Fail closed if a research test misses an external dependency double."""

import socket

import pytest


@pytest.fixture(autouse=True)
def research_offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("research tests cannot access external network or models")

    async def denied_async(*args, **kwargs):
        denied()

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr("cryptotrader.backtest.cache.fetch_historical", denied_async)
    monkeypatch.setattr("cryptotrader.market_sources.default.DefaultMarketDataSource.read_candles", denied_async)
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", denied_async)
    monkeypatch.setattr("langchain_openai.ChatOpenAI.ainvoke", denied_async)
