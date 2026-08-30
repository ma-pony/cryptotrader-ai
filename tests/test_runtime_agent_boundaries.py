"""Runtime-config boundaries for chart context, agents, and provider tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.runtime_config.models import LlmConfig, LlmModelsConfig


@pytest.mark.asyncio
async def test_credentialed_agent_tools_are_unavailable_without_a_key_and_use_only_explicit_key():
    from cryptotrader.agents.data_tools import get_exchange_netflow, get_liquidation_data, get_whale_transfers

    assert json.loads(await get_liquidation_data.ainvoke({"pair": "BTC/USDT"})) == {
        "open_interest": 0.0,
        "liquidations_24h": {},
        "data_available": False,
    }
    assert json.loads(await get_whale_transfers.ainvoke({"pair": "BTC/USDT"})) == {
        "transfers": [],
        "data_available": False,
    }
    assert json.loads(await get_exchange_netflow.ainvoke({"pair": "BTC/USDT"})) == {
        "netflow": 0.0,
        "data_available": False,
    }
    with patch(
        "cryptotrader.data.providers.coinglass.fetch_derivatives",
        new=AsyncMock(return_value={"oi": 3}),
    ) as fetch:
        assert json.loads(await get_liquidation_data.ainvoke({"pair": "BTC/USDT", "provider_key": "explicit-key"})) == {
            "oi": 3
        }
    fetch.assert_awaited_once_with("explicit-key", "BTC")


def test_runtime_llm_factory_owns_timeout_and_fallback_without_global_config(monkeypatch):
    from cryptotrader.agents.base import create_llm

    captured = []

    def factory(config, *, api_key):
        assert api_key == ""

        def build(**kwargs):
            captured.append((config, kwargs))
            return object()

        return build

    monkeypatch.setattr("cryptotrader.agents.base.create_runtime_llm_factory", factory)
    settings = LlmConfig(models=LlmModelsConfig(analysis="primary", fallback="fallback"), timeout=77)

    assert create_llm(settings, model="chosen", timeout=12, with_fallback=False) is not None
    assert captured == [
        (
            settings,
            {
                "model": "chosen",
                "temperature": None,
                "timeout": 12,
                "json_mode": False,
                "with_fallback": False,
                "role": "",
                "track_tokens": True,
            },
        )
    ]
