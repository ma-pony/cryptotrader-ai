"""Tests for state.py — build_initial_state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cryptotrader.state import build_initial_state


class TestBuildInitialState:
    def _mock_config(self):
        cfg = MagicMock()
        cfg.exchange_id = "binance"
        cfg.data.default_timeframe = "1h"
        cfg.data.ohlcv_limit = 200
        cfg.models.analysis = "gpt-4o"
        cfg.models.debate = "gpt-4o-mini"
        cfg.models.verdict = "gpt-4o"
        cfg.models.tech_agent = ""
        cfg.models.chain_agent = ""
        cfg.models.news_agent = ""
        cfg.models.macro_agent = ""
        cfg.infrastructure.database_url = "sqlite:///test.db"
        cfg.infrastructure.redis_url = None
        cfg.debate.convergence_threshold = 0.5
        cfg.debate.max_rounds = 2
        cfg.risk.position.max_single_pct = 0.25
        return cfg

    def test_default(self):
        cfg = self._mock_config()
        result = build_initial_state("BTC/USDT", config=cfg)
        assert result["metadata"]["pair"] == "BTC/USDT"
        assert result["metadata"]["exchange_id"] == "binance"
        assert result["metadata"]["timeframe"] == "1h"
        assert result["metadata"]["ohlcv_limit"] == 200
        assert result["debate_round"] == 0
        assert result["max_debate_rounds"] == 2
        assert result["messages"] == []
        assert result["data"] == {}

    def test_custom_params(self):
        cfg = self._mock_config()
        result = build_initial_state(
            "ETH/USDT",
            engine="live",
            exchange_id="okx",
            timeframe="4h",
            ohlcv_limit=500,
            config=cfg,
        )
        assert result["metadata"]["pair"] == "ETH/USDT"
        assert result["metadata"]["engine"] == "live"
        assert result["metadata"]["exchange_id"] == "okx"
        assert result["metadata"]["timeframe"] == "4h"
        assert result["metadata"]["ohlcv_limit"] == 500

    def test_with_snapshot(self):
        cfg = self._mock_config()
        snap = MagicMock()
        result = build_initial_state("BTC/USDT", snapshot=snap, config=cfg)
        assert result["data"]["snapshot"] is snap

    def test_extra_metadata(self):
        cfg = self._mock_config()
        result = build_initial_state(
            "BTC/USDT",
            config=cfg,
            extra_metadata={"backtest_mode": True, "custom_key": 42},
        )
        assert result["metadata"]["backtest_mode"] is True
        assert result["metadata"]["custom_key"] == 42

    def test_extra_data(self):
        cfg = self._mock_config()
        result = build_initial_state(
            "BTC/USDT",
            config=cfg,
            extra_data={"position_context": {"side": "flat"}},
        )
        assert result["data"]["position_context"]["side"] == "flat"

    def test_loads_config_when_none(self):
        cfg = self._mock_config()
        with patch("cryptotrader.config.load_config", return_value=cfg):
            result = build_initial_state("BTC/USDT")
        assert result["metadata"]["pair"] == "BTC/USDT"
        assert result["metadata"]["analysis_model"] == "gpt-4o"
