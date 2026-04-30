"""Tests for ``[scheduler].pairs`` config schema (FR-100~104).

Covers both legacy ``list[str]`` form and new ``[[scheduler.pairs]]``
table-array form, plus validation errors.
"""

from __future__ import annotations

import pytest

from cryptotrader.config import (
    ConfigurationError,
    SchedulerConfig,
    _build_scheduler_config,
    _parse_pair_dict,
)
from cryptotrader.pair import Pair

# ── Legacy list[str] form ───────────────────────────────────────────────────


class TestLegacyListStr:
    def test_all_spot(self) -> None:
        cfg = _build_scheduler_config({"scheduler": {"pairs": ["BTC/USDT", "ETH/USDT"]}})
        assert isinstance(cfg, SchedulerConfig)
        assert len(cfg.pairs) == 2
        assert all(isinstance(p, Pair) for p in cfg.pairs)
        assert all(p.market_type == "spot" for p in cfg.pairs)
        assert [p.canonical() for p in cfg.pairs] == ["BTC/USDT", "ETH/USDT"]

    def test_empty_list_keeps_other_fields(self) -> None:
        cfg = _build_scheduler_config({"scheduler": {"pairs": [], "interval_minutes": 60, "enabled": True}})
        assert cfg.pairs == []
        assert cfg.interval_minutes == 60
        assert cfg.enabled is True

    def test_no_pairs_key_uses_dataclass_default(self) -> None:
        cfg = _build_scheduler_config({"scheduler": {"enabled": True}})
        # Default factory yields two spot pairs
        assert len(cfg.pairs) == 2
        assert all(isinstance(p, Pair) for p in cfg.pairs)


# ── New [[scheduler.pairs]] table-array form ────────────────────────────────


class TestNewTableArray:
    def test_all_spot(self) -> None:
        cfg = _build_scheduler_config({"scheduler": {"pairs": [{"symbol": "BTC/USDT"}, {"symbol": "ETH/USDT"}]}})
        assert all(p.market_type == "spot" for p in cfg.pairs)

    def test_explicit_market_spot(self) -> None:
        cfg = _build_scheduler_config({"scheduler": {"pairs": [{"symbol": "BTC/USDT", "market": "spot"}]}})
        assert cfg.pairs[0].market_type == "spot"
        assert cfg.pairs[0].canonical() == "BTC/USDT"

    def test_swap_with_settle(self) -> None:
        cfg = _build_scheduler_config(
            {"scheduler": {"pairs": [{"symbol": "BTC/USDT", "market": "swap", "settle": "USDT"}]}}
        )
        assert cfg.pairs[0].market_type == "swap"
        assert cfg.pairs[0].canonical() == "BTC/USDT:USDT"
        assert cfg.pairs[0].settle == "USDT"

    def test_inverse_swap(self) -> None:
        cfg = _build_scheduler_config(
            {"scheduler": {"pairs": [{"symbol": "BTC/USD", "market": "swap", "settle": "BTC"}]}}
        )
        assert cfg.pairs[0].canonical() == "BTC/USD:BTC"
        assert cfg.pairs[0].settle == "BTC"

    def test_mixed_market_types(self) -> None:
        cfg = _build_scheduler_config(
            {
                "scheduler": {
                    "pairs": [
                        {"symbol": "BTC/USDT"},
                        {"symbol": "ETH/USDT", "market": "swap", "settle": "USDT"},
                    ]
                }
            }
        )
        assert cfg.pairs[0].market_type == "spot"
        assert cfg.pairs[1].market_type == "swap"


# ── Validation errors (FR-104) ──────────────────────────────────────────────


class TestValidation:
    def test_mixed_str_and_dict_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config({"scheduler": {"pairs": ["BTC/USDT", {"symbol": "ETH/USDT"}]}})
        assert exc.value.field_path == "scheduler.pairs"
        assert "mixing is not allowed" in exc.value.expected

    def test_swap_missing_settle_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config({"scheduler": {"pairs": [{"symbol": "BTC/USDT", "market": "swap"}]}})
        assert exc.value.field_path == "scheduler.pairs[0].settle"

    def test_spot_with_settle_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config(
                {"scheduler": {"pairs": [{"symbol": "BTC/USDT", "market": "spot", "settle": "USDT"}]}}
            )
        assert exc.value.field_path == "scheduler.pairs[0].settle"
        assert "must be omitted" in exc.value.expected

    def test_invalid_market_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config({"scheduler": {"pairs": [{"symbol": "BTC/USDT", "market": "options"}]}})
        assert exc.value.field_path == "scheduler.pairs[0].market"

    def test_missing_symbol_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config({"scheduler": {"pairs": [{"market": "spot"}]}})
        assert exc.value.field_path == "scheduler.pairs[0].symbol"

    def test_duplicate_canonical_raises(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            _build_scheduler_config({"scheduler": {"pairs": ["BTC/USDT", "BTC/USDT"]}})
        assert exc.value.field_path == "scheduler.pairs"
        assert "BTC/USDT" in exc.value.expected

    def test_duplicate_canonical_swap_vs_spot_ok(self) -> None:
        """BTC/USDT (spot) and BTC/USDT:USDT (swap) have different canonicals — not a duplicate."""
        cfg = _build_scheduler_config(
            {
                "scheduler": {
                    "pairs": [
                        {"symbol": "BTC/USDT"},
                        {"symbol": "BTC/USDT", "market": "swap", "settle": "USDT"},
                    ]
                }
            }
        )
        assert len(cfg.pairs) == 2

    def test_parse_pair_dict_directly(self) -> None:
        # Direct unit test of helper
        p = _parse_pair_dict({"symbol": "BTC/USDT", "market": "swap", "settle": "USDT"}, idx=0)
        assert p.canonical() == "BTC/USDT:USDT"


# ── Integration with full config load ──────────────────────────────────────


class TestIntegrationWithLoadConfig:
    def test_load_config_returns_pair_instances(self) -> None:
        """End-to-end: load_config() resolves [scheduler].pairs to list[Pair]."""
        from cryptotrader.config import load_config

        cfg = load_config()
        assert all(isinstance(p, Pair) for p in cfg.scheduler.pairs)
