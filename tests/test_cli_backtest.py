"""Backtest CLI only exposes the shared TradingCycle implementation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def test_backtest_help_has_no_legacy_llm_switch():
    result = runner.invoke(app, ["backtest", "--help"])

    assert result.exit_code == 0
    assert "--use-llm" not in result.output
    assert "--no-llm" not in result.output


def test_backtest_constructs_engine_without_legacy_strategy_argument():
    with patch("cryptotrader.backtest.engine.BacktestEngine") as engine_type:
        engine_type.return_value.run = AsyncMock(return_value=SimpleNamespace(summary=dict))
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-02",
            ],
        )

    assert result.exit_code == 0
    engine_type.assert_called_once_with(
        "BTC/USDT",
        "2026-01-01",
        "2026-01-02",
        "4h",
        10000.0,
    )
