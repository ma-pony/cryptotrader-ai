"""Backtest CLI only exposes the shared TradingCycle implementation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from cli.main import app
from tests.factories.research_offline import research_offline  # noqa: F401

runner = CliRunner()


def test_backtest_help_has_no_legacy_llm_switch():
    result = runner.invoke(app, ["backtest", "--help"])

    assert result.exit_code == 0
    assert "--use-llm" not in result.output
    assert "--no-llm" not in result.output


def test_backtest_uses_durable_service_and_prints_run_identity():
    with (
        patch("cryptotrader.backtest.service.configured_service") as service,
        patch("cryptotrader.backtest.engine.BacktestEngine") as engine,
    ):
        engine.return_value.run = AsyncMock(return_value=SimpleNamespace(summary=dict))
        service.return_value.run = AsyncMock(
            return_value=SimpleNamespace(
                run_id="run_persisted", status="completed", result=SimpleNamespace(summary=dict)
            )
        )
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
    assert "run_persisted" in result.output
    params = service.return_value.run.call_args.args[0]
    assert params.pair == "BTC/USDT"
    assert params.interval == "4h"


def test_removed_save_directory_option_is_rejected(tmp_path):
    result = runner.invoke(
        app, ["backtest", "--start", "2024-01-01", "--end", "2024-01-02", "--save-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
