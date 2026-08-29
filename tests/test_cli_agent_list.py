"""Tests for database-Runtime-backed CLI discovery commands."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from cli.main import app
from cryptotrader.signals.registry import ComponentMetadata


def test_cli_unmounts_legacy_configuration_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("portfolio", "risk", "live-check", "migrate", "sync"):
        assert command not in result.output


def test_run_command_has_no_graph_option() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--graph" not in result.output


def test_agent_list_reads_runtime_signal_registry_and_closes_runtime() -> None:
    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            document=SimpleNamespace(
                signals=SimpleNamespace(
                    components=(SimpleNamespace(component_id="llm_committee", enabled=True),),
                )
            )
        ),
        signals=SimpleNamespace(
            metadata=lambda: (ComponentMetadata("llm_committee", "LLM committee", "Internal debate"),)
        ),
        close=AsyncMock(),
    )
    with patch("cryptotrader.runtime.build_runtime", AsyncMock(return_value=runtime)):
        result = CliRunner().invoke(app, ["agent", "list"])

    assert result.exit_code == 0
    assert "llm_committee" in result.output
    assert "enabled" in result.output
    runtime.close.assert_awaited_once_with()


def test_scheduler_status_reads_runtime_snapshot_and_closes_runtime() -> None:
    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(
            revision=12,
            document=SimpleNamespace(
                scheduler=SimpleNamespace(
                    enabled=True,
                    pairs=(),
                    interval_minutes=15,
                ),
                execution=SimpleNamespace(
                    books=(SimpleNamespace(id="simulation", enabled=True),),
                ),
            ),
        ),
        cycle=object(),
        close=AsyncMock(),
    )
    with patch("cryptotrader.runtime.build_runtime", AsyncMock(return_value=runtime)):
        result = CliRunner().invoke(app, ["scheduler", "status"])

    assert result.exit_code == 0
    assert "Config revision" in result.output
    assert "12" in result.output
    assert "simulation" in result.output
    runtime.close.assert_awaited_once_with()
