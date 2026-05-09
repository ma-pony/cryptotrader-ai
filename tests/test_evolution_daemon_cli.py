"""spec 022 T029 -- CLI entry-point tests for arena evolution-daemon.

Tests cover:
- --once flag runs run_once() and exits with its exit_code
- EVOLUTION_DAEMON_ENABLED=false causes immediate exit 0
- evolution_daemon.enabled=false (toml) causes immediate exit 0
- Output format contains action name/status lines
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from cli.main import app
from cryptotrader.config import EvolutionDaemonConfig
from cryptotrader.ops.daemon import ActionResult, RunResult

runner = CliRunner()


def _make_run_result(statuses: list[str]) -> RunResult:
    actions = [ActionResult(name=f"action{i}", status=s, duration_ms=10, details={}) for i, s in enumerate(statuses)]
    exit_code = 1 if any(s == "FAIL" for s in statuses) else 0
    return RunResult(actions_run=actions, total_duration_ms=30, exit_code=exit_code)


# ---------------------------------------------------------------------------
# --once flag: calls run_once and exits with exit_code
# ---------------------------------------------------------------------------


def test_evolution_daemon_once_exit_0(tmp_path):
    """--once with all-PASS run -> exit code 0, output contains PASS."""
    run_result = _make_run_result(["PASS", "PASS", "PASS"])

    mock_daemon = MagicMock()
    mock_daemon.run_once = AsyncMock(return_value=run_result)

    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon", return_value=mock_daemon),
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        cfg = MagicMock()
        cfg.evolution_daemon = EvolutionDaemonConfig(enabled=True)
        mock_cfg.return_value = cfg

        result = runner.invoke(app, ["evolution-daemon", "--once"])

    assert result.exit_code == 0
    assert "exit_code=0" in result.output
    assert "PASS" in result.output


def test_evolution_daemon_once_exit_1_on_fail(tmp_path):
    """--once with a FAIL action -> exit code 1."""
    run_result = _make_run_result(["PASS", "FAIL", "PASS"])

    mock_daemon = MagicMock()
    mock_daemon.run_once = AsyncMock(return_value=run_result)

    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon", return_value=mock_daemon),
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        cfg = MagicMock()
        cfg.evolution_daemon = EvolutionDaemonConfig(enabled=True)
        mock_cfg.return_value = cfg

        result = runner.invoke(app, ["evolution-daemon", "--once"])

    assert result.exit_code == 1


def test_evolution_daemon_once_output_format():
    """--once output contains [action_name] STATUS ms lines."""
    actions = [
        ActionResult("pareto", "PASS", 145, {}),
        ActionResult("regime", "PASS", 67, {}),
        ActionResult("skill_proposal", "SKIP", 5, {}),
    ]
    run_result = RunResult(actions_run=actions, total_duration_ms=217, exit_code=0)

    mock_daemon = MagicMock()
    mock_daemon.run_once = AsyncMock(return_value=run_result)

    # EvolutionDaemon is imported inside the CLI function body; patch at source.
    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon", return_value=mock_daemon),
        patch("cryptotrader.config.load_config") as mock_cfg,
        patch("cryptotrader.config._cached_config", None),
    ):
        cfg = MagicMock()
        cfg.evolution_daemon = EvolutionDaemonConfig(enabled=True)
        mock_cfg.return_value = cfg

        result = runner.invoke(app, ["evolution-daemon", "--once"])

    # Rich console strips bracket markup; check status words and durations are present
    assert "PASS" in result.output
    assert "SKIP" in result.output
    assert "145ms" in result.output
    assert "67ms" in result.output
    assert "5ms" in result.output


# ---------------------------------------------------------------------------
# EVOLUTION_DAEMON_ENABLED=false -> immediate exit 0
# ---------------------------------------------------------------------------


def test_evolution_daemon_disabled_by_env():
    """EVOLUTION_DAEMON_ENABLED=false -> immediate exit 0 before daemon import."""
    # env check happens before EvolutionDaemon import; just verify exit + message
    with patch.dict("os.environ", {"EVOLUTION_DAEMON_ENABLED": "false"}):
        result = runner.invoke(app, ["evolution-daemon", "--once"])

    assert result.exit_code == 0
    assert "disabled" in result.output.lower()


def test_evolution_daemon_disabled_by_env_zero():
    """EVOLUTION_DAEMON_ENABLED=0 -> immediate exit 0."""
    with patch.dict("os.environ", {"EVOLUTION_DAEMON_ENABLED": "0"}):
        result = runner.invoke(app, ["evolution-daemon", "--once"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# toml enabled=false -> immediate exit 0
# ---------------------------------------------------------------------------


def test_evolution_daemon_disabled_by_toml():
    """evolution_daemon.enabled=false in config -> immediate exit 0."""
    with (
        patch.dict("os.environ", {"EVOLUTION_DAEMON_ENABLED": "true"}),
        patch("cryptotrader.config.load_config") as mock_cfg,
        patch("cryptotrader.config._cached_config", None),
    ):
        cfg = MagicMock()
        cfg.evolution_daemon = EvolutionDaemonConfig(enabled=False)
        mock_cfg.return_value = cfg

        result = runner.invoke(app, ["evolution-daemon", "--once"])

    assert result.exit_code == 0
    assert "disabled" in result.output.lower()
