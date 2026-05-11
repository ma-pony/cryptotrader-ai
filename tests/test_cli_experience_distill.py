"""tests/test_cli_experience_distill.py — spec 021 T015

3 用例覆盖 arena experience distill CLI 命令：
  1. 默认参数 PASS + 输出 summary
  2. --memory-dir custom 路径生效
  3. --cycles-window N 参数限制生效
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

# 从 cli/main.py 导入 app（需要先确保可以 import）
from cli.main import app

runner = CliRunner()


def _mock_run(
    patterns_created: int = 2,
    patterns_updated: int = 1,
    patterns_archived: int = 0,
    cases_processed: int = 30,
    error: str = "",
):
    run = MagicMock()
    run.patterns_created = patterns_created
    run.patterns_updated = patterns_updated
    run.patterns_archived = patterns_archived
    run.cases_processed = cases_processed
    run.error = error
    return run


def test_experience_distill_default_params(tmp_path):
    """默认参数调用成功，输出包含 cases_processed / patterns_created。"""
    mock_run = _mock_run(patterns_created=3, cases_processed=25)

    with patch("cryptotrader.learning.memory.distill_patterns", return_value=mock_run) as mock_distill, \
         patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 30
        result = runner.invoke(app, ["experience", "distill", "--memory-dir", str(tmp_path)])

    assert result.exit_code == 0, f"stdout: {result.stdout}"
    assert "cases_processed" in result.stdout
    assert "patterns_created" in result.stdout


def test_experience_distill_custom_memory_dir(tmp_path):
    """--memory-dir custom 路径传入 distill_patterns。"""
    custom_dir = tmp_path / "custom_memory"
    custom_dir.mkdir()
    mock_run = _mock_run()
    captured_args = {}

    def fake_distill(memory_dir=None, cycles_window=50):
        captured_args["memory_dir"] = memory_dir
        captured_args["cycles_window"] = cycles_window
        return mock_run

    with patch("cryptotrader.learning.memory.distill_patterns", side_effect=fake_distill), \
         patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 30
        result = runner.invoke(app, ["experience", "distill", "--memory-dir", str(custom_dir)])

    assert result.exit_code == 0, f"stdout: {result.stdout}"
    assert captured_args.get("memory_dir") == custom_dir


def test_experience_distill_cycles_window(tmp_path):
    """--cycles-window 50 正确传给 distill_patterns。"""
    mock_run = _mock_run()
    captured_args = {}

    def fake_distill(memory_dir=None, cycles_window=50):
        captured_args["cycles_window"] = cycles_window
        return mock_run

    with patch("cryptotrader.learning.memory.distill_patterns", side_effect=fake_distill), \
         patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 30
        result = runner.invoke(
            app, ["experience", "distill", "--memory-dir", str(tmp_path), "--cycles-window", "50"]
        )

    assert result.exit_code == 0, f"stdout: {result.stdout}"
    assert captured_args.get("cycles_window") == 50
