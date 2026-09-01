"""Only explicit temporary legacy files are imported; missing evidence stays missing."""

import json

import pytest

from cryptotrader.backtest.comparison import compare_runs
from cryptotrader.backtest.store import BacktestStore
from cryptotrader.db import dispose_engine
from cryptotrader.migrations.workbench import import_backtest_session, migrate_workbench_schema


@pytest.mark.asyncio
async def test_legacy_file_import_preserves_source_and_marks_absent_curve_and_conditions(tmp_path):
    source = tmp_path / "old-run"
    source.mkdir()
    params = {"pair": "BTC/USDT", "start": "2024-01-01", "end": "2024-01-02", "initial_capital": 10000}
    result = {"total_return": 0.12, "sharpe_ratio": 1.5, "max_drawdown": -0.08, "win_rate": 0.65, "decisions": []}
    (source / "params.json").write_text(json.dumps(params))
    (source / "result.json").write_text(json.dumps(result))
    original = (source / "result.json").read_bytes()
    url = f"sqlite+aiosqlite:///{tmp_path}/import.sqlite"
    await migrate_workbench_schema(url)
    run_id = await import_backtest_session(url, source)
    run = await BacktestStore(url).get(run_id)
    assert run.result.equity_curve == []
    assert any("equity_curve" in item for item in run.incomplete_fields)
    assert run.params.interval is None
    assert run.params.fee_rate is None
    assert run.config_snapshot is None
    assert (source / "result.json").read_bytes() == original
    assert compare_runs(run, run).comparable is False
    assert "interval" in compare_runs(run, run).condition_differences
    assert await import_backtest_session(url, source) == run_id
    await dispose_engine(url)
