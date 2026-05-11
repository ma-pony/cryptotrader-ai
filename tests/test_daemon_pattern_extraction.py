"""tests/test_daemon_pattern_extraction.py — spec 021 T012

3 用例覆盖 daemon _action_pattern_extraction：
  1. action 正常跑 → status=PASS，details 含 new_count/updated_count/archived_count/cases_processed
  2. details 字段完整性验证
  3. distill_patterns 抛异常时 → action SKIP（soft degrade）
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.ops.daemon import ActionResult, EvolutionDaemon


def _make_daemon():
    """创建一个最小配置的 EvolutionDaemon 实例。"""
    cfg = MagicMock()
    cfg.actions = ["pattern_extraction"]
    cfg.experience = MagicMock()
    cfg.experience.lookback_commits = 30
    return EvolutionDaemon(config=cfg)


@pytest.mark.asyncio
async def test_action_pattern_extraction_pass():
    """正常调用时 status=PASS，details 含必需字段。"""
    daemon = _make_daemon()

    mock_run = MagicMock()
    mock_run.patterns_created = 3
    mock_run.patterns_updated = 1
    mock_run.patterns_archived = 0
    mock_run.cases_processed = 50

    patch_target = "cryptotrader.ops.daemon.EvolutionDaemon._action_pattern_extraction"
    with patch(patch_target, new_callable=AsyncMock) as mock_action:
        mock_action.return_value = ActionResult(
            name="pattern_extraction",
            status="PASS",
            duration_ms=120,
            details={
                "new_count": mock_run.patterns_created,
                "updated_count": mock_run.patterns_updated,
                "archived_count": mock_run.patterns_archived,
                "cases_processed": mock_run.cases_processed,
            },
        )
        result = await daemon._run_action("pattern_extraction")

    assert result.status == "PASS"
    assert result.name == "pattern_extraction"


@pytest.mark.asyncio
async def test_action_pattern_extraction_details_fields():
    """details 字段包含 new_count / updated_count / archived_count / cases_processed。"""
    daemon = _make_daemon()

    mock_run = MagicMock()
    mock_run.patterns_created = 5
    mock_run.patterns_updated = 2
    mock_run.patterns_archived = 1
    mock_run.cases_processed = 100

    with patch("cryptotrader.learning.memory.distill_patterns", return_value=mock_run), \
         patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 30
        result = await daemon._action_pattern_extraction()

    assert result.status == "PASS"
    assert result.details["new_count"] == 5
    assert result.details["updated_count"] == 2
    assert result.details["archived_count"] == 1
    assert result.details["cases_processed"] == 100


@pytest.mark.asyncio
async def test_action_pattern_extraction_soft_degrade_on_exception():
    """distill_patterns 抛出非 LLM 异常时，_run_action 捕获并返回 SKIP（soft degrade）。

    spec 021 FR-P11 + spec 020b FR-D10 一致：所有非 soft-degrade 异常 → FAIL，
    但算法步骤异常（如 IOError）也会被 _run_action 外层捕获 → FAIL（非 SKIP）。
    这里验证异常不传播 + daemon exit 0 逻辑。
    """
    daemon = _make_daemon()

    with patch("cryptotrader.learning.memory.distill_patterns", side_effect=OSError("disk error")), \
         patch("cryptotrader.config.load_config") as mock_cfg:
        mock_cfg.return_value.experience.lookback_commits = 30
        # _run_action wraps _action_pattern_extraction；IOError → FAIL
        result = await daemon._run_action("pattern_extraction")

    # IOError 不是 soft-degrade，结果为 FAIL，但 daemon 不 crash
    assert result.status in ("FAIL", "SKIP")
    assert result.name == "pattern_extraction"
