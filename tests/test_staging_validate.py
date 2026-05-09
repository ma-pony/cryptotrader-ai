"""spec 020a — tests/test_staging_validate.py

T008 [P] [US1]: 单测 run_step 成功路径 / 失败路径 / 输出格式
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保 scripts/ 目录在 import 路径中
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from staging_validate import StepResult, run_step


class TestStepResult:
    def test_fmt_pass(self):
        r = StepResult(1, "test step", "PASS", 123)
        line = r.fmt()
        assert "[step 1]" in line
        assert "test step" in line
        assert "PASS" in line
        assert "123ms" in line
        assert "ERROR" not in line

    def test_fmt_fail_with_error(self):
        r = StepResult(2, "broken step", "FAIL", 456, "something went wrong")
        line = r.fmt()
        assert "[step 2]" in line
        assert "FAIL" in line
        assert "ERROR: something went wrong" in line

    def test_fmt_no_error_when_pass(self):
        r = StepResult(3, "ok step", "PASS", 0, "")
        assert "ERROR" not in r.fmt()


class TestRunStep:
    def test_success_path(self):
        """run_step returns PASS when fn completes without exception."""

        def _ok():
            pass  # no exception

        result = run_step(1, "ok step", _ok)
        assert result.status == "PASS"
        assert result.name == "ok step"
        assert result.idx == 1
        assert result.duration_ms >= 0
        assert result.error == ""

    def test_failure_path_exception(self):
        """run_step returns FAIL when fn raises."""

        def _fail():
            raise RuntimeError("boom")

        result = run_step(2, "fail step", _fail)
        assert result.status == "FAIL"
        assert "boom" in result.error
        assert result.duration_ms >= 0

    def test_failure_path_assertion_error(self):
        """run_step returns FAIL on AssertionError."""

        def _assert():
            assert False, "my assertion"  # noqa: B011

        result = run_step(3, "assert step", _assert)
        assert result.status == "FAIL"
        assert "my assertion" in result.error

    def test_duration_non_negative(self):
        """Duration is non-negative even for instant functions."""
        result = run_step(1, "instant", lambda: None)
        assert result.duration_ms >= 0

    def test_output_format_matches_spec(self):
        """FR-Z3: stdout format [step N] <name>: PASS|FAIL <duration>ms"""
        result = run_step(5, "otel check", lambda: None)
        fmt = result.fmt()
        # Must match [step N] <name>: PASS <Nms> pattern
        assert fmt.startswith("[step 5]")
        assert "otel check" in fmt
        assert "PASS" in fmt
        assert "ms" in fmt


class TestMigrateStep:
    def test_migrate_missing_script_fails(self, tmp_path):
        """_migrate raises FileNotFoundError for missing script."""
        # Temporarily patch __file__ path to use tmp_path
        import staging_validate as sv
        from staging_validate import _migrate

        original = sv.__file__

        try:
            sv.__file__ = str(tmp_path / "staging_validate.py")
            with pytest.raises(FileNotFoundError):
                _migrate("nonexistent_migrate_script")
        finally:
            sv.__file__ = original

    def test_run_step_wraps_migrate_error(self, tmp_path):
        """run_step wraps _migrate FileNotFoundError as FAIL."""
        import staging_validate as sv
        from staging_validate import _migrate

        original = sv.__file__

        try:
            sv.__file__ = str(tmp_path / "staging_validate.py")
            result = run_step(1, "migrate nonexistent", lambda: _migrate("nonexistent"))
            assert result.status == "FAIL"
            assert "nonexistent" in result.error.lower() or "not found" in result.error.lower()
        finally:
            sv.__file__ = original


class TestMainExitCode:
    def test_main_all_pass(self, monkeypatch):
        """main() returns 0 when all steps pass."""
        from staging_validate import main

        call_count = 0

        def _mock_run_step(idx, name, fn):
            nonlocal call_count
            call_count += 1
            return StepResult(idx, name, "PASS", 10)

        monkeypatch.setattr("staging_validate.run_step", _mock_run_step)
        code = main(dry_run=True)
        assert code == 0
        assert call_count == 5  # 5 steps

    def test_main_one_fail_returns_nonzero(self, monkeypatch):
        """main() returns 1 when any step fails."""
        from staging_validate import main

        call_count = 0

        def _mock_run_step(idx, name, fn):
            nonlocal call_count
            call_count += 1
            # step 3 fails
            status = "FAIL" if idx == 3 else "PASS"
            return StepResult(idx, name, status, 10, "boom" if idx == 3 else "")

        monkeypatch.setattr("staging_validate.run_step", _mock_run_step)
        code = main(dry_run=True)
        assert code == 1

    def test_main_prints_step_results(self, monkeypatch, capsys):
        """main() prints [step N] lines to stdout."""
        from staging_validate import main

        def _mock_run_step(idx, name, fn):
            return StepResult(idx, name, "PASS", 5)

        monkeypatch.setattr("staging_validate.run_step", _mock_run_step)
        main(dry_run=True)
        captured = capsys.readouterr()
        assert "[step 1]" in captured.out
        assert "[step 5]" in captured.out
        assert "PASS" in captured.out
