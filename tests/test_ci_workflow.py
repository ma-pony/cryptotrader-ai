"""Tests for CI workflow configuration correctness (Task 9.1).

Validates:
- Install target uses .[test] not .[dev]
- CI steps execute in the correct order: lint -> format -> pytest -> docker build
- docker build step only runs on main branch
- Any step failure blocks subsequent steps (sequential dependency)
"""

from pathlib import Path

import yaml

CI_WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    with open(CI_WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


class TestInstallTarget:
    """Verify the install command uses the [test] optional-dependency group."""

    def test_ci_workflow_file_exists(self):
        assert CI_WORKFLOW_PATH.exists(), "CI workflow file must exist"

    def test_install_uses_test_group_not_dev(self):
        workflow = _load_workflow()
        test_job = workflow["jobs"]["test"]
        steps = test_job["steps"]
        install_steps = [s for s in steps if "run" in s and "pip install" in s.get("run", "")]
        assert install_steps, "Must have at least one pip install step"
        for step in install_steps:
            cmd = step["run"]
            assert ".[dev]" not in cmd, f"Install must not use .[dev], got: {cmd}"
            assert ".[test]" in cmd, f"Install must use .[test], got: {cmd}"

    def test_install_step_name_descriptive(self):
        workflow = _load_workflow()
        test_job = workflow["jobs"]["test"]
        steps = test_job["steps"]
        install_steps = [s for s in steps if "run" in s and "pip install" in s.get("run", "")]
        for step in install_steps:
            assert step.get("name"), "Install step must have a descriptive name"


class TestStepOrdering:
    """Verify CI steps execute in the correct sequence."""

    def _get_step_names(self, steps: list[dict]) -> list[str]:
        return [s.get("name", "") for s in steps if "run" in s]

    def _find_step_index(self, steps: list[dict], keyword: str) -> int:
        """Return index of first step whose run command contains keyword."""
        for i, step in enumerate(steps):
            if keyword in step.get("run", ""):
                return i
        return -1

    def test_lint_step_exists(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        idx = self._find_step_index(steps, "ruff check")
        assert idx >= 0, "Must have a ruff check (lint) step"

    def test_format_check_step_exists(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        idx = self._find_step_index(steps, "ruff format --check")
        assert idx >= 0, "Must have a ruff format --check step"

    def test_pytest_step_exists(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        idx = self._find_step_index(steps, "pytest")
        assert idx >= 0, "Must have a pytest step"

    def test_lint_before_format_check(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        lint_idx = self._find_step_index(steps, "ruff check")
        format_idx = self._find_step_index(steps, "ruff format --check")
        assert lint_idx < format_idx, (
            f"ruff check (lint) must come before ruff format --check; lint_idx={lint_idx}, format_idx={format_idx}"
        )

    def test_format_check_before_pytest(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        format_idx = self._find_step_index(steps, "ruff format --check")
        pytest_idx = self._find_step_index(steps, "pytest")
        assert format_idx < pytest_idx, (
            f"ruff format --check must come before pytest; format_idx={format_idx}, pytest_idx={pytest_idx}"
        )

    def test_install_before_lint(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        install_idx = self._find_step_index(steps, "pip install")
        lint_idx = self._find_step_index(steps, "ruff check")
        assert install_idx < lint_idx, f"install must come before lint; install_idx={install_idx}, lint_idx={lint_idx}"


class TestDockerBuildJob:
    """Verify docker build job only runs on main branch."""

    def test_docker_job_exists(self):
        workflow = _load_workflow()
        assert "docker" in workflow["jobs"], "Must have a docker job"

    def test_docker_job_depends_on_test_job(self):
        workflow = _load_workflow()
        docker_job = workflow["jobs"]["docker"]
        needs = docker_job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "test" in needs, "docker job must depend on test job via 'needs'"

    def test_docker_job_runs_only_on_main(self):
        workflow = _load_workflow()
        docker_job = workflow["jobs"]["docker"]
        condition = docker_job.get("if", "")
        assert "main" in condition, f"docker job must have 'if' condition restricting to main branch; got: {condition}"

    def test_docker_build_step_exists(self):
        workflow = _load_workflow()
        docker_job = workflow["jobs"]["docker"]
        steps = docker_job["steps"]
        build_steps = [s for s in steps if "docker build" in s.get("run", "")]
        assert build_steps, "docker job must have a docker build step"


class TestStepFailureBlocking:
    """Verify configuration ensures step failure blocks subsequent steps.

    In GitHub Actions, steps within a job are sequential by default and stop on
    failure unless 'continue-on-error: true' is set. This validates that no
    critical steps have continue-on-error enabled.
    """

    def _get_critical_steps(self, steps: list[dict]) -> list[dict]:
        """Return steps that run lint, format, pytest - must not continue on error."""
        keywords = ["ruff check", "ruff format --check", "pytest"]
        return [s for s in steps if any(kw in s.get("run", "") for kw in keywords)]

    def test_lint_step_does_not_continue_on_error(self):
        workflow = _load_workflow()
        steps = workflow["jobs"]["test"]["steps"]
        critical = self._get_critical_steps(steps)
        for step in critical:
            assert not step.get("continue-on-error", False), (
                f"Critical step '{step.get('name', step.get('run', ''))}' must not have continue-on-error: true"
            )

    def test_test_job_has_no_fail_fast_false(self):
        """Verify strategy.fail-fast is not set to false (default is true)."""
        workflow = _load_workflow()
        test_job = workflow["jobs"]["test"]
        strategy = test_job.get("strategy", {})
        fail_fast = strategy.get("fail-fast", True)
        assert fail_fast is not False, "test job strategy.fail-fast must not be false - steps must block on failure"
