"""Tests for CI workflow configuration correctness (Task 9.1).

Validates:
- Install target uses .[test] not .[dev]
- Job ordering: lint -> test -> docker-build (via `needs`)
- docker-build job only runs on main branch
- No critical step has continue-on-error: true
"""

from pathlib import Path

import yaml

CI_WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    with open(CI_WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


def _all_run_commands(job: dict) -> str:
    """Concatenate all `run` strings in a job's steps for substring search."""
    return "\n".join(s.get("run", "") for s in job.get("steps", []))


class TestInstallTarget:
    """Verify the install command uses the [test] optional-dependency group."""

    def test_ci_workflow_file_exists(self):
        assert CI_WORKFLOW_PATH.exists(), "CI workflow file must exist"

    def test_install_uses_test_group_not_dev(self):
        workflow = _load_workflow()
        # Each job that runs Python tooling installs deps via `uv sync --extra test`
        # (replacing the legacy `pip install .[test]`). Verify both forms are
        # acceptable but [dev] is forbidden.
        for job_name in ("lint", "test"):
            cmds = _all_run_commands(workflow["jobs"][job_name])
            assert ".[dev]" not in cmds, f"{job_name} job must not use .[dev], got: {cmds}"
            uses_test_extra = "--extra test" in cmds or ".[test]" in cmds
            assert uses_test_extra, f"{job_name} job must install with [test] extras, got: {cmds}"


class TestJobOrdering:
    """Verify CI jobs execute in the correct sequence via `needs`."""

    def test_lint_job_exists(self):
        workflow = _load_workflow()
        assert "lint" in workflow["jobs"], "Must have a lint job"

    def test_test_job_exists(self):
        workflow = _load_workflow()
        assert "test" in workflow["jobs"], "Must have a test job"

    def test_lint_runs_ruff_check(self):
        workflow = _load_workflow()
        cmds = _all_run_commands(workflow["jobs"]["lint"])
        assert "ruff check" in cmds, "lint job must run `ruff check`"

    def test_lint_runs_ruff_format_check(self):
        workflow = _load_workflow()
        cmds = _all_run_commands(workflow["jobs"]["lint"])
        assert "ruff format --check" in cmds, "lint job must run `ruff format --check`"

    def test_test_runs_pytest(self):
        workflow = _load_workflow()
        cmds = _all_run_commands(workflow["jobs"]["test"])
        assert "pytest" in cmds, "test job must run pytest"

    def test_test_depends_on_lint(self):
        workflow = _load_workflow()
        needs = workflow["jobs"]["test"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "lint" in needs, "test job must depend on lint via `needs`"


class TestDockerBuildJob:
    """Verify docker-build job depends on test and only runs on main branch."""

    def _docker_job(self, workflow: dict) -> dict:
        # Accept either name to be tolerant of future renames.
        for name in ("docker-build", "docker"):
            if name in workflow["jobs"]:
                return workflow["jobs"][name]
        raise AssertionError("Must have a docker-build (or docker) job")

    def test_docker_job_exists(self):
        workflow = _load_workflow()
        self._docker_job(workflow)

    def test_docker_job_depends_on_test_job(self):
        workflow = _load_workflow()
        docker_job = self._docker_job(workflow)
        needs = docker_job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "test" in needs, "docker-build job must depend on test job via 'needs'"

    def test_docker_job_runs_only_on_main(self):
        workflow = _load_workflow()
        docker_job = self._docker_job(workflow)
        condition = docker_job.get("if", "")
        assert "main" in condition, (
            f"docker-build job must have 'if' condition restricting to main branch; got: {condition}"
        )

    def test_docker_build_step_exists(self):
        workflow = _load_workflow()
        docker_job = self._docker_job(workflow)
        cmds = _all_run_commands(docker_job)
        assert "docker build" in cmds, "docker-build job must have a docker build step"


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

    def test_critical_steps_do_not_continue_on_error(self):
        workflow = _load_workflow()
        for job_name in ("lint", "test"):
            steps = workflow["jobs"][job_name]["steps"]
            critical = self._get_critical_steps(steps)
            for step in critical:
                assert not step.get("continue-on-error", False), (
                    f"Critical step '{step.get('name', step.get('run', ''))}' "
                    f"in {job_name} must not have continue-on-error: true"
                )

    def test_test_job_has_no_fail_fast_false(self):
        """Verify strategy.fail-fast is not set to false (default is true)."""
        workflow = _load_workflow()
        test_job = workflow["jobs"]["test"]
        strategy = test_job.get("strategy", {})
        fail_fast = strategy.get("fail-fast", True)
        assert fail_fast is not False, "test job strategy.fail-fast must not be false"
