"""Tests for pyproject.toml configuration correctness (Task 7.1).

Validates:
- pytest-cov coverage flags in addopts
- test optional-dependency group exists with required packages
- dev group contains only dev tooling (ruff, pre-commit)
- test group is separate from dev group
"""

import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


class TestPytestAddopts:
    """Verify pytest addopts includes coverage flags."""

    def test_addopts_present(self):
        data = _load_pyproject()
        addopts = data["tool"]["pytest"]["ini_options"].get("addopts", "")
        assert addopts, "addopts must not be empty"

    def test_addopts_has_cov_src(self):
        data = _load_pyproject()
        addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov=src" in addopts

    def test_addopts_has_cov_report_term_missing(self):
        data = _load_pyproject()
        addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov-report=term-missing" in addopts

    def test_addopts_has_cov_fail_under_70(self):
        data = _load_pyproject()
        addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov-fail-under=70" in addopts

    def test_addopts_has_cov_branch(self):
        data = _load_pyproject()
        addopts = data["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov-branch" in addopts


class TestOptionalDependencies:
    """Verify optional-dependency groups are correctly structured."""

    def test_test_group_exists(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        assert "test" in optional_deps, "test optional-dependency group must exist"

    def test_test_group_has_pytest_cov(self):
        data = _load_pyproject()
        test_deps = data["project"]["optional-dependencies"]["test"]
        has_cov = any("pytest-cov" in dep for dep in test_deps)
        assert has_cov, "test group must include pytest-cov>=5.0"

    def test_test_group_has_pytest(self):
        data = _load_pyproject()
        test_deps = data["project"]["optional-dependencies"]["test"]
        has_pytest = any(dep.startswith("pytest>=") or dep == "pytest" for dep in test_deps)
        assert has_pytest, "test group must include pytest"

    def test_test_group_has_pytest_asyncio(self):
        data = _load_pyproject()
        test_deps = data["project"]["optional-dependencies"]["test"]
        has_asyncio = any("pytest-asyncio" in dep for dep in test_deps)
        assert has_asyncio, "test group must include pytest-asyncio"

    def test_dev_group_does_not_contain_pytest(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        if "dev" not in optional_deps:
            return  # dev group may not exist as optional-dependency
        dev_deps = optional_deps["dev"]
        has_pytest = any(dep.startswith("pytest") for dep in dev_deps)
        assert not has_pytest, "dev optional-dependency group must not contain pytest packages"

    def test_dev_group_has_ruff(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        assert "dev" in optional_deps, "dev optional-dependency group must exist"
        dev_deps = optional_deps["dev"]
        has_ruff = any("ruff" in dep for dep in dev_deps)
        assert has_ruff, "dev group must include ruff"

    def test_dev_group_has_pre_commit(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        dev_deps = optional_deps["dev"]
        has_precommit = any("pre-commit" in dep for dep in dev_deps)
        assert has_precommit, "dev group must include pre-commit"

    def test_otel_group_exists(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        assert "otel" in optional_deps, "otel optional-dependency group must exist"

    def test_otel_group_not_duplicated(self):
        data = _load_pyproject()
        optional_deps = data["project"]["optional-dependencies"]
        # Just verify otel group is a list (not duplicated key - TOML forbids that)
        assert isinstance(optional_deps.get("otel"), list)

    def test_pytest_cov_version_requirement(self):
        data = _load_pyproject()
        test_deps = data["project"]["optional-dependencies"]["test"]
        cov_deps = [dep for dep in test_deps if "pytest-cov" in dep]
        assert len(cov_deps) == 1
        # Must be >=5.0
        dep = cov_deps[0]
        assert ">=5.0" in dep, f"pytest-cov must specify >=5.0, got: {dep}"
