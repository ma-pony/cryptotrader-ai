"""Tests for architecture task 8.2 -- ruff module boundary static checks.

Verifies:
1. pyproject.toml [tool.ruff.lint.flake8-tidy-imports.banned-api] contains
   cryptotrader.nodes and cryptotrader.graph as banned imports
2. TCH rule is enabled in [tool.ruff.lint.select]
3. TID rule is enabled in [tool.ruff.lint.select]
4. per-file-ignores excludes TID251 for nodes/ and other allowed files
5. Domain layer files (agents/, debate/, execution/, risk/, learning/) do NOT
   import from cryptotrader.nodes or cryptotrader.graph
6. Running ruff on a domain-layer file with a forbidden import raises TID251
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
import textwrap
import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"
SRC_DIR = Path(__file__).parent.parent / "src" / "cryptotrader"

DOMAIN_LAYERS = [
    "agents",
    "debate",
    "execution",
    "risk",
    "learning",
]

FORBIDDEN_PREFIXES = ("cryptotrader.nodes", "cryptotrader.graph")


def _load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


# ── Test 1: banned-api config is present ──────────────────────────────────────


class TestBannedApiConfig:
    """Verify banned-api is configured in pyproject.toml."""

    def test_flake8_tidy_imports_banned_api_section_exists(self):
        data = _load_pyproject()
        ruff_lint = data["tool"]["ruff"]["lint"]
        assert "flake8-tidy-imports" in ruff_lint, "[tool.ruff.lint.flake8-tidy-imports] section must exist"
        tid_cfg = ruff_lint["flake8-tidy-imports"]
        assert "banned-api" in tid_cfg, "[tool.ruff.lint.flake8-tidy-imports.banned-api] must be configured"

    def test_cryptotrader_nodes_is_banned(self):
        data = _load_pyproject()
        banned = data["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
        assert "cryptotrader.nodes" in banned, (
            "cryptotrader.nodes must be in banned-api to prevent domain-layer reverse imports"
        )

    def test_cryptotrader_graph_is_banned(self):
        data = _load_pyproject()
        banned = data["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
        assert "cryptotrader.graph" in banned, (
            "cryptotrader.graph must be in banned-api to prevent domain-layer reverse imports"
        )

    def test_banned_api_has_msg_for_nodes(self):
        data = _load_pyproject()
        banned = data["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
        nodes_entry = banned.get("cryptotrader.nodes", {})
        assert "msg" in nodes_entry, "banned-api entry for cryptotrader.nodes must have a 'msg' field"

    def test_banned_api_has_msg_for_graph(self):
        data = _load_pyproject()
        banned = data["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
        graph_entry = banned.get("cryptotrader.graph", {})
        assert "msg" in graph_entry, "banned-api entry for cryptotrader.graph must have a 'msg' field"


# ── Test 2: TCH and TID rules are enabled ─────────────────────────────────────


class TestRuffRulesEnabled:
    """Verify TCH and TID rules are present in ruff lint select list."""

    def test_tid_rule_is_selected(self):
        data = _load_pyproject()
        select = data["tool"]["ruff"]["lint"]["select"]
        assert "TID" in select, "TID (flake8-tidy-imports) must be in [tool.ruff.lint.select]"

    def test_tch_rule_is_selected(self):
        data = _load_pyproject()
        select = data["tool"]["ruff"]["lint"]["select"]
        assert "TCH" in select, "TCH (flake8-type-checking) must be in [tool.ruff.lint.select]"


# ── Test 3: nodes/ per-file-ignores ───────────────────────────────────────────


class TestPerFileIgnores:
    """Verify nodes/ and graph.py are excluded from TID251 ban."""

    def test_nodes_per_file_ignores_tid251(self):
        data = _load_pyproject()
        per_file = data["tool"]["ruff"]["lint"].get("per-file-ignores", {})
        nodes_keys = [k for k in per_file if "nodes" in k]
        assert nodes_keys, "nodes/*.py must appear in per-file-ignores to exempt from TID251"
        nodes_ignored = [rule for k in nodes_keys for rule in per_file[k]]
        assert any("TID251" in v or "TID" in v for v in nodes_ignored), "TID251 must be ignored for nodes/*.py files"

    def test_graph_py_per_file_ignores_tid251(self):
        data = _load_pyproject()
        per_file = data["tool"]["ruff"]["lint"].get("per-file-ignores", {})
        graph_keys = [k for k in per_file if "graph" in k]
        assert graph_keys, "graph.py (or graph_supervisor.py) must appear in per-file-ignores to exempt from TID251"


# ── Test 4: Domain layer has no imports from nodes/ or graph.py ───────────────


class TestDomainLayerImports:
    """Verify domain layer files contain no reverse imports from nodes/ or graph."""

    def _get_all_python_files(self, layer: str) -> list[Path]:
        layer_dir = SRC_DIR / layer
        if not layer_dir.exists():
            return []
        return list(layer_dir.rglob("*.py"))

    def _file_imports_forbidden_module(self, filepath: Path) -> list[str]:
        """Return list of forbidden import strings found in the file."""
        source = filepath.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        import_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
        return [
            f"line {n.lineno}: import {alias.name}"
            for n in import_nodes
            if isinstance(n, ast.Import)
            for alias in n.names
            if alias.name.startswith(FORBIDDEN_PREFIXES)
        ] + [
            f"line {n.lineno}: from {n.module} import ..."
            for n in import_nodes
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(FORBIDDEN_PREFIXES)
        ]

    def test_agents_no_nodes_imports(self):
        files = self._get_all_python_files("agents")
        assert files, "agents/ directory must contain Python files"
        violations = {str(f.relative_to(SRC_DIR)): v for f in files if (v := self._file_imports_forbidden_module(f))}
        assert not violations, "agents/ must not import from cryptotrader.nodes or cryptotrader.graph:\n" + "\n".join(
            f"  {k}: {v}" for k, v in violations.items()
        )

    def test_debate_no_nodes_imports(self):
        files = self._get_all_python_files("debate")
        for f in files:
            v = self._file_imports_forbidden_module(f)
            assert not v, f"debate/{f.name} must not import from cryptotrader.nodes or cryptotrader.graph: {v}"

    def test_execution_no_nodes_imports(self):
        files = self._get_all_python_files("execution")
        for f in files:
            v = self._file_imports_forbidden_module(f)
            assert not v, f"execution/{f.name} must not import from cryptotrader.nodes or cryptotrader.graph: {v}"

    def test_risk_no_nodes_imports(self):
        files = self._get_all_python_files("risk")
        for f in files:
            v = self._file_imports_forbidden_module(f)
            assert not v, f"risk/{f.name} must not import from cryptotrader.nodes or cryptotrader.graph: {v}"

    def test_learning_no_nodes_imports(self):
        files = self._get_all_python_files("learning")
        for f in files:
            v = self._file_imports_forbidden_module(f)
            assert not v, f"learning/{f.name} must not import from cryptotrader.nodes or cryptotrader.graph: {v}"


# ── Test 5: ruff actually catches TID251 in domain-layer-like files ────────────


def _ruff_bin() -> str:
    """Return path to ruff executable, raising if not found."""
    ruff = shutil.which("ruff")
    assert ruff is not None, "ruff binary not found on PATH -- install ruff"
    return ruff


class TestRuffTid251Enforcement:
    """Verify ruff raises TID251 when a domain-layer file imports from nodes/."""

    def test_ruff_detects_nodes_import_in_domain_file(self):
        """A temporary file simulating a domain-layer import must trigger TID251."""
        forbidden_code = textwrap.dedent("""\
            from cryptotrader.nodes import data
        """)

        with tempfile.NamedTemporaryFile(
            suffix=".py",
            mode="w",
            delete=False,
            dir=SRC_DIR / "agents",
        ) as tmp:
            tmp.write(forbidden_code)
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(  # nosec S603
                [_ruff_bin(), "check", "--select=TID251", str(tmp_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )
            # ruff returns exit code 1 when violations are found
            assert result.returncode != 0, (
                "ruff must detect TID251 violation for 'from cryptotrader.nodes import data' "
                f"in a domain-layer file.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "TID251" in result.stdout, f"ruff output must mention TID251. Got:\n{result.stdout}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_ruff_detects_graph_import_in_domain_file(self):
        """A temporary file simulating a domain-layer import of graph must trigger TID251."""
        forbidden_code = textwrap.dedent("""\
            from cryptotrader.graph import build_trading_graph
        """)

        with tempfile.NamedTemporaryFile(
            suffix=".py",
            mode="w",
            delete=False,
            dir=SRC_DIR / "agents",
        ) as tmp:
            tmp.write(forbidden_code)
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(  # nosec S603
                [_ruff_bin(), "check", "--select=TID251", str(tmp_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                check=False,
            )
            assert result.returncode != 0, (
                "ruff must detect TID251 violation for 'from cryptotrader.graph import ...' "
                f"in a domain-layer file.\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert "TID251" in result.stdout, f"ruff output must mention TID251. Got:\n{result.stdout}"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_ruff_clean_on_existing_domain_files(self):
        """Existing domain-layer files must have zero TID251 violations."""
        domain_dirs = [str(SRC_DIR / layer) for layer in DOMAIN_LAYERS if (SRC_DIR / layer).exists()]

        assert domain_dirs, "At least one domain layer directory must exist"

        result = subprocess.run(  # nosec S603
            [_ruff_bin(), "check", "--select=TID251", *domain_dirs],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )
        assert result.returncode == 0, (
            "Existing domain-layer files must have no TID251 violations.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
