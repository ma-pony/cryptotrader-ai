"""Test that every verify=False HTTPS call has a nosec S501 comment on the same line.

This test enforces task 6.3: all httpx.AsyncClient(verify=False) calls in
data/sync.py and data/providers/sosovalue.py must carry a # nosec S501 comment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Files that must have nosec comments on every verify=False line
TARGET_FILES = [
    Path(__file__).parent.parent / "src" / "cryptotrader" / "data" / "sync.py",
    Path(__file__).parent.parent / "src" / "cryptotrader" / "data" / "providers" / "sosovalue.py",
]

# Pattern matching a line with verify=False that lacks the required nosec comment
_VERIFY_FALSE_RE = re.compile(r"verify\s*=\s*False")
_NOSEC_S501_RE = re.compile(r"#\s*nosec\s+S501")


def _lines_missing_nosec(filepath: Path) -> list[tuple[int, str]]:
    """Return (line_number, line_content) for every verify=False line without nosec S501."""
    missing: list[tuple[int, str]] = []
    text = filepath.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _VERIFY_FALSE_RE.search(line) and not _NOSEC_S501_RE.search(line):
            missing.append((lineno, line.rstrip()))
    return missing


@pytest.mark.parametrize("filepath", TARGET_FILES, ids=lambda p: p.name)
def test_verify_false_has_nosec_comment(filepath: Path) -> None:
    """Every verify=False call must carry a # nosec S501 comment on the same line."""
    assert filepath.exists(), f"Target file not found: {filepath}"
    missing = _lines_missing_nosec(filepath)
    if missing:
        details = "\n".join(f"  line {ln}: {content}" for ln, content in missing)
        pytest.fail(f"{filepath.name}: {len(missing)} verify=False line(s) missing # nosec S501:\n{details}")


def test_all_target_files_have_verify_false_calls() -> None:
    """Sanity check: target files actually contain verify=False calls (guard against renames)."""
    for filepath in TARGET_FILES:
        assert filepath.exists(), f"Target file missing: {filepath}"
        text = filepath.read_text(encoding="utf-8")
        assert _VERIFY_FALSE_RE.search(text), (
            f"{filepath.name} contains no verify=False calls — "
            "file may have been renamed or refactored; update TARGET_FILES in this test"
        )
