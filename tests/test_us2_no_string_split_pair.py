"""US2 (spec 013, T027) — regression: no `.split("/")` in pair context.

Failure means a developer added a new ad-hoc string-split on a pair
somewhere outside Pair / pair_adapter (deleted) — re-route through
``Pair.parse(pair).base`` instead.

Walks the source tree directly with stdlib (no subprocess) so the test
is portable and dependency-free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

_SPLIT_SLASH = re.compile(r'\.split\(\s*"\s*/\s*"\s*\)')
_SPLIT_COLON = re.compile(r"\.split\(\s*['\"]\s*:\s*['\"]\s*\)")


def _scan(pattern: re.Pattern[str], *, exclude_basenames: set[str]) -> list[str]:
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name in exclude_basenames:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    return hits


def test_no_string_split_pair_in_src() -> None:
    leaks = _scan(_SPLIT_SLASH, exclude_basenames={"pair.py"})
    if leaks:
        msg = "Use Pair.parse(pair).base instead of pair.split('/'):\n" + "\n".join(leaks)
        pytest.fail(msg)


def test_pair_adapter_is_removed() -> None:
    """T026 deleted the transient adapter once nodes consume Pair directly."""
    assert not (REPO / "src" / "cryptotrader" / "pair_adapter.py").exists()
    assert not (REPO / "tests" / "test_pair_adapter.py").exists()


def test_no_pair_split_colon_in_src() -> None:
    """`pair.split(':')` is the second risky pattern — derivatives split."""
    raw = _scan(_SPLIT_COLON, exclude_basenames={"pair.py"})
    pair_context = [
        line
        for line in raw
        # telegram callback data, file paths, etc. — not pair context
        if not re.search(r"telegram|callback", line, re.IGNORECASE)
    ]
    if pair_context:
        msg = "Use Pair.parse(pair) (not split(':')) for derivatives:\n" + "\n".join(pair_context)
        pytest.fail(msg)
