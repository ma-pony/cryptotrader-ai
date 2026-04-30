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

# Match `.split("/")` and `.split("/", N)` forms — the second form is the
# natural Python idiom and was the escape hatch that let Phase 4 fallback
# guards slip past the original regex (deep-review test FINDING-1).
_SPLIT_SLASH = re.compile(r'\.split\(\s*["\']\s*/\s*["\']')
_SPLIT_COLON = re.compile(r"\.split\(\s*['\"]\s*:\s*['\"]")

# Files with intentional defensive fallback to .split("/", 1) inside a try/except
# guarded by Pair.parse — these are NOT regressions, they are belt-and-suspenders.
_FALLBACK_ALLOWLIST = {
    "agents/data_tools.py",
    "data/news.py",
    "data/onchain.py",
    "data/snapshot.py",
    "risk/checks/correlation.py",
}


_PAIR_TOKEN = re.compile(r"\bpair\b|\bpos_pair\b|\bself\.pair\b|\bo(rder)?\.pair\b|\bp\.pair\b")


def _scan(pattern: re.Pattern[str], *, exclude_basenames: set[str]) -> list[str]:
    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name in exclude_basenames:
            continue
        # Skip files in the fallback allowlist (Phase 4 defensive guards).
        rel = str(path.relative_to(REPO).as_posix())
        if any(rel.endswith(suffix) for suffix in (f"src/cryptotrader/{p}" for p in _FALLBACK_ALLOWLIST)):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Only flag if the line both does a split("/") AND mentions a pair-like token.
            # Eliminates false positives from `entry.model_id.split("/")` and similar.
            if pattern.search(line) and _PAIR_TOKEN.search(line):
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
