"""CI guard for logging conventions (docs/logging-conventions.md).

Forbids `logger.debug(..., exc_info=True)` because at default LOG_LEVEL=INFO
those exceptions are silently dropped. See `portfolio_unknown` /
`redis_unavailable` post-mortem 2026-05-06.
"""

from __future__ import annotations

import pathlib
import re

_SRC = pathlib.Path(__file__).parent.parent / "src"

# Match `<anything>.debug(...exc_info=True...)` allowing for line-wrapped args.
# We scan whole-file content so multi-line debug() calls are also caught.
_PATTERN = re.compile(r"\.debug\s*\([^)]*exc_info\s*=\s*True", re.DOTALL)


def test_no_debug_swallowing_exceptions():
    """No production code under src/ may use logger.debug(..., exc_info=True).

    Use logger.warning(..., exc_info=True) for critical-path exceptions
    (trading / risk / orders / portfolio / exchange) and logger.info(..., exc_info=True)
    for side-effect failures (notifications / UI / dashboard / scheduler housekeeping).
    See docs/logging-conventions.md for the full convention.
    """
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in _PATTERN.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{py.relative_to(_SRC.parent)}:{line_no}")

    assert not offenders, (
        "logger.debug(..., exc_info=True) swallows exceptions silently at "
        "default LOG_LEVEL=INFO. Upgrade to .warning() (critical path) or "
        ".info() (side-effect path). See docs/logging-conventions.md.\n"
        "Offending locations:\n  " + "\n  ".join(offenders)
    )
