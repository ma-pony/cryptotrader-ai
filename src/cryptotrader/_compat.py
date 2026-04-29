"""Python version compatibility shims."""

from __future__ import annotations

import sys
from datetime import timezone
from enum import Enum

if sys.version_info >= (3, 11):
    from datetime import UTC
    from enum import StrEnum
else:
    UTC = timezone.utc

    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042 -- StrEnum backport (Py 3.10 lacks enum.StrEnum)
        """Minimal StrEnum backport for Python 3.10."""
