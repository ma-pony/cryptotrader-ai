"""File-system Skill loader — loads Markdown skill files from search paths."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_SEARCH_PATHS = [
    _PROJECT_ROOT / "skills",
    Path.home() / ".cryptotrader" / "skills",
]


class SkillLoader:
    def __init__(self, search_paths: list[Path] | None = None) -> None:
        self.search_paths = search_paths or list(_DEFAULT_SEARCH_PATHS)

    def _find(self, skill_name: str) -> Path | None:
        for base in self.search_paths:
            candidate = base / f"{skill_name}.md"
            if candidate.exists():
                return candidate
        return None

    def load(self, skill_name: str) -> str:
        path = self._find(skill_name)
        if path is None:
            logger.warning("skill '%s' not found in search paths", skill_name)
            return ""
        try:
            from cryptotrader.security import sanitize_input

            content = path.read_text(encoding="utf-8")
            return sanitize_input(content, max_chars=8000)
        except Exception:
            logger.warning("failed to load skill '%s'", skill_name, exc_info=True)
            return ""
