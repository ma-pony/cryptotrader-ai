"""项目级常量 — Agent Skills 子模块。"""

from pathlib import Path

VALID_AGENT_IDS: frozenset[str] = frozenset({"tech", "chain", "news", "macro"})

DEFAULT_AGENT_SKILLS_DIR = Path("agent_skills")

# initial 5 个 skill 目录名（ensure_skill_dirs 用）
_INITIAL_SKILL_DIRS = [
    "tech-analysis",
    "chain-analysis",
    "news-analysis",
    "macro-analysis",
    "trading-knowledge",
]
