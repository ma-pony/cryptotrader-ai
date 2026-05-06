"""项目级常量 — Agent Skills 子模块。

FR-004c: VALID_AGENT_IDS 是项目级常量，与 nodes/agents.py / verdict / risk gate 等保持一致。
"""

from pathlib import Path

# 有效的 agent ID 集合（不动态化，是项目结构常量）
VALID_AGENT_IDS: frozenset[str] = frozenset({"tech", "chain", "news", "macro"})

# 默认目录路径（相对仓库根）
DEFAULT_AGENT_SKILLS_DIR = Path("agent_skills")
DEFAULT_AGENT_MEMORY_DIR = Path("agent_memory")

# skill 目录名 → scope 默认映射（initial 5 个 skill 的命名约定）
# 注意：middleware 不使用此映射；仅供测试/文档参考
_INITIAL_SKILL_DIRS = [
    "tech-analysis",
    "chain-analysis",
    "news-analysis",
    "macro-analysis",
    "trading-knowledge",
]
