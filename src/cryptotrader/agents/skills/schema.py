"""Skill data model — single source of truth for the SKILL.md schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Skill:
    """高层能力包，对应 agent_skills/<name>/SKILL.md。

    遵循 Anthropic Skills 协议；运行时由 EvolvingSkillProvider 加载。
    """

    name: str
    description: str
    scope: str  # "shared" | "agent:<agent_id>"
    body: str
    file_path: Path = field(default_factory=lambda: Path("."))
    manually_edited: bool = False
    version: str = "1.0"
    mtime: float = 0.0
    regime_tags: list[str] = field(default_factory=list)
    triggers_keywords: list[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5

    @property
    def agent_id(self) -> str | None:
        if self.scope.startswith("agent:"):
            return self.scope.split(":", 1)[1]
        return None

    @property
    def is_shared(self) -> bool:
        return self.scope == "shared"


@dataclass
class AgentSkillSet:
    """单 agent 一次 cycle 加载的 skill 集合（动态 list[Skill]）。"""

    agent_id: str
    skills: list[Skill] = field(default_factory=list)

    def get_combined_body(self) -> str:
        parts = []
        for skill in self.skills:
            parts.append(f"\n\n## Skill: {skill.name}\n\n{skill.body}")
        return "".join(parts)
