"""Skill selector — regime matching and token budget enforcement."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skill_loader import SkillLoader
    from cryptotrader.config import AgentConfig

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_BUDGET_CHARS = 8000


class SkillSelector:
    def select(
        self,
        agent_cfg: AgentConfig,
        regime_tags: list[str],
        loader: SkillLoader,
        token_budget_chars: int = _DEFAULT_TOKEN_BUDGET_CHARS,
    ) -> list[str]:
        skill_names: list[str] = list(agent_cfg.skills)

        for tag, tag_skills in agent_cfg.regime_skills.items():
            if tag in regime_tags:
                for name in tag_skills:
                    if name not in skill_names:
                        skill_names.append(name)

        if not skill_names:
            return []

        results: list[str] = []
        total_chars = 0
        for name in skill_names:
            content = loader.load(name)
            if not content:
                continue
            if total_chars + len(content) > token_budget_chars:
                remaining = token_budget_chars - total_chars
                if remaining > 0:
                    results.append(content[:remaining])
                    logger.warning(
                        "skill budget exceeded, truncating after %d chars",
                        token_budget_chars,
                    )
                break
            results.append(content)
            total_chars += len(content)

        return results
