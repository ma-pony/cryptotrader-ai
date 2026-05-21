"""Evolution Daemon — background action hooks for pattern extraction, skill proposals, Pareto rerank.

spec 022 FR-022-16 / T010: wires `journal.record_evolution_event()` at 3 daemon hooks:
  - _action_pattern_extraction: after new pattern created  (subtype="new_pattern")
  - _action_skill_proposal:     after .draft written        (subtype="new_skill_draft")
  - _action_pareto_rerank:      after Pareto rerank done    (subtype="pareto_rerank")

All hooks soft-fail: journal errors are logged but never propagate.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def _action_pattern_extraction(
    pattern_slug: str,
    agent: str,
    *,
    extra: dict[str, Any] | None = None,
    database_url: str | None = None,
) -> None:
    """Hook called after a new pattern is created by the evolution daemon.

    Writes a `new_pattern` evolution_event to the journal table.

    Args:
        pattern_slug: the slug of the newly created/updated pattern.
        agent: the agent name (tech/chain/news/macro).
        extra: optional extra payload fields.
        database_url: PostgreSQL URL; resolved from config if None.
    """
    try:
        from cryptotrader.journal.events import record_evolution_event

        payload: dict[str, Any] = {"agent": agent}
        if extra:
            payload.update(extra)
        await record_evolution_event(
            event_subtype="new_pattern",
            artifact_name=pattern_slug,
            payload=payload,
            database_url=database_url,
        )
        logger.debug("evolution_event new_pattern: agent=%s slug=%s", agent, pattern_slug)
    except Exception:
        logger.warning("_action_pattern_extraction journal hook failed (soft-fail)", exc_info=True)


async def _action_skill_proposal(
    skill_draft_name: str,
    agent: str,
    *,
    extra: dict[str, Any] | None = None,
    database_url: str | None = None,
) -> None:
    """Hook called after a new skill draft (.draft) is written.

    Writes a `new_skill_draft` evolution_event to the journal table.

    Args:
        skill_draft_name: the name of the draft skill file (e.g. "sma-breakout.draft").
        agent: the agent name (tech/chain/news/macro).
        extra: optional extra payload fields.
        database_url: PostgreSQL URL; resolved from config if None.
    """
    try:
        from cryptotrader.journal.events import record_evolution_event

        payload: dict[str, Any] = {"agent": agent}
        if extra:
            payload.update(extra)
        await record_evolution_event(
            event_subtype="new_skill_draft",
            artifact_name=skill_draft_name,
            payload=payload,
            database_url=database_url,
        )
        logger.debug("evolution_event new_skill_draft: agent=%s name=%s", agent, skill_draft_name)
    except Exception:
        logger.warning("_action_skill_proposal journal hook failed (soft-fail)", exc_info=True)


async def _action_pareto_rerank(
    reranked_count: int,
    *,
    extra: dict[str, Any] | None = None,
    database_url: str | None = None,
) -> None:
    """Hook called after Pareto rerank is done.

    Writes a `pareto_rerank` evolution_event to the journal table.

    Args:
        reranked_count: number of patterns/skills reranked in this pass.
        extra: optional extra payload fields.
        database_url: PostgreSQL URL; resolved from config if None.
    """
    try:
        from cryptotrader.journal.events import record_evolution_event

        payload: dict[str, Any] = {"reranked_count": reranked_count}
        if extra:
            payload.update(extra)
        await record_evolution_event(
            event_subtype="pareto_rerank",
            artifact_name=f"pareto_rerank_{reranked_count}",
            payload=payload,
            database_url=database_url,
        )
        logger.debug("evolution_event pareto_rerank: count=%d", reranked_count)
    except Exception:
        logger.warning("_action_pareto_rerank journal hook failed (soft-fail)", exc_info=True)
