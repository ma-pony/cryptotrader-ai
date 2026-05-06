"""HITL notification integration — Telegram + existing Notifier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from cryptotrader.config import HitlConfig

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


async def notify_hitl_request(
    approval_id: str,
    pair: str,
    trigger_reason: str,
    verdict: dict[str, Any],
    analyses: dict[str, Any],
    hitl_config: HitlConfig,
) -> None:
    """Send approval request notifications via configured channels."""
    if hitl_config.telegram.enabled and hitl_config.telegram.bot_token:
        try:
            from cryptotrader.hitl.telegram import send_approval_notification

            await send_approval_notification(
                approval_id,
                pair,
                trigger_reason,
                verdict,
                hitl_config.telegram,
            )
        except Exception:
            logger.warning("Failed to send Telegram HITL notification", exc_info=True)


async def notify_hitl_decision(
    approval_id: str,
    pair: str,
    trigger_reason: str,
    decision: str,
    decision_by: str,
    latency_seconds: float,
) -> None:
    """Log structured decision event and notify via existing Notifier."""
    _slog.info(
        "hitl_decision",
        approval_id=approval_id,
        pair=pair,
        trigger_reason=trigger_reason,
        decision=decision,
        decision_by=decision_by,
        latency_seconds=round(latency_seconds, 2),
    )

    try:
        from cryptotrader.config import load_config

        config = load_config()
        if config.notifications.enabled and "hitl_decision" in config.notifications.events:
            from cryptotrader.notifications import Notifier

            notifier = Notifier(config.notifications)
            await notifier.send(
                "hitl_decision",
                {
                    "approval_id": approval_id,
                    "pair": pair,
                    "trigger_reason": trigger_reason,
                    "decision": decision,
                    "decision_by": decision_by,
                    "latency_seconds": round(latency_seconds, 2),
                },
            )
    except Exception:
        logger.info("Notifier dispatch failed for hitl_decision", exc_info=True)
