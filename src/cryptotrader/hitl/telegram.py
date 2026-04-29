"""Optional Telegram Bot for HITL approval notifications."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.config import HitlTelegramConfig

logger = logging.getLogger(__name__)

_TRIGGER_LABELS = {
    "position_scale": "大额仓位",
    "divergence": "Agent 分歧",
    "cold_start": "冷启动保护",
}


def _require_telegram():
    try:
        import telegram

        return telegram
    except ImportError as exc:
        raise RuntimeError("python-telegram-bot 未安装。请运行: uv add 'python-telegram-bot>=21.0'") from exc


async def send_approval_notification(
    approval_id: str,
    pair: str,
    trigger_reason: str,
    verdict: dict[str, Any],
    tg_config: HitlTelegramConfig,
) -> int | None:
    """Send a Telegram message with inline approve/reject buttons.

    Returns the message_id on success, None on failure.
    """
    tg = _require_telegram()

    action = verdict.get("action", "unknown")
    position_scale = verdict.get("position_scale", 0)
    confidence = verdict.get("confidence", 0)
    reasoning = (verdict.get("reasoning") or "")[:200]
    label = _TRIGGER_LABELS.get(trigger_reason, trigger_reason)

    text = (
        f"🔔 *HITL 审批请求*\n\n"
        f"*币对*: `{pair}`\n"
        f"*触发原因*: {label}\n"
        f"*裁决*: {action} (仓位 {position_scale:.0%}, 置信度 {confidence:.0%})\n"
        f"*摘要*: {reasoning}\n\n"
        f"请在超时前响应。"
    )

    keyboard = tg.InlineKeyboardMarkup(
        [
            [
                tg.InlineKeyboardButton("✅ 批准", callback_data=f"hitl:{approval_id}:approve"),
                tg.InlineKeyboardButton("❌ 拒绝", callback_data=f"hitl:{approval_id}:reject"),
            ]
        ]
    )

    try:
        bot = tg.Bot(token=tg_config.bot_token)
        async with bot:
            msg = await bot.send_message(
                chat_id=tg_config.chat_id,
                text=text,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
            return msg.message_id
    except Exception:
        logger.warning("Failed to send Telegram HITL message", exc_info=True)
        return None


async def update_telegram_message(
    tg_config: HitlTelegramConfig,
    message_id: int,
    decision: str,
    decision_by: str,
) -> None:
    """Edit an existing Telegram message to show the decision result."""
    tg = _require_telegram()

    if decision == "approve":
        status_text = f"✅ 已批准 (by {decision_by})"
    elif decision == "expired":
        status_text = "⏰ 已超时 — 自动拒绝"
    else:
        status_text = f"❌ 已拒绝 (by {decision_by})"

    try:
        bot = tg.Bot(token=tg_config.bot_token)
        async with bot:
            await bot.edit_message_text(
                chat_id=tg_config.chat_id,
                message_id=message_id,
                text=status_text,
            )
    except Exception:
        logger.debug("Failed to update Telegram HITL message", exc_info=True)


class TelegramApprovalBot:
    """Long-polling Telegram bot that handles inline approve/reject callbacks."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        db_url: str,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._db_url = db_url
        self._app: Any = None

    async def start(self) -> None:
        _require_telegram()
        from telegram.ext import Application, CallbackQueryHandler

        self._app = Application.builder().token(self._bot_token).build()
        self._app.add_handler(CallbackQueryHandler(self._handle_callback, pattern=r"^hitl:"))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram HITL bot started polling")

    async def stop(self) -> None:
        if self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram HITL bot stopped")

    async def _handle_callback(self, update: Any, _context: Any) -> None:
        query = update.callback_query
        if query is None:
            return

        data = query.data or ""
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != "hitl":
            await query.answer("Invalid callback")
            return

        _, approval_id, decision = parts
        if decision not in ("approve", "reject"):
            await query.answer("Invalid decision")
            return

        from cryptotrader.hitl.store import ApprovalStore

        ok = await ApprovalStore.decide(
            self._db_url,
            approval_id,
            status="approved" if decision == "approve" else "rejected",
            decision_by="telegram",
        )

        if not ok:
            await query.answer("Already decided or expired")
            return

        await query.answer("Decision recorded")

        status_text = "✅ 已批准 (by telegram)" if decision == "approve" else "❌ 已拒绝 (by telegram)"

        try:
            await query.edit_message_text(text=status_text)
        except Exception:
            logger.debug("Failed to edit callback message", exc_info=True)

        from cryptotrader.hitl.notifier import notify_hitl_decision

        record = await ApprovalStore.get(self._db_url, approval_id)
        pair = record["pair"] if record else "unknown"
        trigger_reason = record["trigger_reason"] if record else "unknown"
        await notify_hitl_decision(
            approval_id=approval_id,
            pair=pair,
            trigger_reason=trigger_reason,
            decision=decision,
            decision_by="telegram",
            latency_seconds=0.0,
        )
