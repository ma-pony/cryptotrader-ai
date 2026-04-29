"""Build multimodal LangChain messages from chart analysis context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.config import AppConfig

logger = logging.getLogger(__name__)


def build_multimodal_messages(
    payloads: list[dict[str, Any]],
    model: str,
    config: AppConfig,
) -> tuple[list, bool]:
    """Build LangChain-compatible messages from chart capture payloads.

    Returns (messages, degraded) where degraded=True when an image was
    too large and was dropped.
    """
    vision_models = config.chart_analysis.vision_models
    max_bytes = config.chart_analysis.max_image_bytes
    supports_vision = model in vision_models

    content_blocks: list[str | dict[str, Any]] = []
    degraded = False

    for payload in payloads:
        timeframe = payload.get("timeframe", "unknown")
        description = payload.get("description", "")
        data_url: str | None = payload.get("dataUrl")

        if len(payloads) > 1:
            content_blocks.append(
                {
                    "type": "text",
                    "text": f"=== 时间周期: {timeframe} ===",
                }
            )

        if supports_vision and data_url is not None:
            url_byte_len = len(data_url.encode("utf-8"))
            if url_byte_len <= max_bytes:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )
            else:
                degraded = True
                logger.info("Image too large (%d bytes > %d), using text only", url_byte_len, max_bytes)

        if description:
            content_blocks.append({"type": "text", "text": description})

    from langchain_core.messages import HumanMessage

    return [HumanMessage(content=content_blocks)], degraded
