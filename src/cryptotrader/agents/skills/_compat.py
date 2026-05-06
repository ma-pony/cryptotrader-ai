"""兼容性工具函数。"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow_str() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(UTC).strftime("%Y-%m-%d")
