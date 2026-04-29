"""MCP utility functions — response truncation and API key redaction."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 51200  # 50 KB


def truncate_response(data: dict, max_bytes: int = _MAX_RESPONSE_BYTES) -> dict:
    serialized = json.dumps(data, default=str)
    if len(serialized.encode()) <= max_bytes:
        return data

    result = dict(data)
    for key, value in result.items():
        if isinstance(value, list) and len(value) > 1:
            while len(json.dumps(result, default=str).encode()) > max_bytes and len(result[key]) > 1:
                result[key] = result[key][: len(result[key]) // 2]
            result["truncated"] = True
            break

    if "truncated" not in result:
        result["truncated"] = True

    return result


_KEY_PATTERN = re.compile(r"[A-Za-z0-9]{20,}")


def redact_api_key(text: str) -> str:
    return _KEY_PATTERN.sub("***REDACTED***", text)
