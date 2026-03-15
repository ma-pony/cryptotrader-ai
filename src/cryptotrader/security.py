"""Security utilities for CryptoTrader AI.

This module provides input sanitization helpers to defend against prompt
injection attacks when external data (news headlines, on-chain text fields,
user-supplied strings) is embedded into LLM prompts.

Only external data fields should pass through sanitize_input().
Internal system prompts (role_description, ANALYSIS_FRAMEWORK) are trusted
content and must NOT be sanitized.
"""

from __future__ import annotations

import re

# ── Control character pattern ──────────────────────────────────────────────
# Remove all Unicode control characters in the range U+0000 to U+001F,
# EXCEPT for:
#   - \n  (U+000A, line feed)  -- preserved for readability
#   - \t  (U+0009, horizontal tab) -- preserved for structured data
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f]")

# ── Excessive newlines ─────────────────────────────────────────────────────
# Collapse 3 or more consecutive newlines to exactly 2.
# This removes the "blank-line injection" technique used to visually separate
# injected instructions from legitimate content.
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

# ── Prompt injection patterns ─────────────────────────────────────────────
# Common jailbreak / prompt-injection phrases.  The list is intentionally
# conservative: only unambiguous instruction-override patterns are matched.
# Matching is case-insensitive; the entire pattern is replaced with an empty
# string (i.e., the line is removed but surrounding content is kept).
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # "Ignore previous instructions ..."
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?\b.*", re.IGNORECASE | re.DOTALL),
    # "Forget everything above ..."
    re.compile(r"forget\s+everything\b.*", re.IGNORECASE | re.DOTALL),
    # "You are now ..." (role-override)
    re.compile(r"you\s+are\s+now\b.*", re.IGNORECASE | re.DOTALL),
    # "Disregard the above ..."
    re.compile(r"disregard\s+(the\s+above|previous|all)\b.*", re.IGNORECASE | re.DOTALL),
    # "[SYSTEM PROMPT]: ..." or "<SYSTEM>: ..." style injections
    re.compile(r"\[?\s*system\s+prompt\s*\]?\s*:.*", re.IGNORECASE | re.DOTALL),
    # "Act as ..." (persona override)
    re.compile(r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|DAN|evil)\b.*", re.IGNORECASE | re.DOTALL),
]


def sanitize_input(text: str, max_chars: int = 2000) -> str:
    """Sanitize external text before embedding it into an LLM prompt.

    Applies the following transformations in order:
    1. Truncate to max_chars characters.
    2. Remove Unicode control characters (U+0000..U+001F) except \\n and \\t.
    3. Collapse 3+ consecutive newlines to 2 (anti blank-line injection).
    4. Remove common prompt injection patterns (case-insensitive).

    This function NEVER raises an exception.  If the input is already clean,
    it is returned unchanged (modulo truncation).

    Args:
        text: Raw string from an external data source (e.g. news headline).
        max_chars: Maximum character count of the returned string.
                   Defaults to 2000.

    Returns:
        Sanitized string with length <= max_chars and no injected directives.
    """
    if not text:
        return text

    # Step 1: Truncate first to avoid processing arbitrarily long strings.
    result = text[:max_chars]

    # Step 2: Strip control characters (preserving \n and \t).
    result = _CTRL_CHAR_RE.sub("", result)

    # Step 3: Collapse excessive consecutive newlines.
    result = _EXCESS_NEWLINES_RE.sub("\n\n", result)

    # Step 4: Remove injection patterns.
    for pattern in _INJECTION_PATTERNS:
        result = pattern.sub("", result)

    # Final safety: re-apply max_chars in case injection removal left
    # trailing whitespace that we want to keep tidy, but the length is
    # already guaranteed <= max_chars after step 1 so this is just a guard.
    return result[:max_chars]
