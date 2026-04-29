"""Tests for Retry-After aware wait strategy in LLM factory."""

from __future__ import annotations

from unittest.mock import MagicMock

from cryptotrader.llm.factory import _RetryAfterWait


class TestRetryAfterWait:
    def _make_retry_state(self, exc: Exception | None = None) -> MagicMock:
        state = MagicMock()
        state.attempt_number = 1
        if exc is not None:
            state.outcome.exception.return_value = exc
        else:
            state.outcome = None
        return state

    def test_uses_retry_after_header(self):
        exc = Exception("rate limited")
        exc.headers = {"retry-after": "7"}  # type: ignore[attr-defined]
        base_wait = MagicMock(return_value=2.0)
        wait = _RetryAfterWait(base_wait)

        delay = wait(self._make_retry_state(exc))
        assert delay == 7.0
        base_wait.assert_not_called()

    def test_caps_retry_after_at_60(self):
        exc = Exception("rate limited")
        exc.headers = {"retry-after": "120"}  # type: ignore[attr-defined]
        wait = _RetryAfterWait(MagicMock(return_value=2.0))

        delay = wait(self._make_retry_state(exc))
        assert delay == 60.0

    def test_falls_back_to_base_without_header(self):
        exc = ValueError("no headers")
        base_wait = MagicMock(return_value=3.0)
        wait = _RetryAfterWait(base_wait)

        delay = wait(self._make_retry_state(exc))
        assert delay == 3.0

    def test_falls_back_when_no_exception(self):
        base_wait = MagicMock(return_value=1.0)
        wait = _RetryAfterWait(base_wait)

        delay = wait(self._make_retry_state(None))
        assert delay == 1.0
