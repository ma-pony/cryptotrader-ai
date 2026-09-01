"""Per-run LLM request provenance, without retaining prompt bodies or credentials."""

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.callbacks import BaseCallbackHandler

_active = ContextVar("backtest_model_evidence", default=None)


def current_model_evidence():
    return _active.get()


def _model_identity(value):
    if not isinstance(value, str) or "://" in value:
        return None
    return value if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", value) else None


class ModelEvidence(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.records = []
        self._calls = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        content = [[{"role": item.type, "content": item.content} for item in batch] for batch in messages]
        digest = hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        requested = _model_identity((kwargs.get("invocation_params") or {}).get("model"))
        record = {
            "requested_model": requested,
            "actual_model": None,
            "actual_model_reason": "response_identity_unavailable",
            "prompt_hash": digest,
            "prompt_version": "messages-sha256-v1",
            "status": "started",
        }
        self._calls[str(run_id)] = record
        self.records.append(record)

    def on_llm_end(self, response, *, run_id, **kwargs):
        record = self._calls.pop(str(run_id), None)
        if record is None:
            return
        metadata = response.llm_output or {}
        actual = _model_identity(metadata.get("model_name") or metadata.get("model"))
        for batch in response.generations or []:
            for generation in batch:
                message = getattr(generation, "message", None)
                details = getattr(message, "response_metadata", {}) or {}
                actual = actual or _model_identity(details.get("model_name") or details.get("model"))
        record.update(
            actual_model=actual,
            actual_model_reason=None if actual else "response_identity_unavailable",
            status="completed",
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        record = self._calls.pop(str(run_id), None)
        if record is not None:
            record["status"] = "failed"


@contextmanager
def capture_model_evidence():
    evidence = ModelEvidence()
    token = _active.set(evidence)
    try:
        yield evidence
    finally:
        _active.reset(token)
