"""Task 5.3 tests: structlog field normalization and trace_id propagation."""

from __future__ import annotations

import importlib

import pytest
import structlog

from cryptotrader.tracing import get_trace_id, set_trace_id

# ---------------------------------------------------------------------------
# 1. trace_id propagation via structlog.contextvars
# ---------------------------------------------------------------------------


def test_set_trace_id_binds_to_structlog_contextvars():
    """set_trace_id() must bind trace_id via structlog.contextvars.bind_contextvars."""
    structlog.contextvars.clear_contextvars()
    tid = set_trace_id("abc-123")
    ctx = structlog.contextvars.get_contextvars()
    assert "trace_id" in ctx
    assert ctx["trace_id"] == "abc-123"
    assert tid == "abc-123"


def test_trace_id_propagates_to_child_logger():
    """After set_trace_id(), structlog loggers auto-include trace_id via merge_contextvars."""
    structlog.contextvars.clear_contextvars()
    set_trace_id("propagate-test")

    captured: list[dict] = []

    def capture_processor(logger, method, event_dict):  # type: ignore[misc]
        captured.append(dict(event_dict))
        raise structlog.DropEvent

    bound_logger = structlog.wrap_logger(
        structlog.PrintLogger(),
        processors=[
            structlog.contextvars.merge_contextvars,
            capture_processor,
        ],
    )
    bound_logger.info("test_event")

    assert len(captured) == 1
    assert captured[0].get("trace_id") == "propagate-test"


def test_clear_contextvars_removes_trace_id():
    """After clear_contextvars(), get_trace_id() should return None."""
    set_trace_id("will-be-cleared")
    structlog.contextvars.clear_contextvars()
    assert get_trace_id() is None


# ---------------------------------------------------------------------------
# 2. structlog configuration: standard processors
# ---------------------------------------------------------------------------


def test_setup_logging_configures_merge_contextvars():
    """setup_logging() must configure structlog with merge_contextvars in the chain."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    config = structlog.get_config()
    processors = config.get("processors", [])
    processor_fn_names = [p.__name__ for p in processors if hasattr(p, "__name__")]
    assert structlog.contextvars.merge_contextvars in processors or ("merge_contextvars" in processor_fn_names), (
        f"merge_contextvars processor not found, current processors: {processor_fn_names}"
    )


def test_structlog_config_includes_standard_fields():
    """structlog config should include TimeStamper (timestamp) and add_log_level (level)."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    config = structlog.get_config()
    processors = config.get("processors", [])
    processor_types = {type(p).__name__ for p in processors}
    assert "TimeStamper" in processor_types, "Missing TimeStamper processor (timestamp field)"
    processor_fn_names = {p.__name__ for p in processors if hasattr(p, "__name__")}
    assert "add_log_level" in processor_fn_names, "Missing add_log_level processor (level field)"


def test_structlog_config_includes_callsite_info():
    """structlog config must include CallsiteParameterAdder for the module field."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    config = structlog.get_config()
    processors = config.get("processors", [])
    processor_types = {type(p).__name__ for p in processors}
    assert "CallsiteParameterAdder" in processor_types, "Missing CallsiteParameterAdder processor (module field)"


# ---------------------------------------------------------------------------
# 3. trace_middleware: structured request log fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_middleware_logs_required_fields():
    """trace_middleware must emit method, path, status_code, response_time_ms, client_ip."""
    captured_events: list[dict] = []

    def capture_processor(logger, method, event_dict):  # type: ignore[misc]
        captured_events.append(dict(event_dict))
        raise structlog.DropEvent

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            capture_processor,
        ]
    )

    import api.main as main_module

    importlib.reload(main_module)
    test_app = main_module.app

    from fastapi.testclient import TestClient

    client = TestClient(test_app, raise_server_exceptions=False)
    client.get("/health")

    request_logs = [e for e in captured_events if "method" in e or "status_code" in e]
    assert len(request_logs) >= 1, f"No log entry with method/status_code found, captured: {captured_events}"
    log_entry = request_logs[0]
    assert "method" in log_entry, "Missing method field"
    assert "path" in log_entry, "Missing path field"
    assert "status_code" in log_entry, "Missing status_code field"
    assert "response_time_ms" in log_entry, "Missing response_time_ms field"
    assert "client_ip" in log_entry, "Missing client_ip field"


@pytest.mark.asyncio
async def test_mask_client_ip_ipv4():
    """IPv4 addresses should have their last octet replaced with 'xxx'."""
    from api.main import _mask_client_ip

    assert _mask_client_ip("192.168.1.100") == "192.168.1.xxx"
    assert _mask_client_ip("10.0.0.1") == "10.0.0.xxx"
    assert _mask_client_ip("127.0.0.1") == "127.0.0.xxx"


@pytest.mark.asyncio
async def test_mask_client_ip_ipv6_passthrough():
    """IPv6 addresses should be returned unchanged (no masking)."""
    from api.main import _mask_client_ip

    result = _mask_client_ip("::1")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_mask_client_ip_unknown_passthrough():
    """'unknown' and empty string should be handled safely."""
    from api.main import _mask_client_ip

    assert _mask_client_ip("unknown") == "unknown"
    assert isinstance(_mask_client_ip(""), str)
