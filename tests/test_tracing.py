"""Test request tracing."""
import structlog
from cryptotrader.tracing import set_trace_id, get_trace_id


def test_set_and_get_trace_id():
    """Test trace ID context management."""
    structlog.contextvars.clear_contextvars()
    trace_id = set_trace_id()
    assert trace_id is not None
    assert get_trace_id() == trace_id


def test_custom_trace_id():
    """Test setting custom trace ID."""
    structlog.contextvars.clear_contextvars()
    custom_id = "test-trace-123"
    result = set_trace_id(custom_id)
    assert result == custom_id
    assert get_trace_id() == custom_id


def test_trace_id_isolation():
    """Test trace IDs are isolated per context."""
    structlog.contextvars.clear_contextvars()
    id1 = set_trace_id()
    assert get_trace_id() == id1

    id2 = set_trace_id()
    assert get_trace_id() == id2
    assert id1 != id2
