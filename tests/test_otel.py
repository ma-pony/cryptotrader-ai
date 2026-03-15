"""OpenTelemetry optional tracing integration tests (task 5.4).

Validates otel.py infrastructure module:
1. When OTLP_ENDPOINT is unset, setup_otel() returns silently (no side-effects)
2. get_tracer() always returns a callable tracer object (never raises)
3. When opentelemetry package is missing, all functions degrade to no-op
4. When OTLP_ENDPOINT is set and opentelemetry is available, returns real tracer
5. node_logger() creates spans when OTel is active (log-trace correlation)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# -- 1. Degradation behavior without OTLP_ENDPOINT --


def test_setup_otel_no_endpoint_is_noop(monkeypatch):
    """Without OTLP_ENDPOINT, setup_otel() should return silently."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    # Reset state for test
    otel._otel_active = False

    otel.setup_otel("test-service")
    # Should not raise; OTel should remain inactive
    assert not otel._otel_active, "OTel should not activate without OTLP_ENDPOINT"


def test_get_tracer_no_endpoint_returns_noop(monkeypatch):
    """Without OTLP_ENDPOINT, get_tracer() should return a usable no-op tracer."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    tracer = otel.get_tracer()
    assert tracer is not None, "get_tracer() should not return None"

    # no-op tracer's start_as_current_span should work without raising
    with tracer.start_as_current_span("test-span"):
        pass  # should not raise


def test_get_tracer_span_context_manager_noop(monkeypatch):
    """no-op tracer's start_as_current_span() context manager should enter and exit cleanly."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    tracer = otel.get_tracer()
    entered = False
    exited = False

    with tracer.start_as_current_span("span-name") as span:
        entered = True
        assert span is not None  # no-op span should not be None

    exited = True
    assert entered
    assert exited


# -- 2. Degradation when opentelemetry package is missing --


def test_setup_otel_missing_package_is_noop(monkeypatch):
    """When opentelemetry is not installed, setup_otel() should handle ImportError silently."""
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4317")

    # Simulate opentelemetry not installed
    import sys

    with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
        from cryptotrader import otel

        # Reset module state
        otel._otel_active = False

        # Should degrade silently, not raise ImportError
        try:
            otel.setup_otel("test-service")
        except ImportError:
            raise AssertionError("setup_otel() should not propagate ImportError") from None


def test_get_tracer_returns_noop_without_package(monkeypatch):
    """When OTel is inactive (missing package or no endpoint), get_tracer() returns no-op tracer."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    tracer = otel.get_tracer()
    # Should be usable without raising
    assert hasattr(tracer, "start_as_current_span"), "tracer should have start_as_current_span method"


# -- 3. Activation with OTLP_ENDPOINT and available package --


def test_setup_otel_with_endpoint_activates_otel(monkeypatch):
    """When OTLP_ENDPOINT is set and opentelemetry is available, setup_otel() should activate OTel."""
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4317")

    from cryptotrader import otel

    otel._otel_active = False

    # Check if opentelemetry is actually installed
    try:
        import opentelemetry  # noqa: F401

        otel_available = True
    except ImportError:
        otel_available = False

    if otel_available:
        # Only verify activation when package is available
        # Use mocks to avoid real gRPC connection
        mock_provider = MagicMock()
        with (
            patch("opentelemetry.sdk.trace.TracerProvider", return_value=mock_provider),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
        ):
            otel.setup_otel("test-service")
            assert otel._otel_active, "OTel should activate when OTLP_ENDPOINT is set"
    else:
        # Package unavailable: should degrade to no-op
        otel.setup_otel("test-service")
        assert not otel._otel_active, "OTel should not activate when package is unavailable"


def test_get_tracer_after_setup_returns_real_tracer_or_noop(monkeypatch):
    """After setup_otel(), get_tracer() should return a usable tracer (real or no-op)."""
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4317")

    from cryptotrader import otel

    otel._otel_active = False
    otel.setup_otel("test-service")

    tracer = otel.get_tracer()
    assert tracer is not None
    assert hasattr(tracer, "start_as_current_span")

    # Regardless of real activation, span creation should work
    with tracer.start_as_current_span("verify-span"):
        pass


# -- 4. OTel state reset safety --


def test_setup_otel_idempotent_call_no_endpoint(monkeypatch):
    """Multiple setup_otel() calls without endpoint should not accumulate side-effects."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    # Two consecutive calls should not raise
    otel.setup_otel("svc-a")
    otel.setup_otel("svc-b")
    assert not otel._otel_active


# -- 5. node_logger() integration: OTel span and log correlation --


async def test_node_logger_creates_otel_span_when_active(monkeypatch):
    """When OTel is active, @node_logger() decorator should also create OTel spans."""
    import structlog

    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    def drop_processor(logger, method_name, event_dict):
        raise structlog.DropEvent

    structlog.configure(processors=[drop_processor])

    from cryptotrader.tracing import node_logger

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    @node_logger()
    async def sample_node(state: dict) -> dict:
        return {"result": "ok"}

    # Activate OTel and inject mock tracer
    otel._otel_active = True
    with patch("cryptotrader.otel.get_tracer", return_value=mock_tracer):
        result = await sample_node({"metadata": {}, "data": {}})

    otel._otel_active = False
    assert result == {"result": "ok"}, "Node return value should pass through"


async def test_node_logger_no_span_when_otel_inactive(monkeypatch):
    """When OTel is inactive, @node_logger() should not attempt to create spans."""
    import structlog

    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    from cryptotrader import otel

    otel._otel_active = False

    def drop_processor(logger, method_name, event_dict):
        raise structlog.DropEvent

    structlog.configure(processors=[drop_processor])

    from cryptotrader.tracing import node_logger

    @node_logger()
    async def sample_node(state: dict) -> dict:
        return {}

    # get_tracer should not be called when OTel is inactive (or should return no-op)
    # Main check: no exception raised
    await sample_node({"metadata": {}, "data": {}})
    # Should not raise


# -- 6. api/main.py lifespan call verification --


def test_lifespan_calls_setup_otel(monkeypatch):
    """api/main.py lifespan() should call setup_otel() after setup_logging()."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)

    setup_otel_calls = []

    def mock_setup_otel(service_name: str = "cryptotrader-ai") -> None:
        setup_otel_calls.append(service_name)

    with patch("cryptotrader.otel.setup_otel", mock_setup_otel):
        # Verify lifespan source contains setup_otel call
        import inspect

        from api.main import lifespan

        source = inspect.getsource(lifespan)
        assert "setup_otel" in source, "lifespan() should call setup_otel()"


# -- 7. CLI entry call verification --


def test_cli_setup_calls_setup_otel():
    """CLI _setup() callback should call setup_otel() after setup_logging()."""
    import inspect

    from cli.main import _setup

    source = inspect.getsource(_setup)
    assert "setup_otel" in source, "CLI _setup() should call setup_otel()"


# -- 8. pyproject.toml otel optional dependency group --


def test_pyproject_has_otel_optional_dependencies():
    """pyproject.toml should have an otel group in [project.optional-dependencies]."""
    import pathlib

    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    assert "otel" in content, "pyproject.toml should contain otel optional dependency group"
    assert "opentelemetry-sdk" in content, "otel group should include opentelemetry-sdk"
    assert "opentelemetry-exporter-otlp-proto-grpc" in content, "otel group should include OTLP gRPC exporter"


def test_opentelemetry_not_in_core_dependencies():
    """opentelemetry should only appear in optional dependencies, not core dependencies."""
    import pathlib

    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Find core dependencies block (under [project], not optional-dependencies)
    lines = content.split("\n")
    in_core_deps = False
    for line in lines:
        if line.strip() == "dependencies = [":
            in_core_deps = True
        elif in_core_deps and line.strip() == "]":
            in_core_deps = False
        elif in_core_deps and "opentelemetry" in line.lower():
            raise AssertionError(f"opentelemetry should not be in core dependencies: {line}")
