"""OpenTelemetry optional tracing infrastructure module.

Only activates when env var OTLP_ENDPOINT is non-empty AND opentelemetry packages
are installed. Otherwise all APIs degrade to no-op silently.

Usage::

    from cryptotrader.otel import setup_otel, get_tracer

    setup_otel("cryptotrader-ai")   # call once at lifespan / CLI entry

    tracer = get_tracer()
    with tracer.start_as_current_span("my-span") as span:
        ...  # works regardless of whether OTel is active
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# -- internal state --

_otel_active: bool = False
_tracer = None  # real tracer or None (inactive)


# -- No-op implementation --


class _NoOpSpan:
    """Placeholder span that does nothing, used when OTel is inactive."""

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def set_status(self, status: object) -> None:
        pass


class _NoOpTracer:
    """Placeholder tracer that does nothing, used when OTel is inactive.

    start_as_current_span() returns a side-effect-free context manager
    so callers can use the same code regardless of OTel activation state.
    """

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: object):  # type: ignore[override]
        """Return a no-op span context manager."""
        yield _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()


# -- public API --


def setup_otel(service_name: str = "cryptotrader-ai") -> None:
    """Initialize OpenTelemetry SDK.

    Activates only when both conditions are met:
    1. Env var ``OTLP_ENDPOINT`` is non-empty
    2. ``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp-proto-grpc`` are installed

    When either condition is unmet, returns silently without raising.

    Args:
        service_name: OTLP resource service name, defaults to ``"cryptotrader-ai"``.
    """
    global _otel_active, _tracer

    import os

    endpoint = os.environ.get("OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.debug("OTLP_ENDPOINT not set, OTel running in no-op mode")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "opentelemetry packages not installed, OTel running in no-op mode. "
            "To enable tracing: uv pip install 'cryptotrader-ai[otel]'",
        )
        return

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _tracer = provider.get_tracer(service_name)
        _otel_active = True
        logger.info("OpenTelemetry activated", extra={"otlp_endpoint": endpoint, "service": service_name})
    except Exception:
        logger.warning(
            "OpenTelemetry initialization failed, degrading to no-op mode",
            exc_info=True,
        )
        _otel_active = False
        _tracer = None


def get_tracer():
    """Return the current process OTel Tracer.

    - If OTel is active (``setup_otel()`` succeeded), returns a real ``opentelemetry.trace.Tracer``.
    - Otherwise returns :class:`_NoOpTracer` whose ``start_as_current_span()`` is a no-op.

    Callers need not check whether OTel is active:

    .. code-block:: python

        with get_tracer().start_as_current_span("node.my_node") as span:
            ...  # always safe
    """
    if _otel_active and _tracer is not None:
        return _tracer
    return _NOOP_TRACER


def is_active() -> bool:
    """Return whether OTel has been successfully activated. Used by tracing.py."""
    return _otel_active
