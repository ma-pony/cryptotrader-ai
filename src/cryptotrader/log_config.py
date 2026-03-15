"""Logging configuration — call setup_logging() at application startup."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging() -> None:
    """Configure structured logging based on environment.

    Configures both the standard library ``logging`` and ``structlog`` so that:
    - All structlog loggers emit ``timestamp``, ``level``, ``module`` and any
      context-bound fields (e.g. ``trace_id``, ``symbol``, ``node``,
      ``duration_ms``) automatically.
    - ``structlog.contextvars`` is merged into every log record so that
      ``trace_id`` bound via ``set_trace_id()`` propagates to all sub-node
      log entries without explicit passing.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("LOG_FORMAT", "json")

    # ── stdlib logging handler ────────────────────────────────────────────────
    if fmt == "json":
        stdlib_formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        stdlib_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(stdlib_formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ── structlog configuration ───────────────────────────────────────────────
    # Choose the final renderer based on LOG_FORMAT
    final_renderer = structlog.processors.JSONRenderer() if fmt == "json" else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            # Merge context-bound variables (e.g. trace_id set via set_trace_id())
            # into every log event automatically.
            structlog.contextvars.merge_contextvars,
            # Add standard log level field
            structlog.processors.add_log_level,
            # Add ISO-8601 timestamp field
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Add module (callsite) info for the 'module' standard field
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.MODULE,
                ],
            ),
            # Stack info and exception formatting
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            # Final output renderer
            final_renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
