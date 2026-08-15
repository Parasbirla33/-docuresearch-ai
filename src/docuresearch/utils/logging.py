"""Structured logging setup (structlog), shared across the whole application."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_stdio_encoding() -> None:
    """Make stdout/stderr tolerate non-ASCII text (real source titles, non-English topics/scripts).

    Windows consoles default to a legacy codepage (e.g. cp1252), not UTF-8 -
    printing a title with a curly quote/em dash/non-Latin script then raises
    UnicodeEncodeError and crashes the run *after* research has already
    completed and persisted. `errors="replace"` is a last-resort fallback for
    terminals that can't render UTF-8 at all - `reconfigure` itself already
    fixes the common case. Guarded by `hasattr` since captured streams in
    tests (pytest's `capsys`) don't expose `reconfigure`.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure structlog + stdlib logging once, at process start."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound to `name` (typically `__name__`)."""
    return structlog.get_logger(name)
