"""Structured logging built on structlog.

JSON logs in production (parseable, greppable), pretty console logs when
``LOG_FORMAT=console``. Every request gets a ``request_id`` bound into the log
context so a single user request can be traced across provider calls.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    configure_logging()
    return structlog.get_logger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_context(**kwargs: Any) -> None:
    """Bind values into the async-safe log context for the current request."""
    configure_logging()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    configure_logging()
    structlog.contextvars.clear_contextvars()
