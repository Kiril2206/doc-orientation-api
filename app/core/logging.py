"""Structured logging configuration."""

import logging
import sys
import uuid
from contextvars import ContextVar

# Context variable for request-scoped tracing
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    """Inject request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")  # type: ignore[attr-defined]
        return True


def generate_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:12]


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
