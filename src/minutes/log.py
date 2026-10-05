import json
import logging
import sys
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from minutes.config import get_settings

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def bind_correlation_id(cid: str) -> Token[str]:
    return _correlation_id.set(cid)


def get_correlation_id() -> str:
    return _correlation_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        line = {
            "ts": ts.replace("+00:00", "Z"),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event": getattr(record, "event", ""),
            "correlation_id": get_correlation_id(),
            "message": record.getMessage(),
            "fields": getattr(record, "fields", {}),
        }
        return json.dumps(line)


class ComponentAdapter(logging.LoggerAdapter[logging.Logger]):
    # the base class replaces the caller's extra with its own; event and fields arrive there
    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        kwargs["extra"] = {**(self.extra or {}), **(kwargs.get("extra") or {})}
        return msg, kwargs


def configure_logging(level: str) -> None:
    root = logging.getLogger("minutes")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_file = get_settings().log_file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(component: str) -> logging.LoggerAdapter[logging.Logger]:
    """Logger whose calls take extra={"event": ..., "fields": {...}}."""
    return ComponentAdapter(logging.getLogger(f"minutes.{component}"), {"component": component})
