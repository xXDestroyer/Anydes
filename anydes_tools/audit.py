from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


_SECRET_MARKERS = ("secret", "token", "password", "api_key", "credential")
_CONTENT_MARKERS = ("content", "text", "code", "data")
_SAFE_METADATA_KEYS = {
    "content_kind",
    "source_context_type",
    "content_validator_result",
    "content_validator_reason",
    "error_code",
    "reason_code",
}


class ToolAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, event: str, **details: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "details": _sanitize(details),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def _sanitize(value: Any, key: str = "") -> Any:
    normalized = key.casefold()
    if any(marker in normalized for marker in _SECRET_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item, key) for item in value]
    if (
        isinstance(value, str)
        and normalized not in _SAFE_METADATA_KEYS
        and any(marker in normalized for marker in _CONTENT_MARKERS)
    ):
        return f"<redacted:length={len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
