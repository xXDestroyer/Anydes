"""Opt-in observation only; never dispatches or repairs a tool plan."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe(value, key="", depth=0):
    if re.search(r"secret|credential|password|api.?key|authorization|headers|^env$", key, re.I):
        return "REDACTED"
    if depth > 8:
        return "DEPTH_LIMIT"
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: safe(getattr(value, f.name), f.name, depth + 1) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): safe(v, str(k), depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (tuple, list)):
        return [safe(v, key, depth + 1) for v in value[:100]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if re.search(r"\bgsk_[A-Za-z0-9]+|Bearer\s+\S+", value, re.I):
            return "REDACTED"
        if re.search(r"content|text|code$|stdout|stderr|message|summary|prompt|context|goal|payload|source_request|user_request", key, re.I):
            if key in {"error_code", "failure_code", "reason_code", "content_kind", "source_context_type"}:
                return value[:240]
            return {"characters": len(value), "sha256": digest(value.encode("utf-8"))}
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


class Observer:
    def __init__(self, destination: Path):
        self.destination = destination
        self.turn_id = "startup"
        self.calls = {}
        self.lock = threading.RLock()
        self.old_profile = None
        self.old_thread_profile = None
        self.errors = 0

    def emit(self, event, **data):
        with self.lock:
            with self.destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, "turn_id": self.turn_id,
                    "monotonic": time.monotonic(), **safe(data)}, ensure_ascii=True) + "\n")

    def profile(self, frame, event, result):
        try:
            path = frame.f_code.co_filename.replace("\\", "/")
            name = frame.f_code.co_name
            if path.endswith("/anydes_tools/audit.py") and name == "record" and event == "call":
                self.emit("runtime_audit", runtime_event=frame.f_locals.get("event"),
                    details=frame.f_locals.get("details", {}))
                return
            selected = (
                "/anydes_tools/conversation/" in path
                or path.endswith(("/anydes_tools/executor.py", "/anydes_tools/composition.py"))
                or "/anydes/continuity_v2/groq_client.py" in path
                or ("/anydes/hybrid/router.py" in path and name == "stream")
            )
            if not selected or name.startswith("<"):
                return
            identity = id(frame)
            if event == "call":
                self.calls[identity] = time.perf_counter()
            elif event == "return":
                started = self.calls.pop(identity, None)
                if started is None:
                    return
                args = {k: v for k, v in frame.f_locals.items() if k in {
                    "request", "plan", "action", "specification", "test", "decision",
                    "capability", "confirmed_effects", "verification", "failure",
                    "verification_rows", "raw_spec", "error_code", "stage", "stdout"}}
                self.emit("stage_return", module=frame.f_globals.get("__name__"),
                    function=name, owner_type=type(frame.f_locals.get("self")).__name__,
                    duration_ms=(time.perf_counter()-started)*1000,
                    arguments=args, result=result)
        except Exception:
            self.errors += 1

    def __enter__(self):
        self.old_profile = sys.getprofile()
        self.old_thread_profile = threading.getprofile()
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)
        return self

    def __exit__(self, *_):
        sys.setprofile(self.old_profile)
        threading.setprofile(self.old_thread_profile)
