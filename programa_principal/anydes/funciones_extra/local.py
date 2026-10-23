from __future__ import annotations

import datetime
import json
import os
import platform
from zoneinfo import ZoneInfo


def current_time(_: str = "") -> dict[str, object]:
    timezone_name = os.getenv("ANYDES_WEATHER_TIMEZONE", "America/Guayaquil")
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception:
        timezone = datetime.datetime.now().astimezone().tzinfo
    now = datetime.datetime.now(timezone)
    return {
        "current_time": now.isoformat(timespec="seconds"),
        "timezone": timezone_name,
        "date": now.date().isoformat(),
        "hour": now.hour,
        "minute": now.minute,
    }


def system_info(_: str = "") -> dict[str, object]:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def word_count(text: str) -> dict[str, object]:
    lines = text.splitlines()
    words = text.split()
    return {
        "text_preview": text[:160],
        "characters": len(text),
        "lines": len(lines),
        "words": len(words),
    }


def format_json(text: str) -> dict[str, object]:
    parsed = json.loads(text)
    return {
        "parsed": parsed,
        "formatted_json": json.dumps(parsed, indent=2, sort_keys=True),
    }


def run_command(_: str = "") -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "system_command_execution_disabled",
    }
