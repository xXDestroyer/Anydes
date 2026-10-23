from __future__ import annotations

import os
from pathlib import Path


ENV_KEYS = (
    "ANYDES_AI_PROVIDER",
    "ANYDES_AI_MODEL",
    "ANYDES_STT_PROVIDER",
    "ANYDES_TTS_PROVIDER",
    "ANYDES_VOICE_ACTIVATION",
    "ANYDES_WAKE_WORD",
    "ANYDES_VOICE_SECONDS",
    "ANYDES_PROACTIVE_ENABLED",
    "ANYDES_PROACTIVE_INTERVAL_SECONDS",
    "ANYDES_VISION_ENABLED",
    "ANYDES_VISION_PROVIDER",
    "ANYDES_VISION_REACTIVITY_LEVEL",
    "ANYDES_SCREEN_LIVE_MODE",
)


def build_runtime_env(values: dict[str, str]) -> dict[str, str]:
    """Build a child-process environment without importing a GUI toolkit."""
    runtime = os.environ.copy()
    runtime.update(values)
    return runtime


def write_env_file(path: Path, values: dict[str, str]) -> None:
    """Update known configuration keys while preserving unrelated lines."""
    existing: list[str] = []
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in values:
                existing.append(f"{key}={values[key]}")
                seen.add(key)
            else:
                existing.append(line)

    for key in ENV_KEYS:
        if key in values and key not in seen:
            existing.append(f"{key}={values[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")
