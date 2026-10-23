from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable


DEFAULT_RC_MODELS = ("qwen3:14b", "qwen3:8b", "moondream")


def unload_rc_models(
    models: Iterable[str] = DEFAULT_RC_MODELS,
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 5.0,
) -> tuple[str, ...]:
    """Unload only models owned by the RC after its UI has closed."""

    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/ps", timeout=timeout_seconds
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return ()

    requested = {str(model).strip().split(":latest", 1)[0] for model in models}
    resident = {
        str(item.get("name") or item.get("model") or "").split(":latest", 1)[0]
        for item in payload.get("models", ())
        if isinstance(item, dict)
    }
    unloaded: list[str] = []
    for model in sorted(requested & resident):
        body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response.read()
        except (OSError, urllib.error.URLError):
            continue
        unloaded.append(model)
    return tuple(unloaded)

