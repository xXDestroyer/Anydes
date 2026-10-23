from __future__ import annotations

import json
import urllib.error
import urllib.request

VISION_NAME_HINTS = (
    "llava",
    "moondream",
    "vision",
    "bakllava",
    "minicpm-v",
    "gemma3",
    "qwen2.5vl",
    "qwen3-vl",
)

DEFAULT_VISION_PULL_ORDER = (
    "moondream",
    "llava:7b",
    "llava",
    "llama3.2-vision",
)


def list_ollama_models(base_url: str, timeout: float = 8.0) -> list[dict]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []
    models = payload.get("models", [])
    return models if isinstance(models, list) else []


def model_names(models: list[dict]) -> list[str]:
    names: list[str] = []
    for item in models:
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def is_vision_model(model_info: dict) -> bool:
    capabilities = model_info.get("capabilities")
    if isinstance(capabilities, list) and "vision" in capabilities:
        return True
    name = str(model_info.get("name") or model_info.get("model") or "").lower()
    return any(hint in name for hint in VISION_NAME_HINTS)


def find_installed_vision_model(models: list[dict], preferred: str = "") -> str | None:
    if preferred:
        preferred_lower = preferred.lower()
        for item in models:
            name = str(item.get("name") or item.get("model") or "")
            if name.lower() == preferred_lower or name.lower().startswith(f"{preferred_lower}:"):
                return name.split(":")[0] if ":" in name else name
        for item in models:
            name = str(item.get("name") or item.get("model") or "")
            if preferred_lower in name.lower():
                return name

    for item in models:
        if is_vision_model(item):
            return str(item.get("name") or item.get("model") or "").split(":")[0]

    return None


def pull_ollama_model(base_url: str, model_name: str, timeout: float = 900.0) -> tuple[bool, str]:
    payload = json.dumps({"name": model_name, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, detail or str(exc)
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)

    if not raw:
        return True, model_name

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return True, model_name

    status = str(data.get("status", "")).lower()
    if status in {"success", "complete", "completed"}:
        return True, model_name
    if data.get("error"):
        return False, str(data["error"])
    return True, model_name


def resolve_ollama_vision_model(
    base_url: str,
    preferred: str,
    *,
    auto_pull: bool,
) -> tuple[str | None, str | None]:
    models = list_ollama_models(base_url)
    if not models:
        return None, (
            "No pude conectar con Ollama. Abre la app Ollama y vuelve a intentar."
        )

    installed = find_installed_vision_model(models, preferred)
    if installed:
        return installed, None

    if not auto_pull:
        installed_names = ", ".join(model_names(models)[:6]) or "ninguno"
        target = preferred or DEFAULT_VISION_PULL_ORDER[0]
        return None, (
            "No tienes un modelo con vision instalado en Ollama. "
            f"Modelos actuales: {installed_names}. "
            f"Ejecuta `ollama pull {target}` o activa ANYDES_VISION_AUTO_PULL=true."
        )

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    for candidate in DEFAULT_VISION_PULL_ORDER:
        if candidate not in candidates:
            candidates.append(candidate)

    errors: list[str] = []
    for candidate in candidates:
        ok, detail = pull_ollama_model(base_url, candidate)
        if ok:
            models = list_ollama_models(base_url)
            installed = find_installed_vision_model(models, candidate)
            if installed:
                return installed, None
            return candidate, None
        errors.append(f"{candidate}: {detail}")

    return None, (
        "Intenté descargar un modelo de vision automaticamente pero no funcionó. "
        f"Prueba manualmente: `ollama pull {DEFAULT_VISION_PULL_ORDER[0]}`."
    )
