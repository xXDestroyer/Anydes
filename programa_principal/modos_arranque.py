from __future__ import annotations

import subprocess
import sys
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PROGRAM_DIR.parent


@dataclass(frozen=True)
class StartupMode:
    key: str
    name: str
    eyebrow: str
    description: str
    features: tuple[str, ...]
    entrypoint: Path
    accent: str
    recommended: bool = False


MODES: tuple[StartupMode, ...] = (
    StartupMode(
        key="normal",
        name="Normal Mode",
        eyebrow="Experiencia completa",
        description=(
            "La Anydes de siempre, con su mejor calidad conversacional y su "
            "habitacion completa."
        ),
        features=(
            "Qwen3:14B",
            "Avatar y habitacion completos",
            "Mejor calidad conversacional",
            "Recomendado para conversar con calma",
        ),
        entrypoint=PROGRAM_DIR / "lanzar_anydes.py",
        accent="#F08DB9",
        recommended=True,
    ),
    StartupMode(
        key="performance",
        name="Performance Mode",
        eyebrow="Companion ligero",
        description=(
            "Una presencia compacta para jugar o trabajar con mas margen de "
            "recursos."
        ),
        features=(
            "Qwen3:8B",
            "Mini avatar flotante",
            "Misma memoria, SELF y voz B4",
            "Menor consumo de VRAM",
        ),
        entrypoint=PROGRAM_DIR / "lanzar_anydes_performance.py",
        accent="#73D6C3",
    ),
)


def mode_by_key(key: str) -> StartupMode:
    selected = str(key).strip().casefold()
    for mode in MODES:
        if mode.key == selected:
            return mode
    raise KeyError(f"Modo de Anydes desconocido: {key}")


def runtime_command(mode: StartupMode, *, python_executable: str | None = None) -> list[str]:
    executable = python_executable or sys.executable
    if not mode.entrypoint.is_file():
        raise FileNotFoundError(f"No se encontro el runtime de {mode.name}: {mode.entrypoint}")
    return [executable, str(mode.entrypoint)]


def start_runtime(
    mode: StartupMode,
    *,
    python_executable: str | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    prepare: Callable[[StartupMode], tuple[str, ...]] | None = None,
) -> subprocess.Popen:
    """Start exactly one runtime after the user makes a selection."""

    # The child owns the exclusive runtime lease before touching resident models.
    if prepare is not None:
        prepare(mode)
    command = runtime_command(mode, python_executable=python_executable)
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return popen(
        command,
        cwd=str(PROJECT_DIR),
        close_fds=True,
        creationflags=creation_flags,
    )


def prepare_exclusive_brain(
    mode: StartupMode,
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 3.0,
) -> tuple[str, ...]:
    """Start the Groq-first runtime without a resident local Qwen."""

    _ = mode
    candidates = {"qwen3:14b", "qwen3:8b"}
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/api/ps")
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return ()
        raise OSError("No pude verificar los modelos activos; el nuevo modo no se cargara.") from exc
    except (OSError, ValueError) as exc:
        raise OSError("No pude verificar los modelos activos; el nuevo modo no se cargara.") from exc
    running = {
        str(item.get("name") or item.get("model") or "").split(":latest", 1)[0]
        for item in payload.get("models", ())
        if isinstance(item, dict)
    }
    unloaded: list[str] = []
    for model in sorted(candidates & running):
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
        except (OSError, urllib.error.URLError) as exc:
            raise OSError(f"No pude descargar {model}; el nuevo cerebro no se cargara.") from exc
        unloaded.append(model)
    if unloaded:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=timeout_seconds) as response:
                    remaining = json.loads(response.read().decode("utf-8")).get("models", ())
            except (OSError, ValueError) as exc:
                raise OSError("No pude confirmar la descarga del cerebro anterior.") from exc
            names = {str(item.get("name") or item.get("model") or "").split(":latest", 1)[0] for item in remaining}
            if not candidates.intersection(names):
                break
            if time.monotonic() >= deadline:
                raise OSError("El cerebro anterior sigue residente; no se iniciara otro modelo.")
            time.sleep(.1)
    return tuple(unloaded)


def heavy_modules_loaded(module_names: Sequence[str]) -> tuple[str, ...]:
    prefixes = ("torch", "ollama", "anydes.cerebro", "anydes.nucleo.assistant")
    return tuple(sorted(name for name in module_names if name.startswith(prefixes)))
