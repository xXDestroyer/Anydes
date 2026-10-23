"""Official Groq-first runtime with Parakeet and Qwen3:8B fallback."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


PROGRAM = Path(__file__).resolve().parent
ROOT = PROGRAM.parent
PROFILE = PROGRAM / "configuracion" / "anydes_hybrid.env"
MANIFEST = PROGRAM / "configuracion" / "anydes_hybrid_release_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_release() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    changed: list[str] = []
    for item in payload.get("frozen_dependencies", []):
        path = ROOT / str(item["path"])
        if not path.is_file() or _sha256(path).lower() != str(item["sha256"]).lower():
            changed.append(str(item["path"]))
    if changed:
        raise RuntimeError(
            "La release oficial cambio fuera de su manifiesto: " + ", ".join(changed)
        )


def _load_non_secret_profile() -> None:
    for raw in PROFILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("ANYDES_"):
            raise ValueError(f"Clave no permitida en el perfil oficial: {key}")
        os.environ[key] = value.strip().strip('"').strip("'")


def _load_secret() -> None:
    if os.getenv("GROQ_API_KEY", "").strip():
        return
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "GROQ_API_KEY":
            os.environ["GROQ_API_KEY"] = value.strip().strip('"').strip("'")
            return


def _unload_model(model: str) -> None:
    import urllib.request

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20.0) as response:
            response.read()
    except OSError:
        pass


def _install_runtime_lease() -> None:
    import runtime_session

    @contextmanager
    def hybrid_runtime_session(_mode):
        with runtime_session.RuntimeLease():
            for model in ("qwen3:14b", "qwen3:8b", "moondream"):
                _unload_model(model)
            try:
                yield
            finally:
                for model in ("qwen3:8b", "qwen3:14b", "moondream"):
                    _unload_model(model)

    runtime_session.runtime_session = hybrid_runtime_session


def _install_hybrid_transport() -> None:
    from anydes.hybrid.router import HybridRealtimeTransport
    import anydes.realtime.service as service
    import anydes.realtime.transport as transport

    HybridRealtimeTransport.native_local_transport_class = transport.OllamaRealtimeTransport
    transport.OllamaRealtimeTransport = HybridRealtimeTransport
    service.OllamaRealtimeTransport = HybridRealtimeTransport


def run_official(mode: str) -> int:
    if mode not in {"normal", "performance"}:
        raise ValueError(f"Modo oficial desconocido: {mode}")
    vision_quick_test = os.getenv(
        "ANYDES_OFFICIAL_VISION_QUICK_TEST", "false"
    ).strip().casefold() in {"1", "true", "yes", "on"}
    _verify_release()
    _load_non_secret_profile()
    _load_secret()

    # Config dataclass defaults are evaluated when its module is imported, so
    # the complete official profile must be present before any Anydes module.
    from lanzar_anydes import OFFICIAL_PROFILE, OFFICIAL_VISUAL_PROFILE, load_profile
    load_profile(OFFICIAL_PROFILE)
    if mode == "normal":
        load_profile(OFFICIAL_VISUAL_PROFILE)
    else:
        load_profile(PROGRAM / "configuracion" / "anydes_performance.env")
    _load_non_secret_profile()
    if vision_quick_test:
        # The smoke owns a fresh controlled frame and must not replace it with
        # a host-desktop capture before Config is imported.
        os.environ["ANYDES_SCREEN_FRESH_ON_ANALYZE"] = "false"

    primary = os.getenv(
        "ANYDES_PRIMARY_GROQ_MODEL", "openai/gpt-oss-120b"
    ).strip()
    os.environ["PRIMARY_GROQ_MODEL"] = primary
    os.environ["ANYDES_REALTIME_MODEL"] = f"groq:{primary}|fallback:qwen3:8b"
    os.environ["ANYDES_AI_MODEL"] = "qwen3:8b"
    os.environ["ANYDES_EXPECTED_BRAIN"] = os.environ["ANYDES_REALTIME_MODEL"]
    os.environ["ANYDES_HYBRID_LOG_DIR"] = str(
        PROGRAM / "datos" / "telemetria" / "hybrid_router"
    )
    os.environ["ANYDES_COGNITIVE_V2_LOG_DIR"] = str(
        PROGRAM / "datos" / "telemetria" / "continuity_v2"
    )

    from anydes.voice_input.environment import setup
    setup()
    _install_hybrid_transport()
    _install_runtime_lease()

    from anydes.continuity_v2.runtime_patch import install as install_continuity_v2
    install_continuity_v2()
    from anydes_stabilized_runtime import install as install_stabilized
    install_stabilized()
    from anydes.official_runtime.integration import run_mode
    from anydes.hybrid.router import close_all_routers

    smoke_audio_value = os.getenv("ANYDES_OFFICIAL_SMOKE_AUDIO", "").strip()
    smoke_audio = Path(smoke_audio_value).resolve() if smoke_audio_value else None
    smoke_text = os.getenv("ANYDES_OFFICIAL_SMOKE_TEXT", "").strip()
    auto_close_ms = max(
        0, int(os.getenv("ANYDES_OFFICIAL_SMOKE_AUTO_CLOSE_MS", "0") or 0)
    )
    smoke_delay_ms = max(
        1200, int(os.getenv("ANYDES_OFFICIAL_SMOKE_DELAY_MS", "1200") or 1200)
    )
    try:
        return int(
            run_mode(
                mode,
                target_minutes=0,
                smoke_audio=smoke_audio,
                smoke_text=smoke_text,
                auto_close_ms=auto_close_ms,
                smoke_delay_ms=smoke_delay_ms,
                vision_quick_test=vision_quick_test,
                official_runtime=True,
            )
            or 0
        )
    finally:
        close_all_routers()
        from anydes_stabilized_runtime import record_runtime_return
        record_runtime_return()
