from __future__ import annotations

import os
from pathlib import Path


PROGRAM_DIR = Path(__file__).resolve().parent
OFFICIAL_PROFILE = PROGRAM_DIR / "configuracion" / "anydes_oficial.env"
OFFICIAL_VISUAL_PROFILE = PROGRAM_DIR / "configuracion" / "anydes_normal_25d.env"


def load_profile(path: Path = OFFICIAL_PROFILE) -> dict[str, str]:
    """Load the non-secret official profile before importing Anydes."""
    loaded: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro la configuracion oficial: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key.startswith("ANYDES_"):
            raise ValueError(f"Clave no permitida en el perfil oficial: {key}")
        os.environ[key] = value
        loaded[key] = value
    return loaded


def main() -> None:
    from anydes_hybrid_runtime import run_official

    return run_official("normal")


def _run_normal() -> None:
    configured_profile = os.environ.get("ANYDES_PROFILE_PATH", "").strip()
    profile = Path(configured_profile).resolve() if configured_profile else OFFICIAL_PROFILE
    load_profile(profile)
    if profile == OFFICIAL_PROFILE:
        load_profile(OFFICIAL_VISUAL_PROFILE)
    from anydes.interfaz.pyside_companion.app import main as run_companion

    run_companion()


if __name__ == "__main__":
    raise SystemExit(main())
