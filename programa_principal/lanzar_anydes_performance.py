from __future__ import annotations

from pathlib import Path

from lanzar_anydes import OFFICIAL_PROFILE, load_profile


PROGRAM_DIR = Path(__file__).resolve().parent
PERFORMANCE_PROFILE = PROGRAM_DIR / "configuracion" / "anydes_performance.env"


def main() -> None:
    from anydes_hybrid_runtime import run_official

    return run_official("performance")


def _run_performance() -> None:
    load_profile(OFFICIAL_PROFILE)
    load_profile(PERFORMANCE_PROFILE)
    from anydes.interfaz.performance_companion.app import main as run_performance

    run_performance()


if __name__ == "__main__":
    raise SystemExit(main())
