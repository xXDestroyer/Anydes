from __future__ import annotations

from pathlib import Path

from lanzar_anydes import OFFICIAL_PROFILE, load_profile


PROGRAM_DIR = Path(__file__).resolve().parent
HQ_PROFILE = PROGRAM_DIR / "configuracion" / "anydes_hq_efficient.env"


def main() -> None:
    load_profile(OFFICIAL_PROFILE)
    load_profile(HQ_PROFILE)
    from anydes.interfaz.pyside_companion.app import main as run_companion

    run_companion()


if __name__ == "__main__":
    main()
