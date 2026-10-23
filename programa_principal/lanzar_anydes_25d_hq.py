from __future__ import annotations

import os

from lanzar_anydes import load_profile


def main() -> None:
    load_profile()
    os.environ["ANYDES_AVATAR_PRESENTATION"] = "2d_hq_experimental"
    os.environ["ANYDES_STARTUP_MODE"] = "normal"
    from anydes.interfaz.pyside_companion.app import main as run_companion

    run_companion()


if __name__ == "__main__":
    main()
