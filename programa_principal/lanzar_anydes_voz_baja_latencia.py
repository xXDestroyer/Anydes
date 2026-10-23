from __future__ import annotations

import os

from lanzar_anydes import load_profile


def main() -> None:
    load_profile()
    os.environ["ANYDES_NEW_VOICE_DELIVERY_MODE"] = "prosodic"
    os.environ["ANYDES_NEW_VOICE_SEGMENTATION_STRATEGY"] = "fixed15"
    os.environ["ANYDES_NEW_VOICE_COMMON_PHRASE_CACHE_ENABLED"] = "true"
    from anydes.interfaz.pyside_companion.app import main as run_companion

    run_companion()


if __name__ == "__main__":
    main()
