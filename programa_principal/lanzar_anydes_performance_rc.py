from __future__ import annotations

import os

from rc_runtime_cleanup import unload_rc_models


def main() -> None:
    os.environ["ANYDES_RELEASE_CHANNEL"] = "2d_hq_rc_v023"
    exit_code = 0
    try:
        from lanzar_anydes_performance import main as run_performance

        run_performance()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    finally:
        unload_rc_models()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

