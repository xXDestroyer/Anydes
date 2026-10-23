from __future__ import annotations

import os
import sys
import json
from pathlib import Path

from lanzar_anydes import load_profile
from rc_runtime_cleanup import unload_rc_models


def main() -> None:
    from runtime_session import run_desktop
    return run_desktop("normal", _run_rc)


def _run_rc() -> None:
    load_profile()
    os.environ["ANYDES_AVATAR_PRESENTATION"] = "2d_hq_experimental"
    os.environ["ANYDES_STARTUP_MODE"] = "normal"
    os.environ["ANYDES_RELEASE_CHANNEL"] = "2d_hq_rc_v024"
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--enable-webgl --ignore-gpu-blocklist --no-sandbox",
    )
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from anydes.interfaz.pyside_companion.app import AnydesPySideWindow

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = AnydesPySideWindow()
    window.setWindowTitle("Anydes 2.5D HQ v024 - Release Candidate")
    window.show()

    screenshot = os.environ.get("ANYDES_RC_SCREENSHOT_PATH", "").strip()
    output: Path | None = None
    if screenshot:
        output = Path(screenshot).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        screenshot_deadline_ms = int(
            os.environ.get("ANYDES_RC_SCREENSHOT_DEADLINE_MS", "25000") or 25000
        )
        screenshot_started = 0

        def capture_composited_window() -> bool:
            return window.avatar.grab().save(str(output), "PNG")

        def save_when_avatar_is_ready() -> None:
            nonlocal screenshot_started
            screenshot_started += 250
            avatar_view = getattr(getattr(window, "avatar", None), "_vrm_view", None)
            if bool(getattr(avatar_view, "render_ready", False)) or screenshot_started >= screenshot_deadline_ms:
                QTimer.singleShot(900, capture_composited_window)
                return
            QTimer.singleShot(250, save_when_avatar_is_ready)

        QTimer.singleShot(250, save_when_avatar_is_ready)

    auto_close_ms = int(os.environ.get("ANYDES_RUNTIME_AUTO_CLOSE_MS", "0") or 0)
    if auto_close_ms > 0:
        QTimer.singleShot(auto_close_ms, window.close)
    exit_code = app.exec()
    if output is not None:
        if not output.is_file():
            window.grab().save(str(output), "PNG")
        avatar_view = getattr(getattr(window, "avatar", None), "_vrm_view", None)
        diagnostics = {
            "release_channel": os.environ["ANYDES_RELEASE_CHANNEL"],
            "render_ready": bool(getattr(avatar_view, "render_ready", False)),
            "render_error": str(getattr(avatar_view, "render_error", "")),
            "renderer": getattr(avatar_view, "diagnostic_state", {}),
            "screenshot": str(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(diagnostics, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    unload_rc_models()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
