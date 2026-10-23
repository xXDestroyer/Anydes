from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from modos_arranque import StartupMode, heavy_modules_loaded, mode_by_key, start_runtime

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QVBoxLayout,
        QWidget,
    )

    from lanzar_selector_modo import ModeCard, SELECTOR_STYLESHEET
except ImportError as exc:  # pragma: no cover - depende del entorno local.
    _PYSIDE_IMPORT_ERROR = exc
else:
    _PYSIDE_IMPORT_ERROR = None


PROGRAM_DIR = Path(__file__).resolve().parent
RC_NORMAL = StartupMode(
    key="normal",
    name="Normal Mode",
    eyebrow="Release Candidate 2.5D HQ",
    description=(
        "La experiencia completa de Anydes con su avatar 2.5D HQ v023 "
        "y la pasada final de composicion/movimiento v024, "
        "integrado en la habitacion oficial."
    ),
    features=(
        "Qwen3:14B Q4_K_M",
        "Anydes 2.5D HQ v023 + composicion v024 + habitacion",
        "Voz canonica B4, memoria y SELF",
        "Build reversible; no reemplaza produccion",
    ),
    entrypoint=PROGRAM_DIR / "lanzar_anydes_25d_rc.py",
    accent="#F08DB9",
    recommended=True,
)
RC_PERFORMANCE = replace(
    mode_by_key("performance"),
    entrypoint=PROGRAM_DIR / "lanzar_anydes_performance_rc.py",
)
RC_MODES = (RC_NORMAL, RC_PERFORMANCE)

RC_STYLESHEET = SELECTOR_STYLESHEET + """
#RcBadge {
    background: #3D2938;
    border: 1px solid #A95E85;
    border-radius: 7px;
    color: #F4B0CD;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 10px;
}
#RcNotice {
    background: transparent;
    color: #9F97AA;
    font-size: 12px;
}
"""


if _PYSIDE_IMPORT_ERROR is None:

    class ReleaseCandidateSelector(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Anydes 2.5D - Release Candidate")
            self.resize(850, 610)
            self.setMinimumSize(780, 570)
            self.setStyleSheet(RC_STYLESHEET)
            self._launching = False
            self._runtime = None
            self._build()

        def _build(self) -> None:
            root = QWidget()
            root.setObjectName("SelectorRoot")
            self.setCentralWidget(root)
            scene = QVBoxLayout(root)
            scene.setContentsMargins(42, 30, 42, 24)
            scene.setSpacing(0)

            badge_row = QHBoxLayout()
            badge_row.addStretch(1)
            badge = QLabel("STAGING BUILD · v024")
            badge.setObjectName("RcBadge")
            badge_row.addWidget(badge)
            badge_row.addStretch(1)
            scene.addLayout(badge_row)
            scene.addSpacing(14)

            title = QLabel("Probar Anydes 2.5D")
            title.setObjectName("PageTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            subtitle = QLabel(
                "Evalua la candidata visual completa sin modificar la version oficial."
            )
            subtitle.setObjectName("PageSubtitle")
            subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            subtitle.setWordWrap(True)
            scene.addWidget(title)
            scene.addWidget(subtitle)
            scene.addSpacing(24)

            cards = QHBoxLayout()
            cards.setSpacing(18)
            for mode in RC_MODES:
                cards.addWidget(ModeCard(mode, self.start_mode), 1)
            scene.addLayout(cards, 1)

            self.status = QLabel("Nada pesado se carga hasta que elijas un modo")
            self.status.setObjectName("FooterStatus")
            self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            scene.addWidget(self.status)
            notice = QLabel("Abrir Anydes.bat y la experiencia oficial permanecen intactos.")
            notice.setObjectName("RcNotice")
            notice.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            scene.addWidget(notice)

        def start_mode(self, mode: StartupMode) -> None:
            if self._launching:
                return
            self._launching = True
            self.status.setText(f"Abriendo {mode.name}...")
            QApplication.processEvents()
            try:
                self._runtime = start_runtime(mode)
            except (OSError, ValueError) as exc:
                self._launching = False
                self.status.setText("No pude abrir el modo elegido")
                QMessageBox.critical(self, "No pude abrir Anydes RC", str(exc))
                return
            QTimer.singleShot(180, self.close)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--auto-close-ms", type=int, default=0)
    parser.add_argument("--verify-lightweight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if _PYSIDE_IMPORT_ERROR is not None:
        raise RuntimeError("El selector RC requiere PySide6.") from _PYSIDE_IMPORT_ERROR
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.verify_lightweight:
        loaded = heavy_modules_loaded(tuple(sys.modules))
        if loaded:
            raise RuntimeError(f"El selector RC cargo modulos pesados: {', '.join(loaded)}")
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setFont(QFont("Segoe UI", 10))
    window = ReleaseCandidateSelector()
    window.show()
    if args.screenshot:
        output = args.screenshot.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def save_and_close() -> None:
            window.grab().save(str(output), "PNG")
            window.close()

        QTimer.singleShot(700, save_and_close)
    else:
        auto_close_ms = args.auto_close_ms or int(
            os.environ.get("ANYDES_SELECTOR_AUTO_CLOSE_MS", "0") or 0
        )
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, window.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
