from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from modos_arranque import MODES, StartupMode, heavy_modules_loaded, start_runtime

try:
    from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStyle,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depende del entorno local.
    _PYSIDE_IMPORT_ERROR = exc
else:
    _PYSIDE_IMPORT_ERROR = None


SELECTOR_STYLESHEET = """
QWidget {
    background: #111116;
    color: #F7F4FA;
    font-family: "Segoe UI";
    font-size: 14px;
}
#SelectorRoot { background: #111116; }
#PageTitle {
    background: transparent;
    color: #FFFFFF;
    font-size: 31px;
    font-weight: 700;
}
#PageSubtitle {
    background: transparent;
    color: #BEB7C7;
    font-size: 14px;
}
#ModeCard {
    background: #1A191F;
    border: 1px solid #36323D;
    border-radius: 8px;
}
#ModeCard[recommended="true"] { border: 1px solid #A95E85; }
#ModeEyebrow {
    background: transparent;
    color: #BDB5C8;
    font-size: 12px;
    font-weight: 600;
}
#ModeName {
    background: transparent;
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 700;
}
#ModeDescription, #ModeDetails {
    background: transparent;
    color: #D4CEDB;
    font-size: 13px;
}
#RecommendedLabel {
    background: #3D2938;
    border: 1px solid #74445E;
    border-radius: 7px;
    color: #F4B0CD;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 8px;
}
QPushButton#StartButton {
    min-height: 43px;
    border-radius: 7px;
    border: 1px solid #5E5368;
    background: #302C37;
    color: #FFFFFF;
    font-weight: 700;
    padding: 0 16px;
}
QPushButton#StartButton:hover { background: #443D4D; border-color: #F08DB9; }
QPushButton#StartButton[performance="true"] { background: #20443F; border-color: #397D70; }
QPushButton#StartButton[performance="true"]:hover { background: #2B5A52; border-color: #73D6C3; }
QPushButton#DetailsButton {
    min-width: 34px;
    max-width: 34px;
    min-height: 34px;
    max-height: 34px;
    border-radius: 7px;
    border: 1px solid #45404D;
    background: #24222A;
}
QPushButton#DetailsButton:hover { background: #34313B; border-color: #9F94AB; }
#FooterStatus { background: transparent; color: #938B9E; font-size: 12px; }
"""


if _PYSIDE_IMPORT_ERROR is None:

    class ModeCard(QFrame):
        def __init__(self, mode: StartupMode, on_start, parent=None) -> None:
            super().__init__(parent)
            self.mode = mode
            self.setObjectName("ModeCard")
            self.setProperty("recommended", mode.recommended)
            self.setMinimumWidth(290)
            self.setMinimumHeight(300)
            self._details_open = False

            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 10)
            shadow.setColor(QColor(0, 0, 0, 90))
            self.setGraphicsEffect(shadow)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(22, 20, 22, 20)
            layout.setSpacing(10)

            top = QHBoxLayout()
            eyebrow = QLabel(mode.eyebrow.upper())
            eyebrow.setObjectName("ModeEyebrow")
            top.addWidget(eyebrow)
            top.addStretch(1)
            if mode.recommended:
                recommended = QLabel("RECOMENDADO")
                recommended.setObjectName("RecommendedLabel")
                top.addWidget(recommended)
            layout.addLayout(top)

            name = QLabel(mode.name)
            name.setObjectName("ModeName")
            layout.addWidget(name)

            description = QLabel(mode.description)
            description.setObjectName("ModeDescription")
            description.setWordWrap(True)
            description.setMinimumHeight(58)
            layout.addWidget(description)

            self.details = QLabel("\n".join(f"•  {item}" for item in mode.features))
            self.details.setObjectName("ModeDetails")
            self.details.setWordWrap(True)
            self.details.setVisible(False)
            layout.addWidget(self.details)
            layout.addStretch(1)

            controls = QHBoxLayout()
            details_button = QPushButton()
            details_button.setObjectName("DetailsButton")
            details_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
            )
            details_button.setToolTip(f"Ver detalles de {mode.name}")
            details_button.clicked.connect(self.toggle_details)
            controls.addWidget(details_button)
            start = QPushButton("Iniciar")
            start.setObjectName("StartButton")
            start.setProperty("performance", mode.key == "performance")
            start.clicked.connect(lambda: on_start(mode))
            controls.addWidget(start, 1)
            layout.addLayout(controls)

        def toggle_details(self) -> None:
            self._details_open = not self._details_open
            self.details.setVisible(self._details_open)
            target = 382 if self._details_open else 300
            animation = QPropertyAnimation(self, b"minimumHeight", self)
            animation.setDuration(180)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.setStartValue(self.minimumHeight())
            animation.setEndValue(target)
            animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


    class ModeSelectorWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Anydes")
            self.resize(850, 570)
            self.setMinimumSize(780, 530)
            self.setStyleSheet(SELECTOR_STYLESHEET)
            self._launching = False
            self._build()

        def _build(self) -> None:
            root = QWidget()
            root.setObjectName("SelectorRoot")
            self.setCentralWidget(root)
            scene = QVBoxLayout(root)
            scene.setContentsMargins(42, 36, 42, 28)
            scene.setSpacing(0)

            title = QLabel("Elegir modo de Anydes")
            title.setObjectName("PageTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            subtitle = QLabel("Elige la experiencia que mejor acompaña lo que vas a hacer.")
            subtitle.setObjectName("PageSubtitle")
            subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            subtitle.setWordWrap(True)
            scene.addWidget(title)
            scene.addWidget(subtitle)
            scene.addSpacing(28)

            cards = QHBoxLayout()
            cards.setSpacing(18)
            for mode in MODES:
                cards.addWidget(ModeCard(mode, self.start_mode), 1)
            scene.addLayout(cards, 1)

            self.status = QLabel("Nada se cargará hasta que elijas")
            self.status.setObjectName("FooterStatus")
            self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            scene.addWidget(self.status)

        def start_mode(self, mode: StartupMode) -> None:
            if self._launching:
                return
            self._launching = True
            self.status.setText(f"Abriendo {mode.name}...")
            QApplication.processEvents()
            try:
                start_runtime(mode)
            except (OSError, ValueError) as exc:
                self._launching = False
                self.status.setText("No pude abrir el modo elegido")
                QMessageBox.critical(self, "No pude abrir Anydes", str(exc))
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
        raise RuntimeError("El selector de Anydes requiere PySide6.") from _PYSIDE_IMPORT_ERROR
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.verify_lightweight:
        loaded = heavy_modules_loaded(tuple(sys.modules))
        if loaded:
            raise RuntimeError(f"El selector cargó módulos pesados: {', '.join(loaded)}")
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setFont(QFont("Segoe UI", 10))
    window = ModeSelectorWindow()
    window.show()
    if args.screenshot:
        output = args.screenshot.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def save_and_close() -> None:
            window.grab().save(str(output), "PNG")
            window.close()

        QTimer.singleShot(650, save_and_close)
    else:
        auto_mode = os.environ.get("ANYDES_SELECTOR_AUTO_MODE", "").strip().casefold()
        if auto_mode:
            try:
                selected_mode = next(mode for mode in MODES if mode.key == auto_mode)
            except StopIteration:
                selected_mode = None
            if selected_mode is not None:
                QTimer.singleShot(250, lambda: window.start_mode(selected_mode))

        configured_auto_close = int(os.environ.get("ANYDES_SELECTOR_AUTO_CLOSE_MS", "0") or 0)
        auto_close_ms = args.auto_close_ms or configured_auto_close
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, window.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
