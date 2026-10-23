from __future__ import annotations

from anydes.realtime.cancellation import ResponseSuperseded
from anydes.interfaz.turn_interaction import LatestTurn, restore_input_focus

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from anydes.interfaz.console import build_assistant
from anydes.interfaz.performance_companion.mascot import MiniAnydesCanvas
from anydes.interfaz.pyside_companion.view_model import public_status, state_from_intention
from anydes.nucleo.vram_safety import build_vram_safety_manager, set_active_vram_safety_manager
from anydes.voz.manager_factory import build_voice_manager
from anydes.voz.nueva.lip_sync import (
    stretch_final_visual_lip_sync,
    viseme_change_timestamps,
    VISUAL_LIP_SYNC_FINAL_REGION,
    VISUAL_LIP_SYNC_FINAL_STRETCH,
    visual_alignment_from_audio_clock,
)

try:
    from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QColor, QFont
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QStyle,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    _PYSIDE_IMPORT_ERROR = exc
else:
    _PYSIDE_IMPORT_ERROR = None


CHAT_STYLESHEET = """
QWidget { color: #F8F5FA; font-family: "Segoe UI"; font-size: 13px; }
#ChatRoot {
    background: #17171D;
    border: 1px solid #403A48;
    border-radius: 8px;
}
#DragHeader { background: #1E1D24; border-bottom: 1px solid #36323D; }
#AppName { background: transparent; color: #FFFFFF; font-size: 16px; font-weight: 700; }
#ModeLabel { background: transparent; color: #79D9C7; font-size: 11px; font-weight: 600; }
#StatusLabel {
    background: #222129;
    border: 1px solid #3E3946;
    border-radius: 7px;
    color: #D9D2E0;
    padding: 5px 9px;
    font-size: 11px;
}
QPushButton#WindowButton {
    min-width: 30px; max-width: 30px; min-height: 28px; max-height: 28px;
    background: transparent; border: none; border-radius: 6px;
}
QPushButton#WindowButton:hover { background: #34313A; }
#ChatSurface { background: #17171D; border: none; }
QTextEdit { background: transparent; border: none; color: #F4F0F6; padding: 10px 12px; }
#InputBar { background: #1F1E25; border-top: 1px solid #393541; }
QLineEdit {
    background: #292730; border: 1px solid #46404F; border-radius: 7px;
    color: #FFFFFF; padding: 9px 11px; font-size: 13px;
}
QLineEdit:focus { border-color: #E88BB7; }
QPushButton#SendButton {
    min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px;
    background: #66446C; border: 1px solid #A9688A; border-radius: 7px;
    color: #FFFFFF; font-weight: 700;
}
QPushButton#SendButton:hover { background: #80547F; }
QPushButton#VoiceButton {
    min-width: 38px; max-width: 38px; min-height: 38px; max-height: 38px;
    background: #21443F; border: 1px solid #397B70; border-radius: 7px;
}
QPushButton#VoiceButton:hover { background: #2A5A51; }
"""


OVERLAY_STYLESHEET = """
QWidget { color: #F8F5FA; font-family: "Segoe UI"; }
#CompanionRoot { background: transparent; }
#CompanionControls {
    background: rgba(27, 25, 32, 232);
    border: 1px solid rgba(91, 78, 101, 210);
    border-radius: 8px;
}
QPushButton#CompanionButton {
    min-width: 34px; max-width: 34px; min-height: 31px; max-height: 31px;
    background: transparent; border: none; border-radius: 6px;
}
QPushButton#CompanionButton:hover { background: rgba(92, 74, 98, 205); }
QPushButton#CompanionButton:checked { background: rgba(64, 112, 101, 220); }
#CompanionStatus {
    background: rgba(27, 25, 32, 224);
    border: 1px solid rgba(78, 67, 87, 205);
    border-radius: 7px;
    color: #E6DDEB;
    padding: 4px 8px;
    font-size: 10px;
}
"""


if _PYSIDE_IMPORT_ERROR is None:

    class AssistantWorker(QThread):
        answered = Signal(str)
        partial = Signal(str)
        failed = Signal(str)

        def __init__(self, assistant, text: str) -> None:
            super().__init__()
            self.assistant = assistant
            self.text = text
            self.turn_token = None
            self.turn_id = 0

        def run(self) -> None:
            if self.turn_token is None:
                return self._run_owned()
            try:
                with self.turn_token.bind():
                    self._run_owned()
            except ResponseSuperseded:
                pass

        def _run_owned(self) -> None:
            try:
                self._check_current()
                response = self.assistant.respond_realtime(
                    self.text,
                    input_mode="typed",
                    on_text=self._publish_partial,
                )
                self._check_current()
                self.answered.emit(response)
            except ResponseSuperseded:
                pass
            except Exception as exc:
                self.assistant.audit.record(
                    "performance_interface_response_failure",
                    error_type=type(exc).__name__,
                )
                self.failed.emit(
                    "No conseguí completar esa respuesta, pero tu mensaje sigue aquí."
                )

        def _check_current(self):
            if self.turn_token is not None:
                self.turn_token.check()
            if self.isInterruptionRequested():
                raise ResponseSuperseded()

        def _publish_partial(self, text):
            self._check_current()
            self.partial.emit(text)


    class WarmupWorker(QThread):
        completed = Signal(bool)

        def __init__(self, target) -> None:
            super().__init__()
            self.target = target
            self.error_type = ""
            self.error_message = ""

        def run(self) -> None:
            try:
                warmup = getattr(self.target, "warmup", None)
                if callable(warmup):
                    result = warmup()
                else:
                    result = self.target.warmup_realtime()
                ready = bool(result) if isinstance(result, bool) else True
                self.completed.emit(ready)
            except Exception as exc:
                self.error_type = type(exc).__name__
                self.error_message = str(exc)[:240]
                self.completed.emit(False)


    class VoiceWorker(QThread):
        completed = Signal(str, str, str)

        def __init__(self, manager, intention) -> None:
            super().__init__()
            self.manager = manager
            self.intention = intention
            self.generation = getattr(manager, "generation", None)

        def run(self) -> None:
            try:
                options = {} if self.generation is None else {"generation": self.generation}
                result = self.manager.speak(self.intention, **options)
                self.completed.emit(result.status, result.error or "", result.provider)
            except Exception as exc:
                self.completed.emit("failed", type(exc).__name__, "")


    class DragHeader(QFrame):
        def __init__(self, window: QMainWindow) -> None:
            super().__init__()
            self.window = window
            self._origin: QPoint | None = None
            self.setObjectName("DragHeader")
            self.setCursor(Qt.CursorShape.SizeAllCursor)

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            if self._origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.window.move(event.globalPosition().toPoint() - self._origin)
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802
            self._origin = None
            super().mouseReleaseEvent(event)


    class CompactChat(QTextEdit):
        def __init__(self) -> None:
            super().__init__()
            self.setObjectName("ChatSurface")
            self.setReadOnly(True)
            self.setMinimumHeight(260)
            self._messages: list[tuple[str, str, bool]] = []

        def add_message(self, speaker: str, text: str, *, user: bool = False) -> None:
            self._messages.append((speaker, text, user))
            self._render()

        def update_assistant(self, text: str) -> None:
            if self._messages and not self._messages[-1][2]:
                self._messages[-1] = ("Anydes", text, False)
            else:
                self._messages.append(("Anydes", text, False))
            self._render()

        def _render(self) -> None:
            rows = []
            for speaker, message, user in self._messages[-10:]:
                color = "#F08DB9" if user else "#79D9C7"
                rows.append(
                    f'<span style="color:{color};font-weight:600">{html.escape(speaker)}</span>'
                    f'<span style="color:#F4F0F6">  {html.escape(message)}</span>'
                )
            self.setHtml("<br><br>".join(rows))
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


    class PerformanceChatWindow(QMainWindow):
        def __init__(self, controller: "PerformanceCompanionWindow") -> None:
            super().__init__()
            self.controller = controller
            self.setWindowTitle("Conversación con Anydes")
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.resize(400, 520)
            self.setMinimumSize(350, 390)
            self.setStyleSheet(CHAT_STYLESHEET)
            self._build()

        def _build(self) -> None:
            root = QFrame()
            root.setObjectName("ChatRoot")
            shadow = QGraphicsDropShadowEffect(root)
            shadow.setBlurRadius(28)
            shadow.setOffset(0, 8)
            shadow.setColor(QColor(0, 0, 0, 120))
            root.setGraphicsEffect(shadow)
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(1, 1, 1, 1)
            layout.setSpacing(0)

            header = DragHeader(self)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(12, 8, 8, 8)
            header_layout.setSpacing(8)
            names = QVBoxLayout()
            names.setSpacing(0)
            app_name = QLabel("Anydes")
            app_name.setObjectName("AppName")
            mode = QLabel("PERFORMANCE MODE")
            mode.setObjectName("ModeLabel")
            names.addWidget(app_name)
            names.addWidget(mode)
            header_layout.addLayout(names)
            header_layout.addStretch(1)
            self.status = QLabel("Despertando...")
            self.status.setObjectName("StatusLabel")
            header_layout.addWidget(self.status)
            companion = QPushButton()
            companion.setObjectName("WindowButton")
            companion.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton)
            )
            companion.setToolTip("Ocultar o mostrar Mini Anydes")
            companion.clicked.connect(self.controller.toggle_companion_visibility)
            header_layout.addWidget(companion)
            minimize = QPushButton()
            minimize.setObjectName("WindowButton")
            minimize.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMinButton)
            )
            minimize.setToolTip("Minimizar conversación")
            minimize.clicked.connect(self.showMinimized)
            header_layout.addWidget(minimize)
            hide = QPushButton()
            hide.setObjectName("WindowButton")
            hide.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
            hide.setToolTip("Ocultar conversación")
            hide.clicked.connect(self.hide)
            header_layout.addWidget(hide)
            layout.addWidget(header)

            self.chat = CompactChat()
            layout.addWidget(self.chat, 1)
            input_bar = QFrame()
            input_bar.setObjectName("InputBar")
            input_layout = QHBoxLayout(input_bar)
            input_layout.setContentsMargins(10, 9, 10, 9)
            input_layout.setSpacing(7)
            self.input = QLineEdit()
            self.input.setPlaceholderText("Habla con Anydes...")
            self.input.returnPressed.connect(self._send_from_input)
            input_layout.addWidget(self.input, 1)
            voice = QPushButton()
            voice.setObjectName("VoiceButton")
            voice.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume))
            voice.setToolTip("Escuchar")
            voice.clicked.connect(lambda: self.controller._set_state("listening"))
            input_layout.addWidget(voice)
            send = QPushButton("→")
            send.setObjectName("SendButton")
            send.setToolTip("Enviar")
            send.clicked.connect(self._send_from_input)
            input_layout.addWidget(send)
            self.controls = (self.input, voice, send)
            layout.addWidget(input_bar)

        def _send_from_input(self) -> None:
            text = self.input.text().strip()
            if not text:
                return
            self.input.clear()
            self.controller.send_text(text)
            restore_input_focus(self.input)

        def set_ready(self, ready: bool) -> None:
            for control in self.controls:
                control.setEnabled(ready)
            self.input.setPlaceholderText(
                "Habla con Anydes..." if ready else "Anydes está despertando..."
            )
            if ready:
                restore_input_focus(self.input)

        def closeEvent(self, event) -> None:  # noqa: N802
            event.ignore()
            self.hide()


    class PerformanceCompanionWindow(QMainWindow):
        ready_changed = Signal(bool)

        def __init__(self, *, preview: bool = False, show_chat: bool = False) -> None:
            super().__init__()
            self._preview = bool(preview)
            self._show_chat_on_start = bool(show_chat)
            self._closing = False
            self._drag_offset: QPoint | None = None
            self._positioned = False
            self.setWindowTitle("Anydes Companion")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.resize(230, 330)
            self.setFixedSize(230, 330)

            if self._preview:
                self.assistant = SimpleNamespace()
                self.voice_manager = None
            else:
                self.assistant = build_assistant()
                self.voice_manager = build_voice_manager(
                    self.assistant.config,
                    metrics=self.assistant.metrics,
                )
            self.vram_safety = None
            if not self._preview and self.assistant.config.vram_safety_enabled:
                self.vram_safety = build_vram_safety_manager(
                    self.assistant.config,
                    audit=self.assistant.audit,
                )
                attach = getattr(self.voice_manager, "attach_vram_safety", None)
                if callable(attach):
                    attach(self.vram_safety)
                release = getattr(self.voice_manager, "release_temporary_gpu_cache", None)
                if callable(release):
                    self.vram_safety.register_action(
                        "release_tts_temporary_cache",
                        release,
                        warning=True,
                        critical=True,
                        blocked_by_tasks=("tts",),
                    )
                set_active_vram_safety_manager(self.vram_safety)
                self.vram_safety.start()

            self._worker: AssistantWorker | None = None
            self._turns = LatestTurn()
            self._voice_worker: VoiceWorker | None = None
            self._voice_workers: set[VoiceWorker] = set()
            self._assistant_warmup: WarmupWorker | None = None
            self._voice_warmup: WarmupWorker | None = None
            self._assistant_warmup_done = False
            self._assistant_warmup_ok = False
            self._voice_warmup_done = False
            self._voice_warmup_ok = False
            self._streaming = False
            self._smoke_message = ""
            self._smoke_report: Path | None = None
            self._smoke_dispatched = False
            self._smoke_started = 0.0
            self._smoke_answer = ""
            self._smoke_text_complete_ms = 0.0
            self._smoke_playback_started = 0.0
            self._smoke_audible_end = 0.0
            self._smoke_playback_stopped = 0.0
            self._smoke_visual_finished = 0.0
            self._smoke_visual_stretch: dict[str, object] = {}
            self._voice_result = ("", "", "")
            self._active_playback_id = 0
            self._build_overlay()
            self.chat_window = PerformanceChatWindow(self)

            self._voice_timer = QTimer(self)
            self._voice_timer.timeout.connect(self._poll_voice_events)
            if callable(getattr(self.voice_manager, "poll_events", None)):
                self._voice_timer.start(20)
            self._notification_timer = QTimer(self)
            self._notification_timer.timeout.connect(self._poll_notifications)

            if self._preview:
                self._set_ready(True)
                self._set_state("happy")
                self.chat_window.chat.add_message(
                    "Tú", "Quédate conmigo mientras termino esto.", user=True
                )
                self.chat_window.chat.add_message(
                    "Anydes", "Chi. Me quedo cerquita, sin estorbarte."
                )
            else:
                self._notification_timer.start(250)
                self._set_ready(False)
                self._set_state("thinking")
                self._assistant_warmup = WarmupWorker(self.assistant)
                self._assistant_warmup.completed.connect(self._assistant_ready)
                self._voice_warmup = WarmupWorker(self.voice_manager)
                self._voice_warmup.completed.connect(self._voice_ready)
                self._assistant_warmup.start()
                self._voice_warmup.start()

        @property
        def chat(self) -> CompactChat:
            return self.chat_window.chat

        def _build_overlay(self) -> None:
            self.setStyleSheet(OVERLAY_STYLESHEET)
            root = QWidget()
            root.setObjectName("CompanionRoot")
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(5, 2, 5, 5)
            layout.setSpacing(2)

            self.status_badge = QLabel("Despertando...")
            self.status_badge.setObjectName("CompanionStatus")
            self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.status_badge, 0, Qt.AlignmentFlag.AlignHCenter)

            self.avatar = MiniAnydesCanvas(self)
            self.avatar.drag_started.connect(self._start_drag)
            self.avatar.drag_moved.connect(self._move_drag)
            self.avatar.drag_finished.connect(self._finish_drag)
            self.avatar.chat_requested.connect(self.toggle_chat)
            layout.addWidget(self.avatar, 1)

            controls = QFrame()
            controls.setObjectName("CompanionControls")
            row = QHBoxLayout(controls)
            row.setContentsMargins(5, 3, 5, 3)
            row.setSpacing(2)
            chat_button = QPushButton("···")
            chat_button.setObjectName("CompanionButton")
            chat_button.setToolTip("Abrir o cerrar conversación")
            chat_button.clicked.connect(self.toggle_chat)
            row.addWidget(chat_button)
            self.pin_button = QPushButton("↑")
            self.pin_button.setObjectName("CompanionButton")
            self.pin_button.setCheckable(True)
            self.pin_button.setChecked(True)
            self.pin_button.setToolTip("Mantener siempre visible")
            self.pin_button.clicked.connect(self.toggle_always_on_top)
            row.addWidget(self.pin_button)
            close = QPushButton()
            close.setObjectName("CompanionButton")
            close.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
            close.setToolTip("Cerrar Anydes")
            close.clicked.connect(self.close)
            row.addWidget(close)
            layout.addWidget(controls, 0, Qt.AlignmentFlag.AlignHCenter)

        def showEvent(self, event) -> None:  # noqa: N802
            super().showEvent(event)
            if not self._positioned:
                screen = QApplication.primaryScreen()
                if screen is not None:
                    area = screen.availableGeometry()
                    self.move(area.right() - self.width() - 24, area.bottom() - self.height() - 24)
                self._positioned = True
            if self._show_chat_on_start and not self.chat_window.isVisible():
                QTimer.singleShot(80, self.show_chat)

        def _start_drag(self, global_position: QPoint) -> None:
            self._drag_offset = global_position - self.frameGeometry().topLeft()

        def _move_drag(self, global_position: QPoint) -> None:
            if self._drag_offset is not None:
                self.move(global_position - self._drag_offset)

        def _finish_drag(self) -> None:
            self._drag_offset = None

        def toggle_always_on_top(self, enabled: bool) -> None:
            flags = self.windowFlags()
            if enabled:
                flags |= Qt.WindowType.WindowStaysOnTopHint
            else:
                flags &= ~Qt.WindowType.WindowStaysOnTopHint
            self.setWindowFlags(flags)
            self.show()

        def show_chat(self) -> None:
            if not self.chat_window.isVisible():
                self.chat_window.move(
                    max(10, self.x() - self.chat_window.width() - 12),
                    max(10, self.y() + 12),
                )
                self.chat_window.show()
            self.chat_window.raise_()
            self.chat_window.activateWindow()
            self.chat_window.input.setFocus()

        def toggle_chat(self) -> None:
            if self.chat_window.isVisible():
                self.chat_window.hide()
            else:
                self.show_chat()

        def toggle_companion_visibility(self) -> None:
            if self.isVisible():
                self.hide()
                self.chat_window.raise_()
                self.chat_window.activateWindow()
            else:
                self.show()
                self.raise_()

        def _assistant_ready(self, ready: bool) -> None:
            self._assistant_warmup_done = True
            self._assistant_warmup_ok = bool(ready)
            error_type = (
                getattr(self._assistant_warmup, "error_type", "")
                if self._assistant_warmup is not None
                else ""
            )
            self.assistant.audit.record(
                "performance_realtime_warmup_completed",
                ready=bool(ready),
                error_type=error_type,
            )
            self._set_ready(True)
            if not ready:
                self._set_state("curious")
                self.ready_changed.emit(False)
            elif not self._voice_warmup_done:
                self._set_status("Preparando voz")
            elif self._voice_warmup_ok:
                self._set_state("idle")
            self._maybe_start_smoke()

        def _voice_ready(self, ready: bool) -> None:
            self._voice_warmup_done = True
            self._voice_warmup_ok = bool(ready)
            error_type = (
                getattr(self._voice_warmup, "error_type", "")
                if self._voice_warmup is not None
                else ""
            )
            self.assistant.audit.record(
                "voice_warmup_completed",
                ready=bool(ready),
                error_type=error_type,
            )
            self._set_ready(True)
            self._set_state("idle" if ready else "curious")
            self.ready_changed.emit(ready)
            self._maybe_start_smoke()

        def _maybe_start_smoke(self) -> None:
            if (
                not self._smoke_message
                or self._smoke_dispatched
                or not self._assistant_warmup_done
                or not self._voice_warmup_done
            ):
                return
            self._smoke_dispatched = True
            self._smoke_started = time.perf_counter()
            QTimer.singleShot(0, lambda: self.send_text(self._smoke_message))

        def configure_smoke(self, message: str, report_path: Path) -> None:
            self._smoke_message = str(message).strip()
            self._smoke_report = report_path.resolve()
            self._smoke_started = 0.0
            self._smoke_dispatched = False
            self._maybe_start_smoke()

        def _set_ready(self, ready: bool) -> None:
            self.chat_window.set_ready(ready)

        def send_text(self, text: str) -> None:
            if self._closing or not text.strip():
                return
            if self._preview:
                self.chat.add_message("Tú", text, user=True)
                self.chat.add_message("Anydes", "Esta es una vista previa visual.")
                self._set_state("happy")
                return
            self.voice_manager.cancel()
            self.avatar.stop_speaking()
            self._active_playback_id = 0
            self._voice_worker = None
            self._turns.submit(text)
            if self._worker is not None:
                self._worker.requestInterruption()
            self.chat.add_message("Tú", text, user=True)
            self._streaming = False
            self._set_state("thinking")
            restore_input_focus(self.chat_window.input)
            self._start_pending_turn()

        def _start_pending_turn(self):
            if self._closing or self._worker is not None:
                return
            pending = self._turns.take()
            if pending is None:
                return
            revision, text = pending
            worker = AssistantWorker(self.assistant, text)
            worker.turn_token = self._turns.token
            worker.turn_id = revision
            self._worker = worker
            worker.partial.connect(lambda text, r=revision: self._handle_partial(text) if self._turns.current(r) else None)
            worker.answered.connect(lambda text, r=revision: self._handle_answer(text) if self._turns.current(r) else None)
            worker.failed.connect(lambda text, r=revision: self._handle_failure(text) if self._turns.current(r) else None)
            worker.finished.connect(lambda w=worker: self._response_worker_finished(w))
            worker.start()

        def _response_worker_finished(self, worker):
            if self._worker is worker:
                self._worker = None
            worker.deleteLater()
            self._start_pending_turn()

        def _handle_partial(self, text: str) -> None:
            self.chat.update_assistant(text)
            self._streaming = True

        def _handle_answer(self, text: str) -> None:
            if self._streaming:
                self.chat.update_assistant(text)
            else:
                self.chat.add_message("Anydes", text)
            self._streaming = False
            self._smoke_answer = text
            if self._smoke_started:
                self._smoke_text_complete_ms = (time.perf_counter() - self._smoke_started) * 1000.0
            self._set_ready(True)
            intention = self.assistant.last_intention
            state = state_from_intention(intention)
            expression_name = str(getattr(state, "expression", "idle") or "idle")
            self._set_state(expression_name if expression_name in COMPANION_STATES else "idle")
            self._set_status("Hablando")
            if intention is None or not intention.speech_intention.text.strip():
                self._set_state("idle")
                self._finish_smoke("text_only")
                return
            self._voice_worker = VoiceWorker(self.voice_manager, intention.speech_intention)
            self._voice_workers.add(self._voice_worker)
            self._voice_worker.completed.connect(lambda status, error, provider, w=self._voice_worker: self._voice_completed(status, error, provider) if w is self._voice_worker else None)
            self._voice_worker.finished.connect(lambda w=self._voice_worker: self._release_voice_worker(w))
            self._voice_worker.start()

        def _release_voice_worker(self, worker):
            self._voice_workers.discard(worker)
            if worker is self._voice_worker:
                self._voice_worker = None
            worker.deleteLater()

        def _handle_failure(self, text: str) -> None:
            self.chat.update_assistant(text)
            self._streaming = False
            self._set_ready(True)
            self._set_state("curious")
            self._smoke_answer = text
            self._finish_smoke("response_failed")

        def _voice_completed(self, status: str, error: str, provider: str) -> None:
            self._voice_result = (status, error, provider)
            self.assistant.audit.record(
                "interface_voice_completed",
                status=status,
                error_type=error,
                provider=provider,
            )
            if status == "failed":
                self.avatar.stop_speaking()
                self._set_state("curious")
                self.assistant.audit.record(
                    "performance_voice_failure",
                    error_type=error,
                    provider=provider,
                )
                self._finish_smoke(status, provider=provider, error=error)

        def _finish_smoke(self, status: str, **details: object) -> None:
            if self._smoke_report is None:
                return
            backend_trace = getattr(self.voice_manager.backend, "last_trace", None)
            payload = {
                "status": status,
                "model": self.assistant.config.realtime_model,
                "voice_engine": self.assistant.config.new_voice_engine,
                "voice_reference": str(self.assistant.config.new_voice_reference_audio),
                "message": self._smoke_message,
                "answer": self._smoke_answer,
                "text_complete_ms": round(self._smoke_text_complete_ms, 3),
                "total_ms": round((time.perf_counter() - self._smoke_started) * 1000.0, 3),
                "lip_sync_clock": {
                    "playback_started": round(self._smoke_playback_started, 6),
                    "audible_end": round(self._smoke_audible_end, 6),
                    "playback_stopped": round(self._smoke_playback_stopped, 6),
                    "stop_after_audible_end_ms": round(
                        max(0.0, self._smoke_playback_stopped - self._smoke_audible_end) * 1000.0,
                        3,
                    ),
                    "visual_finished": round(self._smoke_visual_finished, 6),
                },
                "visual_stretch": self._smoke_visual_stretch,
                "voice_trace": (
                    {
                        "status": backend_trace.status,
                        "provider": getattr(backend_trace, "provider", ""),
                        "synthesis_ms": round(backend_trace.synthesis_ms, 3),
                        "playback_ms": round(backend_trace.playback_ms, 3),
                        "sample_rate": backend_trace.sample_rate,
                        "channels": backend_trace.channels,
                        "frame_count": backend_trace.frame_count,
                    }
                    if backend_trace is not None
                    else None
                ),
                **details,
            }
            self._smoke_report.parent.mkdir(parents=True, exist_ok=True)
            self._smoke_report.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._smoke_report = None
            QTimer.singleShot(200, self.close)

        def _poll_voice_events(self) -> None:
            poll = getattr(self.voice_manager, "poll_events", None)
            if not callable(poll):
                return
            for event in poll():
                if event.kind == "playback_started":
                    playback_id = int(event.payload.get("playback_id") or 0)
                    self._active_playback_id = playback_id
                    clock = float(event.payload.get("audio_clock") or event.created_monotonic)
                    if not self._smoke_playback_started:
                        self._smoke_playback_started = clock
                    self._smoke_audible_end = max(
                        self._smoke_audible_end,
                        float(
                            event.payload.get("audible_end_clock")
                            or clock + float(event.payload.get("duration_ms") or 0.0) / 1000.0
                        ),
                    )
                    offset, start_delay = visual_alignment_from_audio_clock(clock)
                    timeline = event.payload.get("timeline")
                    raw_timeline = timeline if isinstance(timeline, list) else []
                    duration_ms = float(event.payload.get("duration_ms") or 0.0)
                    if event.payload.get("timeline_source") == "measured_pcm":
                        visual_timeline = raw_timeline
                        visual_duration_ms = float(event.payload.get("active_end_ms", duration_ms))
                    else:
                        visual_timeline, visual_duration_ms = stretch_final_visual_lip_sync(raw_timeline, duration_ms)
                    source_changes = viseme_change_timestamps(raw_timeline)
                    stretched_changes = viseme_change_timestamps(visual_timeline)
                    if len(source_changes) > 1:
                        self._smoke_visual_stretch = {
                            "original_duration_ms": round(duration_ms, 3),
                            "visual_duration_ms": round(visual_duration_ms, 3),
                            "final_region_ratio": VISUAL_LIP_SYNC_FINAL_REGION,
                            "final_region_stretch": VISUAL_LIP_SYNC_FINAL_STRETCH,
                            "cue_count_unchanged": len(visual_timeline) == len(raw_timeline),
                            "post_roll_cues": 0,
                            "source_last_changes": source_changes,
                            "stretched_last_changes": stretched_changes,
                        }
                    self.avatar.start_lip_sync(
                        visual_timeline,
                        visual_duration_ms,
                        offset,
                        start_delay,
                    )
                elif event.kind in {"playback_stopped", "playback_interrupted", "voice_failed"}:
                    playback_id = int(event.payload.get("playback_id") or 0)
                    if event.kind != "voice_failed" and (
                        not self._active_playback_id
                        or playback_id != self._active_playback_id
                    ):
                        continue
                    self._smoke_playback_stopped = event.created_monotonic
                    status, error, provider = self._voice_result
                    if event.kind == "playback_interrupted":
                        status = "interrupted"
                    elif event.kind == "voice_failed":
                        status = "failed"
                    QTimer.singleShot(
                        0,
                        lambda pid=playback_id, kind=event.kind, final_status=status,
                        final_provider=provider, final_error=error, gen=getattr(self.voice_manager, "generation", None): self._finish_visual_playback(
                            pid,
                            kind,
                            final_status or "spoken",
                            final_provider,
                            final_error,
                            gen,
                        ),
                    )

        def _finish_visual_playback(
            self,
            playback_id: int,
            event_kind: str,
            status: str,
            provider: str,
            error: str,
            generation: int | None = None,
        ) -> None:
            if generation is not None and generation != getattr(self.voice_manager, "generation", None):
                return
            if event_kind != "voice_failed" and playback_id != self._active_playback_id:
                return
            self.avatar.stop_speaking()
            self._smoke_visual_finished = time.perf_counter()
            self._active_playback_id = 0
            self._set_state("curious" if event_kind == "voice_failed" else "idle")
            self._finish_smoke(status, provider=provider, error=error)

        def _poll_notifications(self) -> None:
            if self._preview or (self._worker is not None and self._worker.isRunning()):
                return
            for text in self.assistant.drain_realtime_notifications():
                self.chat.add_message("Anydes", text)
                self._set_state("happy")

        def _set_status(self, text: str) -> None:
            self.status_badge.setText(text)
            self.chat_window.status.setText(text)

        def _set_state(self, name: str) -> None:
            normalized = str(name).strip().casefold()
            if normalized not in COMPANION_STATES:
                normalized = "idle"
            self._set_status(status_text(normalized))
            self.avatar.set_state(normalized)

        def closeEvent(self, event) -> None:  # noqa: N802
            if self._closing:
                super().closeEvent(event)
                return
            self._closing = True
            self._turns.close()
            if self._worker is not None:
                self._worker.requestInterruption()
            self.chat_window.hide()
            if self._preview:
                super().closeEvent(event)
                QTimer.singleShot(0, QApplication.instance().quit)
                return
            self._trace_shutdown("started")
            self.voice_manager.cancel()
            self.avatar.stop_speaking()
            close_voice = getattr(self.voice_manager, "close", None)
            if callable(close_voice):
                close_voice()
            self._trace_shutdown("voice_closed")
            for worker in tuple(self._voice_workers):
                if worker.isRunning():
                    worker.wait(5000)
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
            if self._assistant_warmup is not None and self._assistant_warmup.isRunning():
                self._assistant_warmup.wait(5000)
            if self._voice_warmup is not None and self._voice_warmup.isRunning():
                self._voice_warmup.wait(5000)
            if self.vram_safety is not None:
                self.vram_safety.stop()
                set_active_vram_safety_manager(None)
            self.assistant.close()
            self._trace_shutdown("assistant_closed")
            super().closeEvent(event)
            self._trace_shutdown("completed")
            application = QApplication.instance()
            if application is not None:
                QTimer.singleShot(0, application.quit)

        def _trace_shutdown(self, phase: str, **details: object) -> None:
            path = Path(self.assistant.config.data_dir) / "telemetria" / "performance_shutdown.jsonl"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "phase": phase,
                    "monotonic": round(time.perf_counter(), 6),
                    **details,
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            except OSError:
                return


COMPANION_STATES = {
    "speaking",
    "listening",
    "thinking",
    "happy",
    "curious",
    "surprised",
    "shy",
    "sleepy",
    "calm",
    "idle",
}


def companion_pose(state: str) -> tuple[str, str, str]:
    normalized = str(state).strip().casefold()
    poses = {
        "speaking": ("neutral", "speaking", "user"),
        "listening": ("listening", "listening", "user"),
        "thinking": ("thinking", "thinking", "down"),
        "happy": ("happy", "idle", "user"),
        "curious": ("curious", "thinking", "user"),
        "surprised": ("surprised", "idle", "user"),
        "shy": ("shy", "idle", "down"),
        "sleepy": ("sleepy", "idle", "down"),
        "calm": ("neutral", "idle", "user"),
        "idle": ("neutral", "idle", "user"),
    }
    return poses.get(normalized, poses["idle"])


def status_text(state: str) -> str:
    labels = {
        "speaking": "Hablando",
        "listening": "Escuchando",
        "thinking": "Pensando",
        "happy": "Contenta",
        "curious": "Curiosa",
        "surprised": "Sorprendida",
        "shy": "Tímida",
        "sleepy": "Descansando",
        "calm": "Tranquila",
        "idle": "Contigo",
    }
    return labels.get(str(state).strip().casefold(), "Contigo")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-state", choices=sorted(COMPANION_STATES), default="")
    parser.add_argument(
        "--preview-mouth",
        choices=("closed", "small", "medium", "smile", "round"),
        default="",
    )
    parser.add_argument("--preview-blink", action="store_true")
    parser.add_argument("--show-chat", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--chat-screenshot", type=Path)
    parser.add_argument("--auto-close-ms", type=int, default=0)
    parser.add_argument("--smoke-message", default="")
    parser.add_argument("--smoke-report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if _PYSIDE_IMPORT_ERROR is not None:
        raise RuntimeError("Performance Mode requiere PySide6.") from _PYSIDE_IMPORT_ERROR
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = PerformanceCompanionWindow(preview=args.preview, show_chat=args.show_chat)
    if args.preview_state:
        window._set_state(args.preview_state)
    if args.preview_mouth:
        preview_viseme = {
            "closed": "neutral",
            "small": "ih",
            "medium": "aa",
            "smile": "ee",
            "round": "oh",
        }[args.preview_mouth]
        window.avatar.start_lip_sync(
            [{"start_ms": 0.0, "end_ms": 60000.0, "viseme": preview_viseme}],
            60000.0,
        )
    if args.preview_blink:
        window.avatar.set_force_blink(True)
    if args.smoke_message:
        report = args.smoke_report or (
            Path.cwd() / "evaluation-results" / "mode-selector-2026-08-13" / "performance_smoke.json"
        )
        window.configure_smoke(args.smoke_message, report)
    window.show()
    if args.screenshot:
        output = args.screenshot.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def save_and_close() -> None:
            window.grab().save(str(output), "PNG")
            if args.chat_screenshot:
                chat_output = args.chat_screenshot.resolve()
                chat_output.parent.mkdir(parents=True, exist_ok=True)
                window.show_chat()
                QApplication.processEvents()
                window.chat_window.grab().save(str(chat_output), "PNG")
            output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "preview": args.preview,
                        "companion_state": window.avatar.state,
                        "companion_width": window.width(),
                        "companion_height": window.height(),
                        "chat_separate_window": window.chat_window is not window,
                        "uses_vrm_renderer": False,
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            window.close()

        QTimer.singleShot(1100, save_and_close)
    else:
        configured_auto_close = int(os.environ.get("ANYDES_RUNTIME_AUTO_CLOSE_MS", "0") or 0)
        auto_close_ms = args.auto_close_ms or configured_auto_close
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, window.close)
    if args.smoke_message:
        QTimer.singleShot(120000, window.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
