from __future__ import annotations

from anydes.realtime.cancellation import ResponseSuperseded
from anydes.interfaz.turn_interaction import LatestTurn, restore_input_focus

import html
import math
import os
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from anydes.interfaz.pyside_companion.avatar_render_profile import PROFILE, avatar_asset_candidates
from anydes.interfaz.pyside_companion.design_tokens import LAYOUT, PALETTE, companion_stylesheet
from anydes.interfaz.pyside_companion.room_environment import room_background_path, synced_environment
from anydes.interfaz.pyside_companion.view_model import (
    CompanionState,
    camera_mode_for_state,
    inner_world_cards,
    public_status,
    state_from_intention,
)
from anydes.interfaz.pyside_companion.avatar_render_profile import preferred_avatar_model
from anydes.interfaz.pyside_companion.vrm_renderer import (
    VrmAvatarView,
    WEBENGINE_AVAILABLE,
    renderer_assets_available,
)
from anydes.interfaz.pyside_companion.avatar25d_renderer import (
    Experimental25DAvatarView,
    WEBENGINE_25D_AVAILABLE,
)
from anydes.interfaz.console import build_assistant
from anydes.nucleo.consent import is_external_provider, session_consents
from anydes.nucleo.sensors import sensor_hub
from anydes.nucleo.vram_safety import (
    build_vram_safety_manager,
    set_active_vram_safety_manager,
)
from anydes.vision.state import get_screen_state
from anydes.vision.session import (
    VisionSessionController,
    vision_permission_requested,
)
from anydes.vision.residency import (
    build_vision_residency_manager,
    set_active_vision_residency_manager,
)
from anydes.vision.tools import stop_all_screen_watching
from anydes.voz.manager_factory import build_voice_manager
from anydes.voz.nueva.lip_sync import (
    stretch_final_visual_lip_sync,
    visual_alignment_from_audio_clock,
)

try:
    from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRect, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QImageReader, QLinearGradient, QPainter, QPainterPath, QPixmap, QRadialGradient
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFrame,
        QGraphicsBlurEffect,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depende del entorno local del usuario.
    _PYSIDE_IMPORT_ERROR = exc
else:
    _PYSIDE_IMPORT_ERROR = None


if _PYSIDE_IMPORT_ERROR is None:

    class AssistantWorker(QThread):
        answered = Signal(str)
        partial = Signal(str)
        failed = Signal(str)

        def __init__(self, assistant, text: str):
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
                if self.assistant.config.resolved_conversation_mode() == "realtime":
                    response = self.assistant.respond_realtime(
                        self.text,
                        input_mode="typed",
                        on_text=self._publish_partial,
                    )
                else:
                    response = self.assistant.respond(
                        self.text,
                        interactive=True,
                        input_mode="typed",
                    )
                self._check_current()
                self.answered.emit(response)
            except ResponseSuperseded:
                pass
            except Exception as exc:
                error_type = type(exc).__name__
                self.assistant.audit.record(
                    "interface_response_failure",
                    error_type=error_type,
                    failure_layer=_interface_failure_layer(error_type),
                )
                self.failed.emit(
                    _interface_failure_text(
                        error_type,
                        stabilized=self.assistant.config.realtime_audio_stabilization_enabled,
                    )
                )

        def _check_current(self):
            if self.turn_token is not None:
                self.turn_token.check()
            if self.isInterruptionRequested():
                raise ResponseSuperseded()

        def _publish_partial(self, text):
            self._check_current()
            self.partial.emit(text)


    class VoiceWorker(QThread):
        completed = Signal(str, str, str)

        def __init__(self, voice_manager, speech_intention):
            super().__init__()
            self.voice_manager = voice_manager
            self.speech_intention = speech_intention
            self.generation = getattr(voice_manager, "generation", None)

        def run(self) -> None:
            try:
                options = {} if self.generation is None else {"generation": self.generation}
                result = self.voice_manager.speak(self.speech_intention, **options)
                self.completed.emit(
                    result.status,
                    result.error or "",
                    result.provider,
                )
            except Exception as exc:
                self.completed.emit("failed", type(exc).__name__, "")

    class RealtimeWarmupWorker(QThread):
        completed = Signal(bool)

        def __init__(self, assistant):
            super().__init__()
            self.assistant = assistant

        def run(self) -> None:
            try:
                self.assistant.warmup_realtime()
                self.completed.emit(True)
            except Exception as exc:
                self.assistant.audit.record(
                    "realtime_warmup_failure",
                    error_type=type(exc).__name__,
                )
                self.completed.emit(False)


    class VoiceWarmupWorker(QThread):
        completed = Signal(bool)

        def __init__(self, voice_manager):
            super().__init__()
            self.voice_manager = voice_manager

        def run(self) -> None:
            warmup = getattr(self.voice_manager, "warmup", None)
            if not callable(warmup):
                self.completed.emit(True)
                return
            self.completed.emit(bool(warmup()))


    class AvatarStage(QWidget):
        """Escena emocional del avatar: grande, centrada y viva."""

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            model_root: Path | None = None,
            adaptive_camera: bool = False,
        ):
            super().__init__(parent)
            self.setObjectName("AvatarStage")
            self.setMouseTracking(True)
            self.setMinimumHeight(420)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._phase = 0.0
            self._blink = 0.0
            self._state = CompanionState()
            self._adaptive_camera = bool(adaptive_camera)
            self._pixmap = _load_best_pixmap(avatar_asset_candidates())
            self._vrm_view = None
            self._room_pixmap = _load_plain_pixmap(room_background_path())
            self._target_environment = synced_environment(datetime.now().astimezone())
            self._environment = self._target_environment
            self._warm_light = self._environment.warm_light
            self._cool_light = self._environment.cool_light
            self._lamp_glow = self._environment.lamp_glow
            self._window_glow = self._environment.window_glow
            self._particle_density = self._environment.particle_density
            self._weather_intensity = 1.0
            self._parallax_x = 0.0
            self._parallax_y = 0.0
            self._target_parallax_x = 0.0
            self._target_parallax_y = 0.0
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(33)
            self._environment_timer = QTimer(self)
            self._environment_timer.timeout.connect(self._refresh_environment)
            self._environment_timer.start(300000)
            self._idle_camera_timer = QTimer(self)
            self._idle_camera_timer.setSingleShot(True)
            self._idle_camera_timer.timeout.connect(self._show_idle_camera)
            model = preferred_avatar_model(model_root)
            use_25d_experiment = os.environ.get(
                "ANYDES_AVATAR_PRESENTATION", ""
            ).strip().lower() in {"2d_hq", "2d_hq_experimental"}
            if use_25d_experiment and WEBENGINE_25D_AVAILABLE and Experimental25DAvatarView is not None:
                try:
                    self._vrm_view = Experimental25DAvatarView(model, self)
                    self._vrm_view.ready_changed.connect(lambda _ready: self.update())
                    self._vrm_view.set_environment(self._environment)
                    self._vrm_view.show()
                except (OSError, ValueError):
                    self._vrm_view = None
            elif (
                WEBENGINE_AVAILABLE
                and VrmAvatarView is not None
                and renderer_assets_available()
                and model.is_file()
            ):
                try:
                    self._vrm_view = VrmAvatarView(model, self)
                    self._vrm_view.ready_changed.connect(lambda _ready: self.update())
                    self._vrm_view.set_environment(self._environment)
                    self._vrm_view.show()
                except (OSError, ValueError):
                    self._vrm_view = None

        def set_state(self, state: CompanionState) -> None:
            self._state = state
            if self._vrm_view is not None:
                self._vrm_view.set_avatar_state(state)
                if self._adaptive_camera:
                    mode = camera_mode_for_state(state)
                    if mode == "conversation":
                        self._idle_camera_timer.stop()
                        self._vrm_view.set_camera_mode("conversation")
                    else:
                        self._idle_camera_timer.start(520)
            self.update()

        def stop_speaking(self) -> None:
            if self._vrm_view is not None:
                self._vrm_view.stop_speaking()
                if self._adaptive_camera:
                    self._idle_camera_timer.start(520)

        def begin_turn(self, turn_id):
            begin = getattr(self._vrm_view, "begin_turn", None)
            if callable(begin):
                begin(turn_id)

        def start_lip_sync(
            self,
            timeline: list[dict[str, object]],
            *,
            duration_ms: float,
            body_duration_ms: float | None = None,
            offset_ms: float = 0.0,
            start_delay_ms: float = 0.0,
            playback_id: int = 0,
        ) -> None:
            if self._vrm_view is not None:
                self._vrm_view.start_lip_sync(
                    timeline,
                    duration_ms=duration_ms,
                    body_duration_ms=body_duration_ms,
                    offset_ms=offset_ms,
                    start_delay_ms=start_delay_ms,
                    playback_id=playback_id,
                )

        def _show_idle_camera(self) -> None:
            if self._vrm_view is not None and self._adaptive_camera:
                self._vrm_view.set_camera_mode("environment")

        def _tick(self) -> None:
            self._phase += 0.026
            self._blink = (self._blink + 0.018) % 1.0
            self._warm_light = _lerp(self._warm_light, self._target_environment.warm_light, 0.006)
            self._cool_light = _lerp(self._cool_light, self._target_environment.cool_light, 0.006)
            self._lamp_glow = _lerp(self._lamp_glow, self._target_environment.lamp_glow, 0.006)
            self._window_glow = _lerp(self._window_glow, self._target_environment.window_glow, 0.006)
            self._particle_density = _lerp(self._particle_density, self._target_environment.particle_density, 0.006)
            self._weather_intensity = _lerp(self._weather_intensity, 1.0, 0.01)
            self._parallax_x = _lerp(self._parallax_x, self._target_parallax_x, 0.075)
            self._parallax_y = _lerp(self._parallax_y, self._target_parallax_y, 0.075)
            self._environment = replace(
                self._target_environment,
                warm_light=self._warm_light,
                cool_light=self._cool_light,
                lamp_glow=self._lamp_glow,
                window_glow=self._window_glow,
                particle_density=self._particle_density,
            )
            self.update()

        def _refresh_environment(self) -> None:
            new_environment = synced_environment(datetime.now().astimezone())
            if new_environment.weather != self._target_environment.weather:
                self._weather_intensity = 0.0
            self._target_environment = new_environment
            if self._vrm_view is not None:
                self._vrm_view.set_environment(new_environment)
            self.update()

        def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
            center_x = max(1, self.width() / 2)
            center_y = max(1, self.height() / 2)
            self._target_parallax_x = max(-1.0, min(1.0, (event.position().x() - center_x) / center_x))
            self._target_parallax_y = max(-1.0, min(1.0, (event.position().y() - center_y) / center_y))
            if self._vrm_view is not None:
                self._vrm_view.set_pointer(self._target_parallax_x, self._target_parallax_y)
            super().mouseMoveEvent(event)

        def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
            self._target_parallax_x = 0.0
            self._target_parallax_y = 0.0
            if self._vrm_view is not None:
                self._vrm_view.set_pointer(0.0, 0.0)
            super().leaveEvent(event)

        def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
            if self._vrm_view is not None:
                self._vrm_view.setGeometry(self.rect())
                self._vrm_view.raise_()
            super().resizeEvent(event)

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
            _ = event
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            rect = self.rect()
            parallax = (self._parallax_x, self._parallax_y)
            _paint_room_scene(painter, rect, self._room_pixmap, self._phase, self._environment, parallax)
            _paint_floor_glow(painter, rect)
            _paint_living_details(painter, rect, self._phase, self._environment, parallax)
            _paint_avatar_rim_light(painter, rect, self._phase, self._environment)
            if self._vrm_view is None or not self._vrm_view.render_ready:
                self._paint_avatar(painter, rect)
            _paint_foreground_depth(painter, rect, self._phase, self._environment, parallax)
            painter.end()

        def _paint_avatar(self, painter: QPainter, rect: QRect) -> None:
            if self._pixmap.isNull():
                _paint_fallback_avatar(painter, rect, self._phase)
                return

            breath = math.sin(self._phase) * 9
            sway = math.sin(self._phase * 0.47) * 7
            target_h = int(rect.height() * PROFILE.subject_scale)
            target_w = int(target_h * self._pixmap.width() / max(1, self._pixmap.height()))
            target_w = min(target_w, int(rect.width() * 0.78))
            target_h = int(target_w * self._pixmap.height() / max(1, self._pixmap.width()))
            x = int((rect.width() - target_w) / 2 + sway)
            y = int((rect.height() - target_h) / 2 + rect.height() * PROFILE.subject_vertical_bias + breath)
            shadow = QRadialGradient(rect.center().x(), rect.bottom() - 68, rect.width() * 0.24)
            shadow.setColorAt(0.0, QColor(0, 0, 0, 145))
            shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(shadow)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.center().x() - rect.width() * 0.23, rect.bottom() - 122, rect.width() * 0.46, 74)
            painter.drawPixmap(QRect(x, y, target_w, target_h), self._pixmap)


    class StatusPill(QLabel):
        def __init__(self):
            super().__init__("😊 Tranquila")
            self.setObjectName("StatusPill")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setFixedHeight(34)

        def set_public_status(self, text: str) -> None:
            self.setText(text)


    class ScreenShareDialog(QDialog):
        def __init__(self, sources, *, external_provider: str = "", parent=None):
            super().__init__(parent)
            self.setWindowTitle("Compartir con Anydes")
            self.setModal(True)
            self.setMinimumWidth(460)
            layout = QVBoxLayout(self)
            title = QLabel("¿Qué quieres compartir con Anydes?")
            title.setObjectName("InnerWorldTitle")
            detail = QLabel(
                "El permiso dura únicamente mientras esta sesión permanezca abierta."
            )
            detail.setWordWrap(True)
            detail.setObjectName("InnerWorldSubtitle")
            layout.addWidget(title)
            layout.addWidget(detail)
            self.source = QComboBox()
            for item in sources:
                prefix = "Pantalla" if item.kind == "screen" else "Ventana"
                self.source.addItem(f"{prefix}: {item.label}", item.id)
            layout.addWidget(self.source)
            self.external = QCheckBox(
                f"Permitir que {external_provider} analice los frames de esta sesión"
            )
            self.external.setVisible(bool(external_provider))
            self.external.setChecked(False)
            layout.addWidget(self.external)
            privacy = QLabel(
                "Sin esta opción, la captura permanece local. "
                "Cerrar Anydes o pulsar Detener sensores elimina el permiso y el frame."
            )
            privacy.setWordWrap(True)
            privacy.setObjectName("InnerWorldSubtitle")
            layout.addWidget(privacy)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Cancel
                | QDialogButtonBox.StandardButton.Ok
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
                "Compartir ahora"
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def selected_source_id(self) -> str:
            return str(self.source.currentData() or "")

        def external_allowed(self) -> bool:
            return self.external.isVisible() and self.external.isChecked()


    class ChatPanel(QFrame):
        """Conversacion secundaria: pequena, limpia y sin burbujas pesadas."""

        def __init__(self):
            super().__init__()
            self.setObjectName("ChatPanel")
            self._messages: list[tuple[str, str, bool]] = []
            self._shadow(self, blur=24, alpha=55)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(12, 8, 12, 8)
            self.text = QTextEdit()
            self.text.setReadOnly(True)
            self.text.setMinimumHeight(82)
            self.text.setMaximumHeight(126)
            layout.addWidget(self.text)

        def add_message(self, speaker: str, text: str, *, user: bool = False) -> None:
            self._messages.append((speaker, text, user))
            self._render()

        def update_last_assistant_message(self, text: str) -> None:
            if self._messages and not self._messages[-1][2]:
                speaker, _, user = self._messages[-1]
                self._messages[-1] = (speaker, text, user)
            else:
                self._messages.append(("Anydes", text, False))
            self._render()

        def _render(self) -> None:
            rows = []
            for speaker, message, user in self._messages:
                color = PALETTE.pink if user else PALETTE.text_secondary
                rows.append(
                    f'<span style="color:{color}; font-weight:600">'
                    f"{html.escape(speaker)}</span>"
                    f'<span style="color:{PALETTE.text}">  '
                    f"{html.escape(message)}</span>"
                )
            self.text.setHtml("<br>".join(rows))
            scrollbar = self.text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        @staticmethod
        def _shadow(widget: QWidget, *, blur: int, alpha: int) -> None:
            effect = QGraphicsDropShadowEffect(widget)
            effect.setBlurRadius(blur)
            effect.setOffset(0, 12)
            effect.setColor(QColor(0, 0, 0, alpha))
            widget.setGraphicsEffect(effect)


    class InputDock(QFrame):
        def __init__(self, on_send, on_voice):
            super().__init__()
            self.setObjectName("InputDock")
            layout = QHBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(10)
            self.input = QLineEdit()
            self.input.setPlaceholderText("Habla con Anydes...")
            self.input.returnPressed.connect(self._send)
            layout.addWidget(self.input, 1)
            self.voice = QPushButton("🎤")
            self.voice.setObjectName("VoiceButton")
            self.voice.clicked.connect(on_voice)
            layout.addWidget(self.voice)
            self.send = QPushButton("Enviar")
            self.send.setObjectName("SendButton")
            self.send.clicked.connect(self._send)
            layout.addWidget(self.send)
            self._on_send = on_send

        def set_ready(self, ready: bool) -> None:
            self.input.setEnabled(ready)
            self.voice.setEnabled(ready)
            self.send.setEnabled(ready)
            self.input.setPlaceholderText(
                "Habla con Anydes..."
                if ready
                else "Anydes está despertando..."
            )
            if ready:
                restore_input_focus(self.input)

        def _send(self) -> None:
            text = self.input.text().strip()
            if not text:
                return
            self.input.clear()
            self._on_send(text)
            restore_input_focus(self.input)


    class InnerWorldPanel(QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("InnerWorldPanel")
            self.setMaximumWidth(0)
            self.setMinimumWidth(0)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 28, 24, 24)
            layout.setSpacing(14)
            title = QLabel("Mundo Interior")
            title.setObjectName("InnerWorldTitle")
            subtitle = QLabel("Solo señales suaves de presencia, no datos tecnicos.")
            subtitle.setObjectName("InnerWorldSubtitle")
            layout.addWidget(title)
            layout.addWidget(subtitle)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.cards = QWidget()
            self.cards_layout = QVBoxLayout(self.cards)
            self.cards_layout.setContentsMargins(0, 8, 0, 8)
            self.cards_layout.setSpacing(12)
            scroll.setWidget(self.cards)
            layout.addWidget(scroll, 1)

        def set_cards(self, cards) -> None:
            while self.cards_layout.count():
                item = self.cards_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            for card in cards:
                self.cards_layout.addWidget(WorldCard(card.title, card.value, card.description))
            self.cards_layout.addStretch(1)


    class WorldCard(QFrame):
        def __init__(self, title: str, value: str, description: str):
            super().__init__()
            self.setObjectName("WorldCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(5)
            title_label = QLabel(title)
            title_label.setObjectName("WorldCardTitle")
            value_label = QLabel(value)
            value_label.setObjectName("WorldCardValue")
            description_label = QLabel(description)
            description_label.setObjectName("WorldCardDescription")
            description_label.setWordWrap(True)
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            layout.addWidget(description_label)


    class AnydesPySideWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Anydes")
            self.resize(1280, 860)
            self.setMinimumSize(1040, 720)
            self.assistant = build_assistant()
            self.vision_session = (
                VisionSessionController(self.assistant.config)
                if self.assistant.config.realtime_phase4_enabled
                else None
            )
            self.voice_manager = build_voice_manager(
                self.assistant.config,
                metrics=self.assistant.metrics,
            )
            self.vision_residency = build_vision_residency_manager(
                self.assistant.config,
                audit=self.assistant.audit,
            )
            set_active_vision_residency_manager(self.vision_residency)
            self.vram_safety = None
            if self.assistant.config.vram_safety_enabled:
                self.vram_safety = build_vram_safety_manager(
                    self.assistant.config,
                    audit=self.assistant.audit,
                )
                attach_safety = getattr(self.voice_manager, "attach_vram_safety", None)
                if callable(attach_safety):
                    attach_safety(self.vram_safety)
                vision_model = self.assistant.config.resolved_vision_model()
                if vision_model:
                    self.vram_safety.register_action(
                        "suspend_idle_moondream",
                        lambda: self.vision_residency.release_if_idle(
                            reason="vram_pressure",
                        ),
                        blocked_by_tasks=("vision",),
                    )
                release_tts = getattr(self.voice_manager, "release_temporary_gpu_cache", None)
                if callable(release_tts):
                    tts_release_pressure = self.assistant.config.vram_tts_release_pressure
                    if tts_release_pressure == "critical":
                        self.vram_safety.register_action(
                            "inspect_tts_reclaimable",
                            self._inspect_tts_reclaimable,
                            warning=True,
                            critical=False,
                        )
                    self.vram_safety.register_action(
                        "release_tts_temporary_cache",
                        release_tts,
                        warning=tts_release_pressure == "warning",
                        critical=True,
                        blocked_by_tasks=("tts",),
                    )
                set_active_vram_safety_manager(self.vram_safety)
                self.vram_safety.start()
            self._worker: AssistantWorker | None = None
            self._turns = LatestTurn()
            self._closing = False
            self._voice_workers: set[VoiceWorker] = set()
            self._latest_voice_worker: VoiceWorker | None = None
            self._realtime_warmup_worker: RealtimeWarmupWorker | None = None
            self._voice_warmup_worker: VoiceWarmupWorker | None = None
            self._active_playback_id = 0
            self._inner_open = False
            self._streaming_answer = False
            self._build()
            self._refresh_world()
            self._sensor_timer = QTimer(self)
            self._sensor_timer.timeout.connect(self._refresh_sensor_indicator)
            self._sensor_timer.start(400)
            self._voice_event_timer = QTimer(self)
            self._voice_event_timer.timeout.connect(self._poll_voice_events)
            if callable(getattr(self.voice_manager, "poll_events", None)):
                self._voice_event_timer.start(20)
            self._task_notification_timer = QTimer(self)
            self._task_notification_timer.timeout.connect(
                self._poll_task_notifications
            )
            if self.assistant.config.realtime_phase1_enabled:
                self._task_notification_timer.start(250)
            if self.assistant.config.resolved_conversation_mode() == "realtime":
                self.input_dock.set_ready(False)
                self.status.set_public_status("✨ Despertando...")
                self._realtime_warmup_worker = RealtimeWarmupWorker(self.assistant)
                self._realtime_warmup_worker.completed.connect(
                    self._handle_realtime_ready
                )
                self._realtime_warmup_worker.start()

        def _build(self) -> None:
            self.setStyleSheet(companion_stylesheet())
            root = QWidget()
            root.setObjectName("SceneRoot")
            self.setCentralWidget(root)
            scene = QHBoxLayout(root)
            scene.setContentsMargins(0, 0, 0, 0)
            scene.setSpacing(0)

            main_area = QWidget()
            main_layout = QVBoxLayout(main_area)
            main_layout.setContentsMargins(LAYOUT.scene_padding, 22, LAYOUT.scene_padding, 22)
            main_layout.setSpacing(10)

            top_overlay = QHBoxLayout()
            self.sensor_indicator = QLabel("")
            self.sensor_indicator.setObjectName("SensorIndicator")
            self.sensor_indicator.setVisible(False)
            top_overlay.addWidget(self.sensor_indicator)
            self.stop_sensors_button = QPushButton("Detener sensores")
            self.stop_sensors_button.setObjectName("StopSensorsButton")
            self.stop_sensors_button.clicked.connect(self.stop_all_sensors)
            top_overlay.addWidget(self.stop_sensors_button)
            self.share_screen_button = QPushButton("Compartir pantalla")
            self.share_screen_button.setObjectName("ShareScreenButton")
            self.share_screen_button.clicked.connect(
                self.request_screen_permission
            )
            self.share_screen_button.setVisible(self.vision_session is not None)
            top_overlay.addWidget(self.share_screen_button)
            top_overlay.addStretch(1)
            self.inner_button = QPushButton("Mundo Interior")
            self.inner_button.setObjectName("InnerWorldButton")
            self.inner_button.clicked.connect(self.toggle_inner_world)
            top_overlay.addWidget(self.inner_button)
            main_layout.addLayout(top_overlay)

            self.avatar = AvatarStage(
                model_root=self.assistant.config.avatar_model_dir,
                adaptive_camera=self.assistant.config.avatar_adaptive_camera_enabled,
            )
            main_layout.addWidget(self.avatar, 1)

            self.status = StatusPill()
            status_row = QHBoxLayout()
            status_row.addStretch(1)
            status_row.addWidget(self.status)
            status_row.addStretch(1)
            main_layout.addLayout(status_row)

            self.chat = ChatPanel()
            main_layout.addWidget(self.chat, 0)

            self.input_dock = InputDock(self.send_text, self.voice_requested)
            main_layout.addWidget(self.input_dock, 0)

            scene.addWidget(main_area, 1)
            self.inner_world = InnerWorldPanel()
            scene.addWidget(self.inner_world)

        def send_text(self, text: str) -> None:
            if self._closing or not text.strip():
                return
            self._turns.submit(text)
            self.avatar.begin_turn(self._turns.revision)
            self._cancel_active_voice()
            self._latest_voice_worker = None
            if self._worker is not None:
                self._worker.requestInterruption()
            if (
                self.vision_session is not None
                and vision_permission_requested(text)
                and self.vision_session.snapshot().status != "active"
            ):
                self.request_screen_permission()
            self.chat.add_message("Tu", text, user=True)
            self._streaming_answer = False
            self._set_state(CompanionState(status=public_status("thinking"), expression="thinking", gesture="thinking", gaze="down"))
            restore_input_focus(self.input_dock.input)
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
            worker.partial.connect(self._route_partial)
            worker.answered.connect(self._route_answer)
            worker.failed.connect(self._route_failure)
            worker.finished.connect(self._response_finished)
            worker.start()

        def _route_partial(self, text):
            if self._turns.current(self.sender().turn_id):
                self._handle_partial(text)

        def _route_answer(self, text):
            if self._turns.current(self.sender().turn_id):
                self._handle_answer(text)

        def _route_failure(self, text):
            if self._turns.current(self.sender().turn_id):
                self._handle_failure(text)

        def _response_finished(self):
            self._response_worker_finished(self.sender())

        def _response_worker_finished(self, worker):
            if self._worker is worker:
                self._worker = None
            worker.deleteLater()
            self._start_pending_turn()

        def voice_requested(self) -> None:
            self._set_state(CompanionState(status=public_status("listening"), expression="listening", gesture="listening", gaze="user"))

        def _handle_realtime_ready(self, ready: bool) -> None:
            self.input_dock.set_ready(True)
            self.status.set_public_status(
                public_status("idle") if ready else "No pude despertar mi voz interior"
            )
            if ready and self.assistant.config.new_voice_enabled:
                self._voice_warmup_worker = VoiceWarmupWorker(self.voice_manager)
                self._voice_warmup_worker.completed.connect(
                    self._handle_voice_warmup_ready
                )
                self._voice_warmup_worker.start()

        def _handle_voice_warmup_ready(self, ready: bool) -> None:
            self.assistant.audit.record("voice_warmup_completed", ready=ready)

        def toggle_inner_world(self) -> None:
            self._inner_open = not self._inner_open
            animation = QPropertyAnimation(self.inner_world, b"maximumWidth", self)
            animation.setDuration(260)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.setStartValue(self.inner_world.maximumWidth())
            animation.setEndValue(LAYOUT.inner_world_width if self._inner_open else 0)
            animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
            self._refresh_world()
            restore_input_focus(self.input_dock.input)

        def _handle_partial(self, text: str) -> None:
            self.chat.update_last_assistant_message(text)
            self._streaming_answer = True

        def _handle_answer(self, text: str) -> None:
            if self._streaming_answer:
                self.chat.update_last_assistant_message(text)
            else:
                self.chat.add_message("Anydes", text)
            self._streaming_answer = False
            self.input_dock.set_ready(True)
            intention = self.assistant.last_intention
            state = state_from_intention(intention)
            if self.assistant.config.new_voice_enabled:
                state = replace(
                    state,
                    speech={},
                    lip_sync={"enabled": False, "duration_hint_ms": 0},
                )
            self._set_state(state)
            self._refresh_world()
            if intention is not None and intention.speech_intention.text.strip():
                worker = VoiceWorker(
                    self.voice_manager,
                    intention.speech_intention,
                )
                self._voice_workers.add(worker)
                self._latest_voice_worker = worker
                worker.completed.connect(
                    lambda status, error, provider, current=worker: self._handle_voice_completed(
                        current,
                        status,
                        error,
                        provider,
                    )
                )
                worker.finished.connect(
                    lambda current=worker: self._release_voice_worker(current)
                )
                worker.start()

        def _handle_failure(self, text: str) -> None:
            if self._streaming_answer:
                self.chat.update_last_assistant_message(text)
            else:
                self.chat.add_message("Anydes", text)
            self._streaming_answer = False
            self.input_dock.set_ready(True)
            self._set_state(CompanionState())

        def _handle_voice_completed(
            self,
            worker: VoiceWorker,
            status: str,
            error: str,
            provider: str,
        ) -> None:
            if worker is not self._latest_voice_worker:
                return
            if status == "failed":
                self.avatar.stop_speaking()
            error_type = ""
            if error:
                pieces = error.split(":", 2)
                error_type = pieces[1] if len(pieces) >= 2 else pieces[0]
            self.assistant.audit.record(
                "interface_voice_completed",
                status=status,
                provider=provider,
                error_type=error_type,
            )
            backend_trace = getattr(self.voice_manager.backend, "last_trace", None)
            if backend_trace is not None:
                self.assistant.audit.record(
                    "interface_voice_trace",
                    status=backend_trace.status,
                    phase=backend_trace.phase,
                    error_code=backend_trace.error_code,
                    synthesis_ms=round(backend_trace.synthesis_ms, 3),
                    processing_ms=round(backend_trace.processing_ms, 3),
                    playback_ms=round(backend_trace.playback_ms, 3),
                    sample_rate=backend_trace.sample_rate,
                    channels=backend_trace.channels,
                    frame_count=backend_trace.frame_count,
                    cancelled=backend_trace.cancelled,
                    output_mode=backend_trace.output_mode,
                )
            if status == "failed" and error:
                self.assistant.audit.record("interface_voice_failure", error_type=error_type, provider=provider)
                self.status.set_public_status("Mi voz no está disponible por el momento")

        def _release_voice_worker(self, worker: VoiceWorker) -> None:
            self._voice_workers.discard(worker)
            if worker is self._latest_voice_worker:
                self._latest_voice_worker = None
            worker.deleteLater()

        def _cancel_active_voice(self) -> None:
            self.voice_manager.cancel()
            self.avatar.stop_speaking()
            self._active_playback_id = 0

        def _inspect_tts_reclaimable(self) -> dict[str, object]:
            sample = self.vram_safety.latest_sample if self.vram_safety is not None else None
            if sample is None:
                return {"status": "unavailable"}
            reclaimable = max(0.0, sample.tts_reserved_mb - sample.tts_allocated_mb)
            return {
                "status": "observed",
                "allocated_mb": round(sample.tts_allocated_mb, 3),
                "reserved_mb": round(sample.tts_reserved_mb, 3),
                "reclaimable_mb": round(reclaimable, 3),
                "released": False,
            }

        def _poll_voice_events(self) -> None:
            poll = getattr(self.voice_manager, "poll_events", None)
            if not callable(poll):
                return
            for event in poll():
                if event.kind == "playback_started":
                    playback_id = int(event.payload.get("playback_id") or 0)
                    self._active_playback_id = playback_id
                    clock = float(event.payload.get("audio_clock") or event.created_monotonic)
                    offset_ms, start_delay_ms = visual_alignment_from_audio_clock(clock)
                    timeline = event.payload.get("timeline")
                    if not isinstance(timeline, list):
                        timeline = []
                    duration_ms = float(event.payload.get("duration_ms") or 0.0)
                    if event.payload.get("timeline_source") == "measured_pcm":
                        visual_timeline = timeline
                        visual_duration_ms = float(event.payload.get("active_end_ms", duration_ms))
                    else:
                        visual_timeline, visual_duration_ms = stretch_final_visual_lip_sync(timeline, duration_ms)
                    self.avatar.start_lip_sync(
                        visual_timeline,
                        duration_ms=visual_duration_ms,
                        body_duration_ms=duration_ms,
                        offset_ms=offset_ms,
                        start_delay_ms=start_delay_ms,
                        playback_id=playback_id,
                    )
                elif event.kind in {
                    "playback_stopped",
                    "playback_interrupted",
                    "voice_failed",
                }:
                    playback_id = int(event.payload.get("playback_id") or 0)
                    if event.kind != "voice_failed" and (
                        not self._active_playback_id
                        or playback_id != self._active_playback_id
                    ):
                        continue
                    QTimer.singleShot(
                        0,
                        lambda pid=playback_id, kind=event.kind, gen=getattr(self.voice_manager, "generation", None): self._finish_visual_playback(
                            pid,
                            kind,
                            gen,
                        ),
                    )
                elif event.kind == "emergency_voice_started":
                    self.status.set_public_status("Usando la voz de emergencia autorizada")

        def _finish_visual_playback(self, playback_id: int, event_kind: str, generation: int | None = None) -> None:
            if generation is not None and generation != getattr(self.voice_manager, "generation", None):
                return
            if event_kind != "voice_failed" and playback_id != self._active_playback_id:
                return
            self.avatar.stop_speaking()
            self._active_playback_id = 0
            state = state_from_intention(self.assistant.last_intention)
            self._set_state(
                replace(
                    state,
                    status=public_status("idle"),
                    gesture="idle",
                    speech={},
                    lip_sync={"enabled": False, "duration_hint_ms": 0},
                )
            )

        def _set_state(self, state: CompanionState) -> None:
            self.status.set_public_status(state.status)
            self.avatar.set_state(state)

        def _refresh_world(self) -> None:
            self.inner_world.set_cards(inner_world_cards(self.assistant.last_internal_context, self.assistant.last_intention))

        def _refresh_sensor_indicator(self) -> None:
            watching = get_screen_state().is_watching() or sensor_hub.active("screen")
            if watching:
                provider = self.assistant.config.resolved_vision_provider()
                mode = "externa" if is_external_provider(provider) else "local"
                source_label = get_screen_state().source_snapshot()[2]
                source = f": {source_label}" if source_label else ""
                self.sensor_indicator.setText(
                    f"Observando {mode}{source}"
                )
                self.sensor_indicator.setVisible(True)
            else:
                self.sensor_indicator.setVisible(False)

        def request_screen_permission(self) -> None:
            try:
                self._request_screen_permission()
            finally:
                restore_input_focus(self.input_dock.input)

        def _request_screen_permission(self) -> None:
            controller = self.vision_session
            if controller is None:
                return
            sources = controller.request_permission()
            if not sources:
                QMessageBox.warning(
                    self,
                    "No pude iniciar la visión",
                    "No encontré una pantalla o ventana disponible para compartir.",
                )
                return
            provider = self.assistant.config.resolved_vision_provider()
            external_provider = provider if is_external_provider(provider) else ""
            dialog = ScreenShareDialog(
                sources,
                external_provider=external_provider,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                controller.stop()
                self._refresh_sensor_indicator()
                return
            snapshot = controller.start(
                dialog.selected_source_id(),
                allow_external_analysis=dialog.external_allowed(),
            )
            if snapshot.status != "active" or not snapshot.frame_available:
                QMessageBox.warning(
                    self,
                    "No pude iniciar la visión",
                    "La captura no produjo un frame válido. No se concedió el permiso.",
                )
                return
            self._set_state(
                CompanionState(
                    status=public_status("observing"),
                    expression="attentive",
                    gesture="idle",
                    gaze="screen",
                )
            )
            self._refresh_sensor_indicator()

        def _poll_task_notifications(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                return
            for text in self.assistant.drain_realtime_notifications():
                self.chat.add_message("Anydes", text)
                self._set_state(
                    CompanionState(
                        status=public_status("idle"),
                        expression="calm",
                        gesture="idle",
                        gaze="user",
                    )
                )

        def stop_all_sensors(self) -> None:
            if self.vision_session is not None:
                self.vision_session.stop()
            sensor_hub.stop_all()
            stop_all_screen_watching()
            session_consents.revoke_all()
            self.sensor_indicator.setVisible(False)
            self._set_state(CompanionState(status="Sensores detenidos"))

        def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
            self._closing = True
            self._turns.close()
            if self._worker is not None:
                self._worker.requestInterruption()
            self.stop_all_sensors()
            self._cancel_active_voice()
            for worker in tuple(self._voice_workers):
                if worker.isRunning():
                    worker.wait(5000)
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(10000)
            if (
                self._realtime_warmup_worker is not None
                and self._realtime_warmup_worker.isRunning()
            ):
                self._realtime_warmup_worker.wait(10000)
            if (
                self._voice_warmup_worker is not None
                and self._voice_warmup_worker.isRunning()
            ):
                self._voice_warmup_worker.wait(10000)
            if self.vram_safety is not None:
                self.vram_safety.stop()
                set_active_vram_safety_manager(None)
            self.vision_residency.close()
            set_active_vision_residency_manager(None)
            close_voice = getattr(self.voice_manager, "close", None)
            if callable(close_voice):
                close_voice()
            self.assistant.close()
            super().closeEvent(event)


def _load_best_pixmap(paths: tuple[Path, ...]) -> "QPixmap":
    for path in paths:
        if not path.exists():
            continue
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if not image.isNull():
            image = _remove_connected_gray_background(image)
            image = _crop_to_visible_content(image)
            image = _repair_avatar_hair_material(image)
            image = _grade_avatar_image(image)
        pixmap = QPixmap.fromImage(image)
        if not pixmap.isNull():
            return pixmap
    return QPixmap()


def _grade_avatar_image(image: "QImage") -> "QImage":
    """Desatura y oscurece morados del avatar para separarlo de la habitacion."""

    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    width = image.width()
    height = image.height()
    for y in range(height):
        for x in range(width):
            color = image.pixelColor(x, y)
            if color.alpha() <= 8:
                continue
            red, green, blue = color.red(), color.green(), color.blue()
            if blue > 70 and red > 55 and green < 95 and (blue + red) > green * 2.4:
                gray = int(red * 0.32 + green * 0.36 + blue * 0.32)
                color.setRed(max(28, int((red * 0.68 + gray * 0.32) * 0.82)))
                color.setGreen(max(24, int((green * 0.72 + gray * 0.28) * 0.84)))
                color.setBlue(max(42, int((blue * 0.66 + gray * 0.34) * 0.82)))
                image.setPixelColor(x, y, color)
            elif red > 170 and green < 130 and blue > 120:
                color.setRed(min(255, int(red * 1.02)))
                color.setGreen(int(green * 0.92))
                color.setBlue(int(blue * 0.92))
                image.setPixelColor(x, y, color)
    return image


def _repair_avatar_hair_material(image: "QImage") -> "QImage":
    """Corrige zonas grises del render actual sin tocar el fondo ni la piel."""

    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    width = image.width()
    height = image.height()
    top_limit = int(height * 0.18)
    left_limit = int(width * 0.24)
    right_limit = int(width * 0.76)
    if top_limit <= 0:
        return image

    for y in range(top_limit):
        vertical_mix = y / max(1, top_limit)
        for x in range(left_limit, right_limit):
            color = image.pixelColor(x, y)
            if color.alpha() <= 8:
                continue
            red, green, blue = color.red(), color.green(), color.blue()
            saturation = max(red, green, blue) - min(red, green, blue)
            brightness = (red + green + blue) / 3
            cap_dx = (x - width * 0.5) / max(1.0, width * 0.22)
            cap_dy = (y - height * 0.065) / max(1.0, height * 0.058)
            cap_mask = cap_dx * cap_dx + cap_dy * cap_dy < 1.0
            local_saturation_limit = 190 if cap_mask else 118
            if saturation > local_saturation_limit or brightness > 196 or brightness < 54:
                continue

            center_bias = 1.0 - abs((x / max(1, width)) - 0.5) * 1.75
            shade = max(0.0, min(1.0, (brightness - 62) / 104))
            cap_lift = max(0.0, 1.0 - vertical_mix * 1.45) if cap_mask else 0.0
            red_target = int(202 + 46 * shade + 18 * center_bias + 18 * cap_lift)
            green_target = int(68 + 25 * shade + 10 * vertical_mix + 12 * cap_lift)
            blue_target = int(96 + 28 * shade + 8 * cap_lift)
            color.setRed(max(red, min(255, red_target)))
            color.setGreen(max(48, min(122, green_target)))
            color.setBlue(max(78, min(148, blue_target)))
            image.setPixelColor(x, y, color)
    return image


def _load_plain_pixmap(path: Path) -> "QPixmap":
    if not path.exists():
        return QPixmap()
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    return QPixmap.fromImage(reader.read())


def _remove_connected_gray_background(image: "QImage") -> "QImage":
    """Hace transparente el fondo gris conectado a los bordes del render."""

    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    width = image.width()
    height = image.height()
    if width <= 0 or height <= 0:
        return image

    corners = (
        image.pixelColor(0, 0),
        image.pixelColor(width - 1, 0),
        image.pixelColor(0, height - 1),
        image.pixelColor(width - 1, height - 1),
    )
    base = QColor(
        sum(color.red() for color in corners) // len(corners),
        sum(color.green() for color in corners) // len(corners),
        sum(color.blue() for color in corners) // len(corners),
    )

    def is_background(x: int, y: int) -> bool:
        color = image.pixelColor(x, y)
        diff = abs(color.red() - base.red()) + abs(color.green() - base.green()) + abs(color.blue() - base.blue())
        saturation = max(color.red(), color.green(), color.blue()) - min(color.red(), color.green(), color.blue())
        return diff < 90 and saturation < 42

    step = 1
    queue: deque[tuple[int, int]] = deque()
    seen: set[tuple[int, int]] = set()
    for x in range(0, width, step):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(0, height, step):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if x < 0 or y < 0 or x >= width or y >= height or (x, y) in seen:
            continue
        seen.add((x, y))
        if not is_background(x, y):
            continue
        color = image.pixelColor(x, y)
        color.setAlpha(0)
        image.setPixelColor(x, y, color)
        queue.append((x + step, y))
        queue.append((x - step, y))
        queue.append((x, y + step))
        queue.append((x, y - step))

    return image


def _crop_to_visible_content(image: "QImage") -> "QImage":
    width = image.width()
    height = image.height()
    left = width
    top = height
    right = 0
    bottom = 0
    for y in range(height):
        for x in range(width):
            if image.pixelColor(x, y).alpha() > 8:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    if left >= right or top >= bottom:
        return image
    padding_x = int((right - left) * 0.05)
    padding_y = int((bottom - top) * 0.04)
    left = max(0, left - padding_x)
    top = max(0, top - padding_y)
    right = min(width - 1, right + padding_x)
    bottom = min(height - 1, bottom + padding_y)
    return image.copy(QRect(left, top, right - left + 1, bottom - top + 1))


def _lerp(current: float, target: float, amount: float) -> float:
    return current + (target - current) * amount


def _paint_room_scene(
    painter: "QPainter",
    rect: "QRect",
    pixmap: "QPixmap",
    phase: float,
    environment,
    parallax: tuple[float, float],
) -> None:
    if pixmap.isNull():
        _paint_fallback_room(painter, rect)
    else:
        _draw_cover_pixmap(painter, rect, pixmap, phase, parallax)

    _paint_time_of_day_grade(painter, rect, environment)
    _paint_window_weather(painter, rect, phase, environment, parallax)

    glow = QRadialGradient(rect.center().x(), rect.center().y() + rect.height() * 0.08, rect.width() * 0.48)
    glow.setColorAt(0.0, QColor(255, 180, 98, 26))
    glow.setColorAt(0.45, QColor(70, 125, 205, 18))
    glow.setColorAt(1.0, QColor(20, 17, 31, 0))
    painter.fillRect(rect, glow)


def _draw_cover_pixmap(
    painter: "QPainter",
    rect: "QRect",
    pixmap: "QPixmap",
    phase: float,
    parallax: tuple[float, float],
) -> None:
    scale = max(rect.width() / max(1, pixmap.width()), rect.height() / max(1, pixmap.height()))
    width = int(pixmap.width() * scale)
    height = int(pixmap.height() * scale)
    parallax_x = int(math.sin(phase * 0.18) * 5 + parallax[0] * -18)
    parallax_y = int(math.cos(phase * 0.14) * 3 + parallax[1] * -10)
    target = QRect(
        int((rect.width() - width) / 2 + parallax_x),
        int((rect.height() - height) / 2 + parallax_y),
        width,
        height,
    )
    painter.drawPixmap(target, pixmap)


def _paint_fallback_room(painter: "QPainter", rect: "QRect") -> None:
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0.0, QColor(PALETTE.background))
    gradient.setColorAt(0.45, QColor(PALETTE.panel))
    gradient.setColorAt(1.0, QColor(PALETTE.panel_secondary))
    painter.fillRect(rect, gradient)


def _paint_time_of_day_grade(painter: "QPainter", rect: "QRect", environment) -> None:
    if environment.time_of_day == "dawn":
        warm = QColor(255, 190, 92, int(44 * environment.warm_light))
        cool = QColor(112, 165, 230, int(22 * environment.cool_light))
    elif environment.time_of_day == "day":
        warm = QColor(255, 242, 205, int(28 * environment.warm_light))
        cool = QColor(135, 190, 255, int(30 * environment.cool_light))
    elif environment.time_of_day == "sunset":
        warm = QColor(255, 132, 62, int(58 * environment.warm_light))
        cool = QColor(95, 132, 210, int(18 * environment.cool_light))
    else:
        warm = QColor(255, 166, 86, int(42 * environment.warm_light))
        cool = QColor(42, 92, 168, int(58 * environment.cool_light))

    window_light = QLinearGradient(rect.topLeft(), rect.center())
    window_light.setColorAt(0.0, cool)
    window_light.setColorAt(0.58, QColor(cool.red(), cool.green(), cool.blue(), 0))
    painter.fillRect(rect, window_light)

    lamp = QRadialGradient(int(rect.width() * 0.74), int(rect.height() * 0.54), int(rect.width() * 0.30))
    lamp.setColorAt(0.0, QColor(warm.red(), warm.green(), warm.blue(), int(86 * environment.lamp_glow)))
    lamp.setColorAt(0.58, QColor(warm.red(), warm.green(), warm.blue(), int(22 * environment.lamp_glow)))
    lamp.setColorAt(1.0, QColor(warm.red(), warm.green(), warm.blue(), 0))
    painter.fillRect(rect, lamp)

    vignette = QRadialGradient(rect.center(), rect.width() * 0.62)
    vignette.setColorAt(0.0, QColor(20, 17, 31, 0))
    vignette.setColorAt(0.72, QColor(12, 12, 16, 10))
    vignette.setColorAt(1.0, QColor(8, 8, 12, 92))
    painter.fillRect(rect, vignette)


def _paint_window_weather(
    painter: "QPainter",
    rect: "QRect",
    phase: float,
    environment,
    parallax: tuple[float, float],
) -> None:
    window = QRect(int(rect.width() * 0.05), int(rect.height() * 0.04), int(rect.width() * 0.42), int(rect.height() * 0.68))
    weather = environment.weather
    if weather in {"rain", "storm"}:
        _paint_rain_layers(painter, window, phase, storm=weather == "storm", parallax=parallax)
        _paint_glass_droplets(painter, window, phase)
        if weather == "storm":
            _paint_lightning(painter, rect, phase)
    elif weather == "snow":
        _paint_snow_layers(painter, window, phase, parallax=parallax)
    elif weather == "cloudy":
        _paint_diffuse_cloud_light(painter, window, phase)
    else:
        _paint_sun_beams(painter, window, rect, phase)


def _paint_rain_layers(painter: "QPainter", window: "QRect", phase: float, *, storm: bool, parallax: tuple[float, float]) -> None:
    layers = (
        (36, 0.55, 86, 12, 1),
        (54, 0.95, 112, 18, 2),
        (22, 1.45, 146, 26, 3),
    )
    for count, speed, alpha, length, width in layers:
        painter.setPen(QColor(170, 205, 245, min(185, alpha + (34 if storm else 0))))
        pen = painter.pen()
        pen.setWidth(width)
        painter.setPen(pen)
        for index in range(count + (18 if storm else 0)):
            x = window.left() + int((index * 43 + phase * 130 * speed + parallax[0] * 18 * width) % max(1, window.width()))
            y = window.top() + int((index * 79 + phase * 210 * speed) % max(1, window.height()))
            painter.drawLine(x, y, x - int(5 + width * 2), y + length)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(210, 230, 255, 55))
    for index in range(18 if storm else 10):
        x = window.left() + int((index * 91 + phase * 70) % max(1, window.width()))
        y = window.bottom() - int((index * 17 + phase * 23) % max(18, window.height() // 6))
        painter.drawEllipse(x, y, 3, 1)


def _paint_glass_droplets(painter: "QPainter", window: "QRect", phase: float) -> None:
    painter.setPen(QColor(210, 230, 255, 80))
    for index in range(18):
        x = window.left() + int((index * 67) % max(1, window.width()))
        y = window.top() + int((index * 97 + phase * (12 + index % 5)) % max(1, window.height()))
        painter.drawLine(x, y, x - 1, y + 12 + index % 9)
        painter.drawPoint(x + 1, y + 2)


def _paint_lightning(painter: "QPainter", rect: "QRect", phase: float) -> None:
    pulse = max(0.0, math.sin(phase * 1.7 + 2.2))
    if pulse < 0.965:
        return
    strength = int((pulse - 0.965) / 0.035 * 135)
    flash = QRadialGradient(int(rect.width() * 0.28), int(rect.height() * 0.16), int(rect.width() * 0.72))
    flash.setColorAt(0.0, QColor(205, 225, 255, strength))
    flash.setColorAt(0.42, QColor(120, 170, 255, int(strength * 0.35)))
    flash.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.fillRect(rect, flash)


def _paint_snow_layers(painter: "QPainter", window: "QRect", phase: float, *, parallax: tuple[float, float]) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    for count, speed, size, alpha in ((30, 18, 2, 95), (20, 31, 3, 128), (10, 48, 4, 150)):
        painter.setBrush(QColor(255, 255, 255, alpha))
        for index in range(count):
            x = window.left() + int((index * 61 + math.sin(phase * 0.7 + index) * 32 + parallax[0] * 12) % max(1, window.width()))
            y = window.top() + int((index * 47 + phase * speed) % max(1, window.height()))
            painter.drawEllipse(x, y, size, size)


def _paint_diffuse_cloud_light(painter: "QPainter", window: "QRect", phase: float) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    for index in range(7):
        cx = window.left() + int((index * 103 + phase * 7) % max(1, window.width()))
        cy = window.top() + int(window.height() * (0.18 + (index % 3) * 0.13))
        cloud = QRadialGradient(cx, cy, int(window.width() * 0.22))
        cloud.setColorAt(0.0, QColor(205, 214, 225, 30))
        cloud.setColorAt(1.0, QColor(205, 214, 225, 0))
        painter.fillRect(window, cloud)


def _paint_sun_beams(painter: "QPainter", window: "QRect", rect: "QRect", phase: float) -> None:
    origin_x = window.left() + int(window.width() * 0.22)
    origin_y = window.top() + int(window.height() * 0.18)
    painter.setPen(Qt.PenStyle.NoPen)
    for index in range(4):
        offset = math.sin(phase * 0.16 + index) * 14
        path = QPainterPath(QPointF(origin_x + index * 24, origin_y + offset))
        path.lineTo(QPointF(rect.center().x() + index * 52, rect.bottom()))
        path.lineTo(QPointF(rect.center().x() - 120 + index * 46, rect.bottom()))
        path.closeSubpath()
        gradient = QLinearGradient(QPointF(origin_x, origin_y), QPointF(rect.center().x(), rect.bottom()))
        gradient.setColorAt(0.0, QColor(255, 232, 176, 34))
        gradient.setColorAt(1.0, QColor(255, 232, 176, 0))
        painter.fillPath(path, gradient)


def _paint_living_details(painter: "QPainter", rect: "QRect", phase: float, environment, parallax: tuple[float, float]) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    particle_count = int(42 * environment.particle_density)
    for index in range(particle_count):
        x = int((index * 97 + math.sin(phase * 0.7 + index) * 24 + parallax[0] * 16) % max(1, rect.width()))
        y = int((index * 53 + phase * 9 + math.cos(phase * 0.4 + index) * 18 + parallax[1] * 8) % max(1, rect.height()))
        alpha = 28 + int((math.sin(phase + index) + 1.0) * 24)
        painter.setBrush(QColor(255, 210, 160, alpha))
        painter.drawEllipse(x, y, 2, 2)

    curtain_sway = math.sin(phase * 0.55) * 8
    painter.setPen(QColor(210, 225, 245, 34))
    for index in range(3):
        x = int(rect.width() * (0.23 + index * 0.018) + curtain_sway + parallax[0] * 5)
        painter.drawLine(x, int(rect.height() * 0.08), x + 12, int(rect.height() * 0.63))

    leaf_sway = math.sin(phase * 0.72) * 5
    painter.setPen(QColor(145, 236, 190, 36))
    base_x = int(rect.width() * 0.90)
    base_y = int(rect.height() * 0.25)
    for index in range(6):
        painter.drawLine(base_x, base_y + index * 16, int(base_x + leaf_sway + 22), base_y + index * 16 + 8)

    _paint_city_life(painter, rect, phase)


def _paint_city_life(painter: "QPainter", rect: "QRect", phase: float) -> None:
    city = QRect(int(rect.width() * 0.08), int(rect.height() * 0.07), int(rect.width() * 0.36), int(rect.height() * 0.52))
    painter.setPen(Qt.PenStyle.NoPen)
    for index in range(24):
        twinkle = (math.sin(phase * 0.22 + index * 1.7) + 1.0) * 0.5
        if twinkle < 0.35:
            continue
        x = city.left() + int((index * 47) % max(1, city.width()))
        y = city.top() + int((index * 29) % max(1, city.height()))
        painter.setBrush(QColor(255, 210, 130, int(28 + twinkle * 55)))
        painter.drawRoundedRect(QRect(x, y, 3, 5), 1, 1)
    painter.setBrush(QColor(255, 232, 190, 90))
    car_x = city.left() + int((phase * 42) % max(1, city.width()))
    car_y = city.bottom() - 18
    painter.drawEllipse(car_x, car_y, 3, 2)
    painter.drawEllipse(car_x + 8, car_y + 1, 2, 2)


def _paint_avatar_rim_light(painter: "QPainter", rect: "QRect", phase: float, environment) -> None:
    cx = rect.center().x() + int(math.sin(phase * 0.3) * 4)
    cy = int(rect.height() * 0.43)
    rim = QRadialGradient(cx, cy, rect.width() * 0.24)
    rim.setColorAt(0.0, QColor(255, 196, 120, int(62 * environment.lamp_glow)))
    rim.setColorAt(0.42, QColor(86, 150, 230, int(36 * environment.cool_light)))
    rim.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.fillRect(rect, rim)


def _paint_foreground_depth(painter: "QPainter", rect: "QRect", phase: float, environment, parallax: tuple[float, float]) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    x_shift = int(parallax[0] * 24 + math.sin(phase * 0.35) * 3)
    y_base = int(rect.height() * 0.78 + parallax[1] * 10)
    painter.setBrush(QColor(12, 18, 14, 80))
    for index in range(5):
        path = QPainterPath()
        x = int(rect.width() * 0.03) + x_shift + index * 14
        path.moveTo(x, rect.bottom())
        path.cubicTo(x + 18, y_base - 70, x + 24, y_base - 24, x + 45, rect.bottom())
        path.closeSubpath()
        painter.fillPath(path, QColor(18, 42, 28, 48))


def _paint_floor_glow(painter: "QPainter", rect: "QRect") -> None:
    glow = QRadialGradient(rect.center().x(), rect.bottom() - 60, rect.width() * 0.38)
    glow.setColorAt(0.0, QColor(255, 180, 94, 38))
    glow.setColorAt(0.45, QColor(70, 120, 190, 18))
    glow.setColorAt(1.0, QColor(20, 17, 31, 0))
    painter.fillRect(rect, glow)


def _paint_fallback_avatar(painter: "QPainter", rect: "QRect", phase: float) -> None:
    cx = rect.center().x()
    cy = rect.center().y() + int(math.sin(phase) * 8)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#F48FBF"))
    painter.drawEllipse(cx - 92, cy - 188, 184, 184)
    painter.setBrush(QColor("#FFD2E3"))
    painter.drawEllipse(cx - 66, cy - 144, 132, 140)
    painter.setBrush(QColor("#6D4ACF"))
    path = QPainterPath()
    path.moveTo(cx - 132, cy + 210)
    path.lineTo(cx - 74, cy - 2)
    path.lineTo(cx + 74, cy - 2)
    path.lineTo(cx + 132, cy + 210)
    path.closeSubpath()
    painter.drawPath(path)


def _interface_failure_layer(error_type: str) -> str:
    if error_type in {"DataProtectionError", "PermissionError", "OSError"}:
        return "local_state"
    if error_type in {"TimeoutError", "URLError", "ConnectionError"}:
        return "transport"
    return "interface_response"


def _interface_failure_text(error_type: str, *, stabilized: bool) -> str:
    if not stabilized:
        return "Perdon... algo interrumpio mi respuesta. Podrias repetirmelo?"
    layer = _interface_failure_layer(error_type)
    if layer == "local_state":
        return "No pude acceder correctamente a mi estado local esta vez. Tu mensaje sigue aquí; puedes intentarlo otra vez."
    if layer == "transport":
        return "Mi respuesta se interrumpió antes de terminar. Puedo intentarla otra vez."
    return "No conseguí completar esa respuesta, pero tu mensaje no se perdió. Repítemelo, porfa."


def main() -> None:
    if _PYSIDE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "La nueva interfaz de Anydes requiere PySide6. Instala dependencias con: "
            "pip install -r requirements-interface.txt"
        ) from _PYSIDE_IMPORT_ERROR
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = AnydesPySideWindow()
    window.show()
    auto_close_ms = int(os.environ.get("ANYDES_RUNTIME_AUTO_CLOSE_MS", "0") or 0)
    if auto_close_ms > 0:
        QTimer.singleShot(auto_close_ms, window.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
