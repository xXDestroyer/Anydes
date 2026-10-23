from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .store import SettingsStore, WorkspaceValidationError


class AnydesSettingsDialog(QDialog):
    """Companion-styled, allowlisted view over real runtime state."""

    categories = (
        "General",
        "Archivos",
        "Audio",
        "Voz",
        "Cerebro",
        "Herramientas",
        "Permisos",
        "Visión",
        "Memoria",
        "Estado",
        "Avanzado",
    )

    def __init__(self, owner, controller, settings: SettingsStore) -> None:
        super().__init__(owner)
        self.owner = owner
        self.controller = controller
        self.settings = settings
        self.setWindowTitle("Configuración de Anydes")
        self.setModal(False)
        self.setMinimumSize(820, 610)
        self.resize(940, 680)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._labels: dict[str, QLabel] = {}
        self._advanced_visible = False
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self.refresh()

    def _build(self) -> None:
        self.setStyleSheet(
            "QDialog{background:#18171d;color:#f3eef3;}"
            "QListWidget{background:#201e25;border:0;border-right:1px solid #39343e;"
            "padding:8px;color:#dcd4dc;font-size:13px;}"
            "QListWidget::item{padding:10px 12px;border-radius:4px;}"
            "QListWidget::item:selected{background:#59475d;color:white;}"
            "QLabel{color:#e9e2e9;}"
            "QLabel#SectionTitle{font-size:19px;font-weight:600;color:#ffffff;}"
            "QLabel#SectionHint{color:#aca5b0;}"
            "QLabel#Value{color:#c9e4d7;}"
            "QPushButton{background:#3c3441;color:white;border:1px solid #625369;"
            "padding:8px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#514258;}"
            "QFrame#Divider{background:#39343e;max-height:1px;}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 18, 14)
        title = QLabel("Configuración de Anydes")
        title.setObjectName("SectionTitle")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        close = QPushButton("Cerrar")
        close.clicked.connect(self.hide)
        header_layout.addWidget(close)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setFixedWidth(180)
        for category in self.categories:
            self.navigation.addItem(category)
        body.addWidget(self.navigation)
        self.pages = QStackedWidget()
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)

        self.pages.addWidget(self._general_page())
        self.pages.addWidget(self._files_page())
        self.pages.addWidget(self._audio_page())
        self.pages.addWidget(self._voice_page())
        self.pages.addWidget(self._brain_page())
        self.pages.addWidget(self._tools_page())
        self.pages.addWidget(self._permissions_page())
        self.pages.addWidget(self._vision_page())
        self.pages.addWidget(self._memory_page())
        self.pages.addWidget(self._system_page())
        self.pages.addWidget(self._advanced_page())
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)

    def _general_page(self):
        page, layout = self._page("General", "Estado esencial sin exponer controles críticos.")
        self._value(layout, "general_workspace", "Workspace activo")
        self._value(layout, "general_tools", "Herramientas")
        self._value(layout, "general_brain", "Cerebro")
        self._value(layout, "general_audio", "Entrada de voz")
        self._value(layout, "general_protection", "Protección del sistema")
        layout.addStretch(1)
        return page

    def _files_page(self):
        page, layout = self._page(
            "Archivos y workspace",
            "Las carpetas autorizadas no amplían el acceso a sus directorios padre.",
        )
        self._value(layout, "permanent_workspace", "Workspace permanente")
        row = QHBoxLayout()
        choose = QPushButton("Seleccionar carpeta")
        choose.clicked.connect(lambda: self._choose_workspace(permanent=True))
        remove = QPushButton("Quitar")
        remove.clicked.connect(self._clear_permanent)
        open_button = QPushButton("Abrir carpeta")
        open_button.clicked.connect(lambda: self._open_workspace("permanent"))
        for button in (choose, remove, open_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self._divider(layout)
        self._value(layout, "temporary_workspace", "Workspace temporal")
        row = QHBoxLayout()
        choose = QPushButton("Seleccionar")
        choose.clicked.connect(lambda: self._choose_workspace(permanent=False))
        revoke = QPushButton("Revocar")
        revoke.clicked.connect(self._clear_temporary)
        open_button = QPushButton("Abrir carpeta")
        open_button.clicked.connect(lambda: self._open_workspace("temporary"))
        for button in (choose, revoke, open_button):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self._divider(layout)
        self._value(layout, "active_workspace", "Workspace usado ahora")
        layout.addStretch(1)
        return page

    def _audio_page(self):
        page, layout = self._page(
            "Audio y micrófono",
            "El motor funciona en segundo plano; el diagnóstico se abre solo cuando lo solicitas.",
        )
        self._value(layout, "audio_device", "Micrófono")
        self._value(layout, "audio_stream", "Stream")
        self._value(layout, "audio_stt", "Parakeet")
        self._value(layout, "audio_vad", "VAD")
        self._value(layout, "audio_ptt", "Push-to-Talk")
        self._value(layout, "audio_preroll", "Pre-roll")
        open_diagnostics = QPushButton("Abrir diagnóstico de audio")
        open_diagnostics.clicked.connect(self.open_audio_diagnostics)
        layout.addWidget(open_diagnostics, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return page

    def _voice_page(self):
        page, layout = self._page("Voz", "La identidad y el timbre B4 permanecen bloqueados.")
        self._value(layout, "voice_status", "B4")
        self._value(layout, "voice_backend", "Backend")
        self._value(layout, "voice_output", "Salida")
        note = QLabel("Los ajustes experimentales de timbre no están disponibles desde este panel.")
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _brain_page(self):
        page, layout = self._page("Cerebro", "Información del router activo, sin modelos arbitrarios.")
        self._value(layout, "brain_provider", "Proveedor actual")
        self._value(layout, "brain_model", "Modelo principal")
        self._value(layout, "brain_fallback", "Fallback")
        self._value(layout, "brain_state", "Estado")
        layout.addStretch(1)
        return page

    def _tools_page(self):
        page, layout = self._page("Herramientas", "Capacidades verificadas por el Tool Registry.")
        self._value(layout, "tools_enabled", "Sistema")
        self._value(layout, "tools_count", "Herramientas")
        self._value(layout, "capabilities_count", "Capacidades")
        self._value(layout, "tools_workspace", "Workspace activo")
        self._value(layout, "tools_names", "Familias disponibles")
        self._value(layout, "tools_capabilities", "Vista de capacidades")
        layout.addStretch(1)
        return page

    def _permissions_page(self):
        page, layout = self._page("Permisos", "Las protecciones críticas no pueden desactivarse aquí.")
        self._value(layout, "permission_permanent", "Workspace permanente")
        self._value(layout, "permission_temporary", "Workspace temporal")
        self._value(layout, "permission_scopes", "Scopes de esta sesión")
        policies = QLabel(
            "Lectura y creación: según scope autorizado.\n"
            "Edición: validada localmente.\n"
            "Sobrescritura, mover y eliminar: requieren confirmación.\n"
            "Ejecutar código: confirmación y sandbox aislado."
        )
        policies.setWordWrap(True)
        layout.addWidget(policies)
        core = QLabel("Protección del sistema de Anydes: ACTIVADA")
        core.setStyleSheet("color:#9bd5b7;font-weight:600;")
        layout.addWidget(core)
        layout.addStretch(1)
        return page

    def _vision_page(self):
        page, layout = self._page("Visión y privacidad", "La visión permanece on-demand y sanitizada.")
        self._value(layout, "vision_available", "Visión")
        self._value(layout, "vision_model", "Modelo")
        self._value(layout, "vision_state", "Estado")
        self._value(layout, "vision_privacy", "Privacidad")
        layout.addStretch(1)
        return page

    def _memory_page(self):
        page, layout = self._page("Memoria", "Vista informativa; SELF no se puede editar aquí.")
        self._value(layout, "memory_active", "Memoria")
        self._value(layout, "memory_version", "Contexto")
        self._value(layout, "memory_state", "Estado")
        layout.addStretch(1)
        return page

    def _system_page(self):
        page, layout = self._page("Estado de Anydes", "Lectura directa del runtime actual.")
        for key, title in (
            ("system_groq", "Groq"),
            ("system_fallback", "Qwen8B fallback"),
            ("system_parakeet", "Parakeet"),
            ("system_b4", "B4"),
            ("system_tools", "Tools"),
            ("system_vision", "Vision"),
            ("system_memory", "Memory"),
            ("system_avatar", "Avatar"),
        ):
            self._value(layout, key, title)
        layout.addStretch(1)
        return page

    def _advanced_page(self):
        page, layout = self._page("Avanzado", "Diagnóstico local, colapsado de forma predeterminada.")
        toggle = QPushButton("Mostrar diagnóstico")
        toggle.setCheckable(True)
        toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_details = QLabel("")
        self.advanced_details.setWordWrap(True)
        self.advanced_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.advanced_details.setVisible(False)
        layout.addWidget(self.advanced_details)
        layout.addStretch(1)
        return page

    def _page(self, title: str, hint: str):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(12)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        subtitle = QLabel(hint)
        subtitle.setObjectName("SectionHint")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self._divider(layout)
        scroll.setWidget(container)
        return scroll, layout

    def _value(self, layout, key: str, title: str) -> None:
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight:600;color:#ffffff;")
        value = QLabel("Cargando...")
        value.setObjectName("Value")
        value.setWordWrap(True)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title_label)
        layout.addWidget(value)
        self._labels[key] = value

    @staticmethod
    def _divider(layout) -> None:
        line = QFrame()
        line.setObjectName("Divider")
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

    def showEvent(self, event) -> None:
        self.refresh()
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        workspace = self.settings.snapshot()
        registry_snapshot = self.controller.registry.capability_snapshot()
        health = self.controller.registry.health_status()
        brain = _brain_status()
        audio = _audio_status(self.owner)
        voice = _voice_status(self.owner)
        vision = _vision_status(self.owner)
        memory = _memory_status(self.owner)
        tools_ready = all(item.get("enabled") for item in health.values())

        values = {
            "general_workspace": f"{workspace.active_workspace} ({workspace.active_workspace_kind})",
            "general_tools": "Ready" if tools_ready else "Error",
            "general_brain": brain["state"],
            "general_audio": audio["stt"],
            "general_protection": "ACTIVADA",
            "permanent_workspace": workspace.permanent_workspace or "No configurado",
            "temporary_workspace": workspace.temporary_workspace or "No configurado",
            "active_workspace": f"{workspace.active_workspace} ({workspace.active_workspace_kind})",
            "audio_device": audio["device"],
            "audio_stream": audio["stream"],
            "audio_stt": audio["stt"],
            "audio_vad": audio["vad"],
            "audio_ptt": audio["ptt"],
            "audio_preroll": audio["pre_roll"],
            "voice_status": voice["status"],
            "voice_backend": voice["backend"],
            "voice_output": voice["output"],
            "brain_provider": brain["provider"],
            "brain_model": brain["model"],
            "brain_fallback": brain["fallback"],
            "brain_state": brain["state"],
            "tools_enabled": "Activo" if tools_ready else "Revisar estado",
            "tools_count": str(len(registry_snapshot.tools)),
            "capabilities_count": str(len(registry_snapshot.available_capabilities)),
            "tools_workspace": workspace.active_workspace,
            "tools_names": ", ".join(
                str(item["tool_id"]).removeprefix("official.").title()
                for item in registry_snapshot.tools
            ),
            "tools_capabilities": ", ".join(registry_snapshot.available_capabilities),
            "permission_permanent": workspace.permanent_workspace or "No configurado",
            "permission_temporary": workspace.temporary_workspace or "No configurado",
            "permission_scopes": ", ".join(workspace.authorized_external_scopes) or "Ninguno adicional",
            "vision_available": vision["available"],
            "vision_model": vision["model"],
            "vision_state": vision["state"],
            "vision_privacy": vision["privacy"],
            "memory_active": memory["active"],
            "memory_version": memory["version"],
            "memory_state": memory["state"],
            "system_groq": brain["state"],
            "system_fallback": "Ready" if brain["local_ready"] else "Disponible on-demand",
            "system_parakeet": audio["stt"],
            "system_b4": voice["status"],
            "system_tools": "Ready" if tools_ready else "Error",
            "system_vision": vision["state"],
            "system_memory": memory["state"],
            "system_avatar": "Ready" if getattr(self.owner, "avatar", None) is not None else "Unavailable",
        }
        for key, value in values.items():
            self._labels[key].setText(str(value))
        if self._advanced_visible:
            self.advanced_details.setText(_advanced_text(
                workspace,
                health,
                brain,
                audio,
                voice,
            ))

    def open_audio_diagnostics(self) -> None:
        panel = getattr(self.owner, "input_panel", None)
        if panel is None:
            QMessageBox.warning(self, "Audio", "El panel de audio no está disponible.")
            return
        opener = getattr(panel, "show_from_settings", None)
        if callable(opener):
            opener()
        else:
            panel.show()
            panel.raise_()
            panel.activateWindow()

    def _choose_workspace(self, *, permanent: bool) -> None:
        snapshot = self.settings.snapshot()
        start = snapshot.permanent_workspace if permanent else snapshot.temporary_workspace
        start = start or snapshot.active_workspace
        selected = QFileDialog.getExistingDirectory(self, "Seleccionar workspace", start)
        if not selected:
            return
        try:
            if permanent:
                self.settings.set_permanent_workspace(selected)
            else:
                self.settings.set_temporary_workspace(selected)
        except WorkspaceValidationError as exc:
            QMessageBox.warning(self, "Carpeta no autorizada", _workspace_error(str(exc)))
        self.refresh()

    def _clear_permanent(self) -> None:
        self.settings.clear_permanent_workspace()
        self.refresh()

    def _clear_temporary(self) -> None:
        self.settings.clear_temporary_workspace()
        self.refresh()

    def _open_workspace(self, kind: str) -> None:
        snapshot = self.settings.snapshot()
        path = snapshot.permanent_workspace if kind == "permanent" else snapshot.temporary_workspace
        if path and Path(path).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced_visible = bool(checked)
        self.advanced_details.setVisible(self._advanced_visible)
        sender = self.sender()
        if isinstance(sender, QPushButton):
            sender.setText("Ocultar diagnóstico" if checked else "Mostrar diagnóstico")
        self.refresh()


def build_settings_button(parent, callback) -> QToolButton:
    button = QToolButton(parent)
    button.setText("⚙")
    button.setToolTip("Configuración")
    button.setAccessibleName("Abrir configuración de Anydes")
    button.setFixedSize(36, 34)
    button.setStyleSheet(
        "QToolButton{background:rgba(43,39,48,210);color:#f3eef3;border:1px solid "
        "rgba(126,105,132,180);border-radius:4px;font-size:18px;}"
        "QToolButton:hover{background:rgba(76,62,82,230);}"
    )
    button.clicked.connect(callback)
    return button


def _brain_status() -> dict[str, object]:
    try:
        from anydes.hybrid.router import router_snapshots

        snapshots = router_snapshots()
    except Exception:
        snapshots = []
    configured_model = os.getenv("PRIMARY_GROQ_MODEL", "openai/gpt-oss-120b")
    if not snapshots:
        return {
            "provider": "Groq",
            "model": configured_model,
            "fallback": "Qwen3:8B",
            "state": "Loading",
            "local_ready": False,
        }
    current = snapshots[-1]
    state = str(current.get("state") or "Unavailable")
    label = {
        "GROQ_HEALTHY": "Online",
        "GROQ_RECOVERED": "Recovering",
        "GROQ_DEGRADED": "Degraded",
        "PRELOADING_QWEN8B": "Degraded / loading fallback",
        "LOCAL_FALLBACK_8B": "Local fallback",
        "PROBING_GROQ": "Recovering",
    }.get(state, state)
    return {
        "provider": "Groq" if state != "LOCAL_FALLBACK_8B" else "Local",
        "model": str(current.get("primary_model") or configured_model),
        "fallback": str(current.get("local_fallback_model") or "Qwen3:8B"),
        "state": label,
        "local_ready": bool(current.get("local_ready")),
    }


def _audio_status(owner) -> dict[str, str]:
    panel = getattr(owner, "input_panel", None)
    if panel is None:
        return {
            "device": "Unavailable",
            "stream": "Unavailable",
            "stt": "Unavailable",
            "vad": "Unavailable",
            "ptt": "Unavailable",
            "pre_roll": "Unavailable",
        }
    device = "No seleccionado"
    try:
        device = panel.device.currentText() or device
    except Exception:
        pass
    engine = getattr(panel, "engine", None)
    recognizer = getattr(engine, "recognizer", None)
    vad = getattr(panel, "vad", None)
    ptt = getattr(panel, "ptt", None)
    pre_roll = getattr(panel, "pre_roll", None)
    return {
        "device": device,
        "stream": str(getattr(panel, "stream_state", "Cerrado")),
        "stt": "Ready" if recognizer is not None else "Ready on-demand",
        "vad": "Activo" if vad is not None and vad.isChecked() else "Inactivo",
        "ptt": "Ready" if ptt is not None and ptt.isEnabled() else "Loading",
        "pre_roll": f"{pre_roll.currentData()} ms" if pre_roll is not None else "400 ms",
    }


def _voice_status(owner) -> dict[str, str]:
    manager = getattr(owner, "voice_manager", None)
    if manager is None:
        return {"status": "Unavailable", "backend": "B4", "output": "Sistema predeterminado"}
    warmup = getattr(manager, "_warmup_thread", None)
    status = "Loading" if warmup is not None and warmup.is_alive() else "Ready"
    engine = getattr(manager, "engine", None)
    backend = type(engine).__name__ if engine is not None else "B4 / qwen3-tts-faster"
    return {"status": status, "backend": backend, "output": "Sistema predeterminado"}


def _vision_status(owner) -> dict[str, str]:
    config = getattr(getattr(owner, "assistant", None), "config", None)
    available = bool(getattr(config, "vision_enabled", False))
    session = getattr(owner, "vision_session", None)
    try:
        from anydes.vision.state import get_screen_state

        watching = get_screen_state().is_watching()
    except Exception:
        watching = False
    return {
        "available": "Disponible" if available else "Unavailable",
        "model": "Moondream on-demand" if available else "No activo",
        "state": "Observando" if watching else "Ready on-demand" if session is not None else "Inactivo",
        "privacy": "Sanitización activa",
    }


def _memory_status(owner) -> dict[str, str]:
    assistant = getattr(owner, "assistant", None)
    service = getattr(assistant, "_realtime_service", None)
    continuity = getattr(getattr(service, "session", None), "continuity_v2", None)
    return {
        "active": "Activa" if assistant is not None else "Unavailable",
        "version": "Memory/Context v2" if continuity is not None else "Memoria base",
        "state": "Ready" if assistant is not None else "Unavailable",
    }


def _advanced_text(workspace, health, brain, audio, voice) -> str:
    health_rows = ", ".join(
        f"{tool_id.removeprefix('official.')}: {item.get('status')}"
        for tool_id, item in sorted(health.items())
    )
    return (
        f"Workspace activo: {workspace.active_workspace}\n"
        f"Workspace interno: {workspace.internal_workspace}\n"
        f"Scopes de sesión: {', '.join(workspace.authorized_external_scopes) or 'ninguno'}\n"
        f"Router: {brain['state']} | {brain['model']} | fallback {brain['fallback']}\n"
        f"STT: {audio['stt']} | stream {audio['stream']}\n"
        f"B4: {voice['status']} | {voice['backend']}\n"
        f"Registry health: {health_rows}\n"
        "Core protection: ACTIVADA"
    )


def _workspace_error(code: str) -> str:
    if "PROTECTED" in code:
        return "Esa carpeta pertenece al sistema protegido de Anydes."
    if "REPARSE" in code:
        return "No se permiten enlaces o redirecciones de carpeta inesperadas."
    if "absolute" in code:
        return "Selecciona una ruta absoluta válida."
    if "unavailable" in code or "directory" in code:
        return "La carpeta no existe o no está disponible."
    return "Esa carpeta no puede autorizarse de forma segura."
