"""Staging presentation only. One existing window owns one existing core."""
from dataclasses import replace
import json
from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QFrame, QStackedLayout, QStyle)
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor
from PySide6.QtWebEngineWidgets import QWebEngineView

from assets import ShellAssetServer, ROOT, LAB
from policy import RendererKind, status_label, load_selection, save_selection, can_switch

HERE = Path(__file__).resolve().parent
STYLE = '''
QWidget { background:#0b0b10; color:#e9e5ef; font-family:'Segoe UI'; font-size:13px; }
QDialog { background:#0b0b10; }
QLabel#Brand { font-size:28px; font-weight:600; color:#f3e9f4; }
QLabel#Subtle { color:#a29cab; font-size:12px; }
QLabel#Activity { color:#d5bddc; font-size:12px; }
QPushButton { background:#18151f; border:1px solid #39303f; border-radius:6px; padding:10px 16px; }
QPushButton:hover { background:#2a2032; border-color:#b47baf; }
QPushButton:focus { border-color:#e1a3d3; }
QPushButton:disabled { color:#625b6b; border-color:#242129; }
QPushButton#PresenceCard { background:#111016; border:1px solid #47354c; padding:10px; border-radius:8px; }
QPushButton#PresenceCard:hover { background:#201725; border-color:#da9acb; }
QCheckBox { spacing:10px; color:#b9b0c2; }
QCheckBox::indicator { width:17px; height:17px; }
QFrame#ChatPanel { border-top:1px solid #302735; border-radius:0; background:#0f0e14; }
QTextEdit { background:#0f0e14; border:0; color:#eee7f1; font-size:14px; padding:8px; }
QLineEdit { background:#15121b; border:1px solid #3d3146; border-radius:6px; padding:13px; color:#f0e9f4; }
QFrame#InputDock { background:#0b0b10; border:0; }
QPushButton#VoiceButton,QPushButton#SendButton { min-width:26px; min-height:22px; }
'''


class PresenceChoice(QDialog):
    def __init__(self, parent=None, current=None):
        super().__init__(parent)
        self.kind = None
        self.setWindowTitle('Anydes | Elegir presencia')
        self.setStyleSheet(STYLE)
        self.setMinimumSize(760, 580)
        self.resize(920, 650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(38, 30, 38, 28)
        layout.setSpacing(16)
        brand = QLabel('ANYDES'); brand.setObjectName('Brand'); brand.setFixedHeight(44)
        layout.addWidget(brand)
        heading = QLabel('Elige tu presencia'); heading.setFixedHeight(28)
        layout.addWidget(heading)
        row = QHBoxLayout(); row.setSpacing(20)
        previews = {
            RendererKind.HOLOGRAM: LAB / 'h1_3c/evidence/qa/THINKING.png',
            RendererKind.AVATAR_2_5D: ROOT / 'programa_principal/anydes/assets/avatar_25d_hq/assets/anydes_master_alpha_v1.png',
        }
        for kind, title in [(RendererKind.HOLOGRAM, 'HOLOGRAMA'), (RendererKind.AVATAR_2_5D, 'AVATAR 2.5D')]:
            button = QPushButton(); button.setObjectName('PresenceCard')
            button.setAccessibleName(title); button.setMinimumSize(310, 340)
            content = QVBoxLayout(button); content.setContentsMargins(14, 14, 14, 20)
            image = QLabel(); image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = QPixmap(str(previews[kind]))
            if not pixmap.isNull():
                # Crop a scene preview, not the laboratory's controls or labels.
                if kind == RendererKind.HOLOGRAM:
                    pixmap = pixmap.copy(int(pixmap.width()*.18), int(pixmap.height()*.12),
                                         int(pixmap.width()*.64), int(pixmap.height()*.63))
                image.setPixmap(pixmap.scaled(300, 270, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
            content.addWidget(image, 1)
            text = QLabel(title); text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content.addWidget(text)
            for child in (image, text):
                child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            button.clicked.connect(lambda checked=False, value=kind: self.select(value))
            row.addWidget(button)
        layout.addLayout(row, 1)
        self.remember = QCheckBox('Recordar mi elecci\u00f3n')
        self.remember.setChecked(current is not None)
        layout.addWidget(self.remember)

    def select(self, kind):
        self.kind = kind
        self.accept()


class LocalRequests(QWebEngineUrlRequestInterceptor):
    def __init__(self, port, parent=None):
        super().__init__(parent)
        self.ports = {port, 8781}

    def interceptRequest(self, info):
        url = info.requestUrl()
        allowed = url.scheme() in {'data', 'blob', 'about'} or (
            url.scheme() == 'http' and url.host() == '127.0.0.1' and url.port() in self.ports)
        if not allowed:
            info.block(True)


class PresenceSurface(QWidget):
    """Common renderer facade, retaining the legacy avatar's playback methods."""
    def __init__(self, kind, legacy_factory, record, parent=None, **legacy_options):
        super().__init__(parent)
        self.kind, self.record = RendererKind(kind), record
        self.legacy = self.web = self.server = None
        self.failed = False
        self.stopped = False
        self.frame = dict(state='IDLE', connected=False)
        self.setMinimumHeight(280)
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.neutral = QLabel('Anydes'); self.neutral.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.neutral)
        try:
            if self.kind == RendererKind.HOLOGRAM:
                self.server = ShellAssetServer()
                self.web = QWebEngineView(self)
                self.profile = QWebEngineProfile(self)
                self.web.setPage(QWebEnginePage(self.profile, self.web))
                self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
                self.web.page().setBackgroundColor(QColor('#09090f'))
                self.interceptor = LocalRequests(self.server.port, self.web)
                self.profile.setUrlRequestInterceptor(self.interceptor)
                self.web.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                self.web.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
                self.web.page().renderProcessTerminated.connect(lambda *args: self.fail('WEB_PROCESS_TERMINATED'))
                self.web.loadFinished.connect(lambda ok: None if ok else self.fail('LOAD_FAILED'))
                self.web.setUrl(QUrl(self.server.url))
                self.stack.addWidget(self.web); self.stack.setCurrentWidget(self.web)
            else:
                self.legacy = legacy_factory(**legacy_options)
                self.legacy.setMinimumHeight(280)
                self.stack.addWidget(self.legacy); self.stack.setCurrentWidget(self.legacy)
                view = getattr(self.legacy, '_vrm_view', None)
                if view:
                    view.render_failed.connect(lambda *args: self.fail('AVATAR_RENDER_FAILED'))
                    view.page().renderProcessTerminated.connect(lambda *args: self.fail('WEB_PROCESS_TERMINATED'))
            self.record('renderer_initialized', renderer=self.kind.value)
        except (OSError, RuntimeError, ValueError) as error:
            self.fail(type(error).__name__)
        self.ready = False
        self.started = time.monotonic()
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(1000)
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start()

    def check_health(self):
        if self.failed:
            return
        if self.web:
            self.web.page().runJavaScript(
                'JSON.stringify(window.anydesPresence ? {frames:window.anydesPresence.snapshot().frames,webgl:window.anydesPresence.snapshot().webgl,connected:window.anydesPresence.snapshot().connectedToAnydes}:null)',
                self.web_health)
        elif self.legacy:
            view = getattr(self.legacy, '_vrm_view', None)
            if view and view.render_ready and not self.ready:
                self.ready = True
                self.record('renderer_ready', renderer=self.kind.value)

    def web_health(self, value):
        if self.failed:
            return
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = None
        if isinstance(value, dict) and value.get('webgl') and value.get('frames', 0) > 2:
            if not self.ready:
                self.ready = True
                self.record('renderer_ready', renderer=self.kind.value, sse_connected=value.get('connected'))
        elif time.monotonic()-self.started > 20:
            self.fail('RENDERER_NOT_READY')

    def fail(self, code):
        if self.stopped:
            return
        self.failed = True
        self.stack.setCurrentWidget(self.neutral)
        self.record('renderer_failed', renderer=self.kind.value, code=code)

    def set_presence(self, frame, mode):
        previous = self.frame.get('state')
        self.frame = dict(frame)
        self.working_mode = mode
        # Hologram consumes the same adapter through SSE. Legacy appearance and
        # lip-sync retain their native behavior; no new animation is invented.
        if previous != frame['state'] and self.legacy and not self.failed and frame['state'] in {'IDLE', 'LISTENING', 'THINKING'}:
            state = self.legacy._state
            gesture = {'IDLE': 'idle', 'LISTENING': 'listening', 'THINKING': 'thinking'}[frame['state']]
            self.set_state(replace(state, gesture=gesture, status=status_label(frame['state'])))

    def set_state(self, state):
        self.legacy_call('set_state', state)

    def legacy_call(self, method, *args, **kwargs):
        if self.legacy and not self.failed:
            try:
                getattr(self.legacy, method)(*args, **kwargs)
            except (RuntimeError, ValueError, OSError) as error:
                self.fail(f'{method}:{type(error).__name__}')

    def begin_turn(self, turn_id):
        self.legacy_call('begin_turn', turn_id)

    def stop_speaking(self):
        self.legacy_call('stop_speaking')

    def start_lip_sync(self, *args, **kwargs):
        self.legacy_call('start_lip_sync', *args, **kwargs)

    def shutdown(self):
        if self.stopped:
            return
        self.stopped = True
        self.health_timer.stop()
        if self.web:
            self.web.stop()
            self.web.setUrl(QUrl('about:blank'))
        if self.server:
            self.server.close()
        if self.legacy:
            for timer in self.legacy.findChildren(QTimer):
                timer.stop()
            view = getattr(self.legacy, '_vrm_view', None)
            if view:
                view.close()


def install_shell(module, session, service):
    base, legacy_factory = module.AnydesPySideWindow, module.AvatarStage
    diagnostic_write_failures = 0

    def record(event, **fields):
        nonlocal diagnostic_write_failures
        try:
            with (Path(session)/'shell.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event=event, monotonic=time.monotonic(), **fields))+'\n')
        except OSError:
            diagnostic_write_failures += 1

    def persist(kind, remember):
        try:
            save_selection(session, kind, remember)
        except OSError as error:
            record('ui_selection_not_saved', error_type=type(error).__name__)

    class ProfessionalWindow(base):
        def __init__(self):
            remembered = load_selection(Path(session).parent)
            if remembered is None:
                picker = PresenceChoice()
                if picker.exec() != QDialog.DialogCode.Accepted:
                    raise SystemExit(0)
                chosen = picker.kind
                persist(chosen, picker.remember.isChecked())
            else:
                chosen = remembered
            self.presence_kind = chosen
            super().__init__()
            self.setWindowTitle('Anydes | Professional · Staging')
            self.setMinimumSize(820, 720)
            self.resize(1120, 900)
            self.presence_timer = QTimer(self)
            self.presence_timer.timeout.connect(self.update_presence)
            self.presence_timer.start(120)
            self.last_presence = None
            self.record_timer = QTimer(self)
            self.record_timer.timeout.connect(self.sample_resources)
            self.record_timer.start(1500)
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent()
            self.update_presence()
            record('shell_opened', renderer=chosen.value, core_count=1,
                   assistant_id=hex(id(self.assistant)), voice_id=hex(id(self.voice_manager)))

        def _build(self):
            super()._build()
            old = self.takeCentralWidget()
            # Reparent the existing controls, not their controllers or state.
            root = QWidget(); root.setObjectName('ProfessionalShell')
            layout = QVBoxLayout(root); layout.setContentsMargins(24, 16, 24, 16); layout.setSpacing(10)
            top = QHBoxLayout()
            brand = QLabel('ANYDES'); brand.setObjectName('Brand'); top.addWidget(brand)
            top.addStretch(1)
            self.activity = QLabel('En reposo'); self.activity.setObjectName('Activity'); top.addWidget(self.activity)
            change = QPushButton(); change.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
            change.setToolTip('Cambiar presencia'); change.setAccessibleName('Cambiar presencia')
            change.clicked.connect(self.change_presence); top.addWidget(change)
            for button in (self.stop_sensors_button, self.share_screen_button, self.inner_button):
                top.addWidget(button)
            layout.addLayout(top)
            self.presence_layout = layout
            layout.addWidget(self.avatar, 6)
            self.chat.text.setMinimumHeight(140); self.chat.text.setMaximumHeight(230)
            self.chat.setGraphicsEffect(None)
            layout.addWidget(self.chat, 2)
            layout.addWidget(self.input_dock)
            self.sensor_indicator.setParent(root); self.sensor_indicator.hide()
            self.status.setParent(root); self.status.hide()
            self.inner_world.setParent(root); self.inner_world.hide()
            self.setCentralWidget(root)
            old.deleteLater()
            self.setStyleSheet(STYLE)

        def update_presence(self):
            frame = dict(service.latest)
            mode = 'NONE'
            if frame['state'] == 'WORKING':
                mode = 'VERIFYING' if frame['source'] == 'tools.verification' else next(reversed(tuple(service.modes.values())), 'NONE')
            self.activity.setText(status_label(frame['state'], mode, frame['connected']))
            self.avatar.set_presence(frame, mode)
            key = (frame['state'], mode, frame['connected'])
            if key != self.last_presence:
                record('presence_displayed', renderer=self.presence_kind.value, state=key[0],
                       working_mode=mode, connected=key[2], sequence=frame['sequence'])
                self.last_presence = key

        def change_presence(self):
            running = bool(self._worker and self._worker.isRunning())
            if not can_switch(service.latest['state'], service.latest['connected'], running):
                self.activity.setText('Disponible al terminar la respuesta')
                return
            picker = PresenceChoice(self, self.presence_kind)
            if picker.exec() != QDialog.DialogCode.Accepted:
                return
            persist(picker.kind, picker.remember.isChecked())
            if picker.kind == self.presence_kind:
                return
            previous = self.avatar
            self.presence_kind = picker.kind
            self.avatar = surface()
            self.presence_layout.replaceWidget(previous, self.avatar)
            previous.shutdown(); previous.hide(); previous.deleteLater()
            self.last_presence = None
            self.update_presence()
            record('renderer_switched', renderer=self.presence_kind.value,
                   assistant_id=hex(id(self.assistant)), voice_id=hex(id(self.voice_manager)))

        def sample_resources(self):
            import psutil
            processes = [self.process] + self.process.children(recursive=True)
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            record('resources', state=service.latest['state'], host_cpu_percent=self.process.cpu_percent(),
                   tree_rss_mb=round(rss/1048576, 1), process_count=len(processes))

        def closeEvent(self, event):
            super().closeEvent(event)
            if event.isAccepted():
                self.presence_timer.stop(); self.record_timer.stop()
                self.avatar.shutdown()
                record('shell_closed', diagnostic_write_failures=diagnostic_write_failures)

    def surface(parent=None, **options):
        # The base _build factory is called only while this single window exists.
        window = next((w for w in __import__('PySide6.QtWidgets', fromlist=['QApplication']).QApplication.topLevelWidgets()
                       if isinstance(w, ProfessionalWindow)), None)
        kind = window.presence_kind if window else RendererKind.HOLOGRAM
        return PresenceSurface(kind, legacy_factory, record, parent, **options)

    module.AvatarStage = surface
    module.AnydesPySideWindow = ProfessionalWindow
