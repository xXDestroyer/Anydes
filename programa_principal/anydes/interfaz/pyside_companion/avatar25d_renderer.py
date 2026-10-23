from __future__ import annotations

import json
import mimetypes
import os
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[4]
AVATAR_ROOT = (
    PROJECT_ROOT / "programa_principal" / "anydes" / "assets" / "avatar_25d_hq"
).resolve()
THREE_ROOT = (
    PROJECT_ROOT
    / "programa_principal"
    / "anydes"
    / "interfaz"
    / "pyside_companion"
    / "vrm_stage"
    / "vendor"
).resolve()


class Avatar25DAssetServer:
    """Read-only loopback server for the official 2.5D avatar assets."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="anydes-25d-assets",
            daemon=True,
        )
        self._stopped = False

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=2.0)
        self._server.server_close()

    def _handler(self):
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):  # noqa: N802
                self._serve(include_body=False)

            def do_GET(self):  # noqa: N802
                self._serve(include_body=True)

            def log_message(self, format, *args):
                _ = (format, args)

            def _serve(self, *, include_body: bool) -> None:
                request_path = unquote(urlsplit(self.path).path)
                requests.append(request_path[:240])
                target: Path | None = None
                avatar_prefix = "/programa_principal/anydes/assets/avatar_25d_hq/"
                vendor_prefix = "/programa_principal/anydes/interfaz/pyside_companion/vrm_stage/vendor/"
                if request_path.startswith(avatar_prefix):
                    target = (AVATAR_ROOT / request_path.removeprefix(avatar_prefix)).resolve(strict=False)
                    if AVATAR_ROOT not in target.parents:
                        target = None
                elif request_path.startswith(vendor_prefix):
                    target = (THREE_ROOT / request_path.removeprefix(vendor_prefix)).resolve(strict=False)
                    if THREE_ROOT not in target.parents:
                        target = None
                if target is None or not target.is_file():
                    self.send_error(404)
                    return
                payload = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)

        return Handler


try:
    from PySide6.QtCore import QTimer, QUrl, Qt, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    WEBENGINE_25D_AVAILABLE = False
    Experimental25DAvatarView = None
else:
    WEBENGINE_25D_AVAILABLE = True

    class _DiagnosticPage(QWebEnginePage):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.messages: list[str] = []

        def javaScriptConsoleMessage(self, level, message, line_number, source_id):  # noqa: N802
            _ = (level, source_id)
            self.messages.append(f"line {line_number}: {message}"[:500])
            self.messages[:] = self.messages[-30:]


    class Experimental25DAvatarView(QWebEngineView):
        """Official HQ 2.5D avatar surface."""

        ready_changed = Signal(bool)
        render_failed = Signal(str)

        def __init__(self, _model_path: Path | None = None, parent=None) -> None:
            super().__init__(parent)
            self.render_ready = False
            self.render_error = ""
            self.page_loaded: bool | None = None
            self.last_status: object = None
            self._pending_state: dict[str, Any] = {}
            self._pending_environment: dict[str, Any] = {}
            self._pending_lip_sync: dict[str, Any] = {}
            self._turn_id = 0
            self._poll_attempts = 0
            self._asset_server = Avatar25DAssetServer()
            self._asset_server.start()
            self._diagnostic_page = _DiagnosticPage(self)
            self.setPage(self._diagnostic_page)
            self.destroyed.connect(self._stop_asset_server)
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.page().setBackgroundColor(QColor(0, 0, 0, 0))
            settings = self.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
            self.loadFinished.connect(self._on_page_loaded)
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(180)
            self._poll_timer.timeout.connect(self._poll_status)
            framing = os.environ.get("ANYDES_25D_FRAMING", "balanced").strip().lower()
            if framing not in {"control", "gentle", "balanced", "close", "face_closeup"}:
                framing = "balanced"
            face_hq = os.environ.get("ANYDES_25D_FACE_HQ", "control").strip().lower()
            if face_hq not in {"control", "moderate", "strong", "hybrid", "hybrid_refined"}:
                face_hq = "control"
            detail_mode = os.environ.get("ANYDES_25D_DETAIL_MODE", "room").strip().lower()
            if detail_mode not in {"room", "closeup"}:
                detail_mode = "room"
            mouth_profile = os.environ.get("ANYDES_25D_MOUTH_PROFILE", "control").strip().lower()
            if mouth_profile not in {"control", "moderate", "visible", "moderate_95", "moderate_90"}:
                mouth_profile = "control"
            self.setUrl(QUrl(
                f"{self._asset_server.base_url}/programa_principal/anydes/assets/avatar_25d_hq/stage/index.html"
                f"?profile=full&integrated=1&framing={framing}&facehq={face_hq}&detail={detail_mode}&mouth={mouth_profile}"
            ))

        @property
        def diagnostic_messages(self) -> tuple[str, ...]:
            return tuple(self._diagnostic_page.messages)

        @property
        def diagnostic_state(self) -> dict[str, Any]:
            return {
                "page_loaded": self.page_loaded,
                "last_status": self.last_status,
                "requests": list(self._asset_server.requests[-30:]),
            }

        def set_avatar_state(self, state: Any) -> None:
            self._pending_state = _json_mapping(state)
            self._pending_state["turn_id"] = self._turn_id
            if self.render_ready:
                self._call("applyState", self._pending_state)

        def set_environment(self, environment: Any) -> None:
            self._pending_environment = _json_mapping(environment)
            if self.render_ready:
                self._call("setEnvironment", self._pending_environment)

        def set_pointer(self, x: float, y: float) -> None:
            if self.render_ready:
                self.page().runJavaScript(
                    f"window.Anydes25DAvatar?.setPointer({float(x):.6f},{-float(y):.6f});"
                )

        def set_camera_mode(self, _mode: str) -> None:
            return

        def set_rest_pose_enabled(self, _enabled: bool) -> None:
            return

        def stop_speaking(self) -> None:
            self._pending_lip_sync = {}
            if self.render_ready:
                self.page().runJavaScript("window.Anydes25DAvatar?.stopSpeaking();")

        def begin_turn(self, turn_id: int) -> None:
            self._turn_id = int(turn_id)
            self._pending_lip_sync = {}
            self._pending_state = {}
            if self.render_ready:
                self._call("beginTurn", {"turn_id": self._turn_id})

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
            self._pending_lip_sync = {
                "turn_id": self._turn_id,
                "timeline": timeline,
                "duration_ms": max(0.0, float(duration_ms)),
                "body_duration_ms": max(0.0, float(duration_ms if body_duration_ms is None else body_duration_ms)),
                "offset_ms": max(0.0, float(offset_ms)),
                "start_delay_ms": max(0.0, float(start_delay_ms)),
                "playback_id": int(playback_id),
            }
            if self.render_ready:
                self._call("startLipSyncTimeline", self._pending_lip_sync)

        def closeEvent(self, event) -> None:  # noqa: N802
            self._poll_timer.stop()
            self._stop_asset_server()
            super().closeEvent(event)

        def _stop_asset_server(self, *_args) -> None:
            self._asset_server.stop()

        def _on_page_loaded(self, successful: bool) -> None:
            self.page_loaded = successful
            if not successful:
                self._fail("stage_page_load_failed")
                return
            self._poll_timer.start()

        def _poll_status(self) -> None:
            self._poll_attempts += 1
            self.page().runJavaScript(
                "JSON.stringify(window.Anydes25DAvatar ? window.Anydes25DAvatar.status() : null);",
                self._handle_status,
            )
            if self._poll_attempts > 120 and not self.render_ready:
                self._fail("avatar_25d_renderer_timeout")

        def _handle_status(self, status: object) -> None:
            if isinstance(status, str):
                try:
                    status = json.loads(status)
                except json.JSONDecodeError:
                    pass
            self.last_status = status
            if self.render_ready or not isinstance(status, dict):
                return
            if not bool(status.get("ready")) or int(status.get("frames") or 0) < 2:
                return
            self._poll_timer.stop()
            self.render_ready = True
            self.ready_changed.emit(True)
            self._call("beginTurn", {"turn_id": self._turn_id})
            if self._pending_state:
                self._call("applyState", self._pending_state)
            if self._pending_environment:
                self._call("setEnvironment", self._pending_environment)
            if self._pending_lip_sync:
                self._call("startLipSyncTimeline", self._pending_lip_sync)

        def _call(self, method: str, payload: dict[str, Any]) -> None:
            serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            self.page().runJavaScript(f"window.Anydes25DAvatar?.{method}({serialized});")

        def _fail(self, code: str) -> None:
            self._poll_timer.stop()
            self.render_ready = False
            self.render_error = code
            self.ready_changed.emit(False)
            self.render_failed.emit(code)


def _json_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return {}
