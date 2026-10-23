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

from anydes.avatar.vrm import inspect_vrm


STAGE_DIR = Path(__file__).resolve().parent / "vrm_stage"
REST_POSE_PROFILE_NAME = "Anydes_Rest_HandsInHoodiePocket.json"


class VrmAssetServer:
    """Read-only loopback server exposing only renderer assets and one VRM."""

    def __init__(self, model_path: Path) -> None:
        info = inspect_vrm(model_path)
        if not info.valid:
            raise ValueError("A valid local VRM model is required.")
        self.model_path = info.path
        pose_profile = self.model_path.with_name(REST_POSE_PROFILE_NAME)
        self.pose_profile_path = pose_profile if pose_profile.is_file() else None
        self.requests: list[str] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="anydes-vrm-assets",
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
        model_path = self.model_path
        pose_profile_path = self.pose_profile_path
        stage_root = STAGE_DIR.resolve()
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
                if request_path == "/model.vrm":
                    target = model_path
                elif request_path == "/pose.json" and pose_profile_path is not None:
                    target = pose_profile_path
                elif request_path.startswith("/stage/"):
                    relative = request_path.removeprefix("/stage/") or "index.html"
                    target = (stage_root / relative).resolve(strict=False)
                    if stage_root not in target.parents:
                        self.send_error(404)
                        return
                else:
                    self.send_error(404)
                    return
                if not target.is_file():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                try:
                    payload = target.read_bytes()
                except OSError:
                    self.send_error(500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if include_body:
                    self.wfile.write(payload)

        return Handler

try:
    from PySide6.QtCore import QTimer, QUrl, QUrlQuery, Qt, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    WEBENGINE_AVAILABLE = False
    VrmAvatarView = None
else:
    WEBENGINE_AVAILABLE = True

    class _DiagnosticPage(QWebEnginePage):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.messages: list[str] = []

        def javaScriptConsoleMessage(self, level, message, line_number, source_id):  # noqa: N802
            _ = (level, source_id)
            self.messages.append(f"line {line_number}: {message}"[:500])
            if len(self.messages) > 20:
                self.messages.pop(0)

    class VrmAvatarView(QWebEngineView):
        """Transparent Three.js surface that renders one validated local VRM."""

        ready_changed = Signal(bool)
        render_failed = Signal(str)

        def __init__(self, model_path: Path, parent=None) -> None:
            super().__init__(parent)
            info = inspect_vrm(model_path)
            if not info.valid:
                raise ValueError("A valid local VRM model is required.")
            self.model_path = info.path
            self.render_ready = False
            self.render_error = ""
            self._pending_state: dict[str, Any] = {}
            self._pending_environment: dict[str, Any] = {}
            self._pending_camera_mode = "environment"
            self._pending_rest_pose_enabled = True
            self._poll_attempts = 0
            self.page_loaded: bool | None = None
            self.last_status: object = None
            self._asset_server = VrmAssetServer(self.model_path)
            self._asset_server.start()
            self._diagnostic_page = _DiagnosticPage(self)
            self.setPage(self._diagnostic_page)
            self.destroyed.connect(self._stop_asset_server)
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.page().setBackgroundColor(QColor(0, 0, 0, 0))
            settings = self.settings()
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
                True,
            )
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled,
                True,
            )
            self.loadFinished.connect(self._on_page_loaded)
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(180)
            self._poll_timer.timeout.connect(self._poll_status)
            self.setUrl(renderer_url(self.model_path, base_url=self._asset_server.base_url))

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
            if self.render_ready:
                self._call("applyState", self._pending_state)

        def set_environment(self, environment: Any) -> None:
            self._pending_environment = _json_mapping(environment)
            if self.render_ready:
                self._call("applyEnvironment", self._pending_environment)

        def set_pointer(self, x: float, y: float) -> None:
            if not self.render_ready:
                return
            script = f"window.AnydesAvatar?.setPointer({float(x):.6f}, {float(y):.6f});"
            self.page().runJavaScript(script)

        def set_camera_mode(self, mode: str) -> None:
            selected = str(mode).strip().lower()
            if selected not in {"environment", "conversation", "face_review"}:
                raise ValueError(
                    "camera mode must be environment, conversation or face_review"
                )
            self._pending_camera_mode = selected
            if self.render_ready:
                self.page().runJavaScript(
                    f"window.AnydesAvatar?.setCameraMode({json.dumps(selected)});"
                )

        def set_rest_pose_enabled(self, enabled: bool) -> None:
            self._pending_rest_pose_enabled = bool(enabled)
            if self.render_ready:
                value = "true" if self._pending_rest_pose_enabled else "false"
                self.page().runJavaScript(
                    f"window.AnydesAvatar?.setRestPoseEnabled({value});"
                )

        def stop_speaking(self) -> None:
            if self.render_ready:
                self.page().runJavaScript("window.AnydesAvatar?.stopSpeaking();")

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
            payload = {
                "timeline": timeline,
                "duration_ms": max(0.0, float(duration_ms)),
                "body_duration_ms": max(
                    0.0,
                    float(duration_ms if body_duration_ms is None else body_duration_ms),
                ),
                "offset_ms": max(0.0, float(offset_ms)),
                "start_delay_ms": max(0.0, float(start_delay_ms)),
                "playback_id": int(playback_id),
            }
            if self.render_ready:
                self._call("startLipSyncTimeline", payload)

        def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
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
            self._poll_attempts = 0
            self._poll_timer.start()

        def _poll_status(self) -> None:
            self._poll_attempts += 1
            self.page().runJavaScript(
                "JSON.stringify(window.AnydesAvatar ? window.AnydesAvatar.status() : null);",
                self._handle_status,
            )
            if self._poll_attempts > 100 and not self.render_ready:
                self._fail("vrm_renderer_timeout")

        def _handle_status(self, status: object) -> None:
            if isinstance(status, str):
                try:
                    status = json.loads(status)
                except json.JSONDecodeError:
                    pass
            self.last_status = status
            if self.render_ready:
                return
            if not isinstance(status, dict):
                return
            error = str(status.get("error") or "")
            if error:
                self._fail("vrm_model_load_failed")
                return
            if not bool(status.get("ready")):
                return
            if int(status.get("frames") or 0) < 2 or not bool(status.get("modelVisible")):
                return
            self._poll_timer.stop()
            self.render_ready = True
            self.ready_changed.emit(True)
            if self._pending_environment:
                self._call("applyEnvironment", self._pending_environment)
            if self._pending_state:
                self._call("applyState", self._pending_state)
            self.set_camera_mode(self._pending_camera_mode)
            self.set_rest_pose_enabled(self._pending_rest_pose_enabled)

        def _call(self, method: str, payload: dict[str, Any]) -> None:
            serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            self.page().runJavaScript(
                f"window.AnydesAvatar?.{method}({serialized});"
            )

        def _fail(self, public_code: str) -> None:
            self._poll_timer.stop()
            self.render_ready = False
            self.render_error = public_code
            self.ready_changed.emit(False)
            self.render_failed.emit(public_code)


def renderer_url(model_path: Path, *, base_url: str = ""):
    if not WEBENGINE_AVAILABLE:
        return ""
    if base_url:
        version = _renderer_asset_version()
        suffix = _render_profile_suffix()
        return QUrl(
            f"{base_url}/stage/index.html?model={base_url}/model.vrm&v={version}{suffix}"
        )
    stage = (STAGE_DIR / "index.html").resolve()
    url = QUrl.fromLocalFile(str(stage))
    query = QUrlQuery()
    query.addQueryItem("model", QUrl.fromLocalFile(str(model_path.resolve())).toString())
    _append_render_profile(query)
    url.setQuery(query)
    return url


def _append_render_profile(query) -> None:
    profile = os.environ.get("ANYDES_AVATAR_RENDER_PROFILE", "").strip().lower()
    if profile != "hq_efficient":
        return
    stage = os.environ.get("ANYDES_HQ_EFFICIENT_STAGE", "postprocess").strip().lower()
    allowed = {
        "baseline",
        "eyes",
        "hair",
        "skin",
        "clothing",
        "lighting",
        "postprocess",
    }
    query.addQueryItem("profile", "hq_efficient")
    query.addQueryItem("hqStage", stage if stage in allowed else "postprocess")


def _render_profile_suffix() -> str:
    profile = os.environ.get("ANYDES_AVATAR_RENDER_PROFILE", "").strip().lower()
    if profile != "hq_efficient":
        return ""
    stage = os.environ.get("ANYDES_HQ_EFFICIENT_STAGE", "postprocess").strip().lower()
    allowed = {
        "baseline",
        "eyes",
        "hair",
        "skin",
        "clothing",
        "lighting",
        "postprocess",
    }
    selected = stage if stage in allowed else "postprocess"
    return f"&profile=hq_efficient&hqStage={selected}"


def renderer_assets_available() -> bool:
    required = (
        STAGE_DIR / "index.html",
        STAGE_DIR / "stage.css",
        STAGE_DIR / "renderer.js",
        STAGE_DIR / "vendor" / "three.module.js",
        STAGE_DIR / "vendor" / "GLTFLoader.js",
        STAGE_DIR / "vendor" / "three-vrm-materials-mtoon.module.js",
        STAGE_DIR / "vendor" / "THREE-VRM-MTOON-LICENSE.txt",
        STAGE_DIR / "utils" / "BufferGeometryUtils.js",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _renderer_asset_version() -> int:
    candidates = (
        STAGE_DIR / "index.html",
        STAGE_DIR / "stage.css",
        STAGE_DIR / "renderer.js",
    )
    return max((path.stat().st_mtime_ns for path in candidates if path.exists()), default=0)


def _json_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("VRM renderer payload must be a dataclass or mapping.")
