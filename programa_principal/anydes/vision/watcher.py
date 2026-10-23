from __future__ import annotations

import threading
from contextlib import contextmanager
from uuid import uuid4

from anydes.nucleo.config import Config
from anydes.nucleo.sensors import sensor_hub
from anydes.vision.capture import CaptureSource, capture_source_jpeg
from anydes.vision.state import get_screen_state

_active_watcher: ScreenWatcher | None = None
_active_watcher_lock = threading.RLock()


class ScreenWatcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = get_screen_state()
        self._source: CaptureSource | None = None
        self._session_id = ""
        self._capture_lock = threading.RLock()

    def start(self, source: CaptureSource | None = None) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        self._source = source or CaptureSource(
            id=f"monitor:{self.config.screen_monitor}",
            kind="screen",
            label=(
                "Todas las pantallas"
                if self.config.screen_monitor == 0
                else f"Pantalla {self.config.screen_monitor}"
            ),
            monitor=self.config.screen_monitor,
        )
        self._session_id = f"vision:{uuid4()}"
        self._stop_event.clear()
        if not self._capture_once():
            self._source = None
            self._session_id = ""
            return False
        self._state.set_watching(True)
        _set_active_watcher(self)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="anydes-screen-watch")
        self._thread.start()
        sensor_hub.register("screen", str(id(self)), self.stop)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._state.set_watching(False)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._source = None
        self._session_id = ""
        _clear_active_watcher(self)
        sensor_hub.unregister("screen", str(id(self)))

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        interval = self.config.resolved_screen_watch_interval_seconds()
        while not self._stop_event.wait(interval):
            self._capture_once()

    def _capture_once(self) -> bool:
        with self._capture_lock:
            source = self._source
            if source is None or self._stop_event.is_set():
                return False
            try:
                jpeg = capture_source_jpeg(
                    source,
                    max_width=self.config.screen_max_width,
                    max_height=self.config.screen_max_height,
                    quality=self.config.screen_jpeg_quality,
                )
                if not jpeg:
                    raise RuntimeError("empty_screen_capture")
                self._state.store_capture(
                    jpeg,
                    source_id=source.id,
                    source_kind=source.kind,
                    source_label=source.label,
                    session_id=self._session_id,
                )
                return True
            except Exception as exc:
                self._state.set_error(type(exc).__name__)
                return False

    def capture_now(self) -> bool:
        return self._capture_once()


def refresh_active_screen_frame() -> bool:
    with _active_watcher_lock:
        watcher = _active_watcher
    return bool(watcher is not None and watcher.capture_now())


@contextmanager
def active_screen_frame_lease(*, refresh: bool = True):
    """Keep the analyzed frame stable while the local visual model reads it."""

    with _active_watcher_lock:
        watcher = _active_watcher
    if watcher is None:
        yield False
        return
    with watcher._capture_lock:
        ready = watcher.capture_now() if refresh else True
        yield ready


def _set_active_watcher(watcher: ScreenWatcher) -> None:
    global _active_watcher
    with _active_watcher_lock:
        _active_watcher = watcher


def _clear_active_watcher(watcher: ScreenWatcher) -> None:
    global _active_watcher
    with _active_watcher_lock:
        if _active_watcher is watcher:
            _active_watcher = None
