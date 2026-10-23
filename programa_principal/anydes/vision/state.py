from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class FrameSnapshot:
    jpeg_bytes: bytes
    captured_at: datetime
    frame_hash: str
    width: int
    height: int
    source_id: str
    source_kind: str
    source_label: str
    session_id: str


@dataclass
class ScreenState:
    watching: bool = False
    last_capture_at: datetime | None = None
    last_capture_bytes: bytes | None = None
    source_id: str = ""
    source_kind: str = ""
    source_label: str = ""
    session_id: str = ""
    last_frame_hash: str = ""
    frame_width: int = 0
    frame_height: int = 0
    last_error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def store_capture(
        self,
        jpeg_bytes: bytes,
        *,
        captured_at: datetime | None = None,
        source_id: str = "",
        source_kind: str = "",
        source_label: str = "",
        session_id: str = "",
    ) -> None:
        captured = captured_at or datetime.now(timezone.utc)
        width, height = _jpeg_dimensions(jpeg_bytes)
        frame_hash = hashlib.sha256(jpeg_bytes).hexdigest()
        with self._lock:
            self.last_capture_bytes = jpeg_bytes
            self.last_capture_at = captured
            self.last_frame_hash = frame_hash
            self.frame_width = width
            self.frame_height = height
            if source_id:
                self.source_id = source_id
            if source_kind:
                self.source_kind = source_kind
            if source_label:
                self.source_label = source_label
            if session_id:
                self.session_id = session_id
            self.last_error = ""

    def clear_capture(self) -> None:
        with self._lock:
            self.last_capture_bytes = None
            self.last_capture_at = None
            self.source_id = ""
            self.source_kind = ""
            self.source_label = ""
            self.session_id = ""
            self.last_frame_hash = ""
            self.frame_width = 0
            self.frame_height = 0
            self.last_error = ""

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self.last_capture_bytes

    def latest_frame(self) -> FrameSnapshot | None:
        with self._lock:
            if (
                self.last_capture_bytes is None
                or self.last_capture_at is None
                or not self.last_frame_hash
            ):
                return None
            return FrameSnapshot(
                jpeg_bytes=self.last_capture_bytes,
                captured_at=self.last_capture_at,
                frame_hash=self.last_frame_hash,
                width=self.frame_width,
                height=self.frame_height,
                source_id=self.source_id,
                source_kind=self.source_kind,
                source_label=self.source_label,
                session_id=self.session_id,
            )

    def set_watching(self, active: bool) -> None:
        with self._lock:
            self.watching = active

    def set_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def source_snapshot(self) -> tuple[str, str, str, str, str]:
        with self._lock:
            return (
                self.source_id,
                self.source_kind,
                self.source_label,
                self.session_id,
                self.last_error,
            )

    def is_watching(self) -> bool:
        with self._lock:
            return self.watching

    def perception_snapshot(self) -> tuple[bool, datetime | None, bool]:
        with self._lock:
            return (
                self.watching,
                self.last_capture_at,
                bool(self.last_capture_bytes),
            )


_screen_state = ScreenState()


def get_screen_state() -> ScreenState:
    return _screen_state


def _jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(jpeg_bytes)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0
