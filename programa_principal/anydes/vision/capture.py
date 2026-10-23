from __future__ import annotations

import base64
import ctypes
import io
import os
from dataclasses import dataclass


def _require_capture_deps():
    try:
        import mss
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Para ver la pantalla instala dependencias de vision: pip install mss Pillow"
        ) from exc
    return mss, Image


@dataclass(frozen=True)
class CaptureSource:
    id: str
    kind: str
    label: str
    monitor: int | None = None
    region: tuple[int, int, int, int] | None = None
    native_handle: int | None = None


def list_capture_sources() -> tuple[CaptureSource, ...]:
    """Return currently selectable screens and visible windows.

    Window titles are kept in memory for the permission dialog only. They are
    never persisted or sent to a vision provider.
    """

    mss, _ = _require_capture_deps()
    sources: list[CaptureSource] = []
    with _mss_context(mss) as sct:
        monitors = tuple(sct.monitors)
    if monitors:
        sources.append(
            CaptureSource(
                id="monitor:0",
                kind="screen",
                label="Todas las pantallas",
                monitor=0,
            )
        )
    for index in range(1, len(monitors)):
        sources.append(
            CaptureSource(
                id=f"monitor:{index}",
                kind="screen",
                label=f"Pantalla {index}",
                monitor=index,
            )
        )
    if os.name == "nt":
        sources.extend(_visible_windows())
    return tuple(sources)


def capture_source_jpeg(
    source: CaptureSource,
    *,
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 85,
) -> bytes:
    if source.kind == "window":
        region = (
            _window_region(source.native_handle)
            if source.native_handle is not None
            else source.region
        )
        if region is None:
            raise ValueError("window_capture_region_missing")
        return _capture_jpeg(
            region=region,
            max_width=max_width,
            max_height=max_height,
            quality=quality,
        )
    return capture_screen_jpeg(
        monitor=int(source.monitor or 0),
        max_width=max_width,
        max_height=max_height,
        quality=quality,
    )


def capture_screen_jpeg(
    *,
    monitor: int = 0,
    max_width: int = 1280,
    max_height: int = 720,
    quality: int = 85,
) -> bytes:
    mss, _ = _require_capture_deps()
    with _mss_context(mss) as sct:
        monitors = sct.monitors
        if monitor < 0 or monitor >= len(monitors):
            monitor = 1 if len(monitors) > 1 else 0
        selected = monitors[monitor]
    region = (
        int(selected["left"]),
        int(selected["top"]),
        int(selected["width"]),
        int(selected["height"]),
    )
    return _capture_jpeg(
        region=region,
        max_width=max_width,
        max_height=max_height,
        quality=quality,
    )


def _capture_jpeg(
    *,
    region: tuple[int, int, int, int],
    max_width: int,
    max_height: int,
    quality: int,
) -> bytes:
    mss, Image = _require_capture_deps()
    left, top, width, height = region
    if width <= 0 or height <= 0:
        raise ValueError("capture_region_empty")
    monitor = {"left": left, "top": top, "width": width, "height": height}
    with _mss_context(mss) as sct:
        shot = sct.grab(monitor)
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    if max_width > 0 and max_height > 0:
        img.thumbnail((max_width, max_height))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def capture_screen_base64(**kwargs) -> str:
    return base64.b64encode(capture_screen_jpeg(**kwargs)).decode("ascii")


def _mss_context(mss_module):
    factory = getattr(mss_module, "MSS", None) or mss_module.mss
    return factory()


def _visible_windows() -> tuple[CaptureSource, ...]:
    try:
        import win32gui
    except ImportError:
        return ()

    found: list[CaptureSource] = []

    def collect(hwnd, _extra) -> None:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return
            if win32gui.IsIconic(hwnd):
                return
            title = " ".join(str(win32gui.GetWindowText(hwnd) or "").split())
            if not title or title.casefold() in {"program manager", "default ime"}:
                return
            region = _window_region(int(hwnd))
            if region is None:
                return
            left, top, width, height = region
            if width < 160 or height < 120:
                return
            found.append(
                CaptureSource(
                    id=f"window:{int(hwnd)}",
                    kind="window",
                    label=title[:96],
                    region=(int(left), int(top), width, height),
                    native_handle=int(hwnd),
                )
            )
        except Exception:
            return

    try:
        win32gui.EnumWindows(collect, None)
    except Exception:
        return ()
    found.sort(key=lambda item: item.label.casefold())
    return tuple(found[:40])


def _window_region(hwnd: int | None) -> tuple[int, int, int, int] | None:
    """Resolve current physical window bounds instead of reusing stale geometry."""

    if os.name != "nt" or hwnd is None:
        return None
    try:
        import win32gui

        if (
            not win32gui.IsWindow(int(hwnd))
            or not win32gui.IsWindowVisible(int(hwnd))
            or win32gui.IsIconic(int(hwnd))
        ):
            return None
        rect = _dwm_extended_frame_bounds(int(hwnd))
        if rect is None:
            rect = tuple(int(value) for value in win32gui.GetWindowRect(int(hwnd)))
        left, top, right, bottom = rect
        width = int(right - left)
        height = int(bottom - top)
        if width <= 0 or height <= 0:
            return None
        return int(left), int(top), width, height
    except Exception:
        return None


def _dwm_extended_frame_bounds(hwnd: int) -> tuple[int, int, int, int] | None:
    """DWM returns physical bounds and avoids DPI-virtualized GetWindowRect values."""

    try:
        class Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = Rect()
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.c_void_p(hwnd),
            ctypes.c_uint(9),  # DWMWA_EXTENDED_FRAME_BOUNDS
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result != 0:
            return None
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        return None
