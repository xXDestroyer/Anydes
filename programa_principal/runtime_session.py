"""One desktop runtime at a time; release its models after the UI closes."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import sys


class RuntimeAlreadyOpen(RuntimeError):
    pass


class RuntimeLease:
    def __enter__(self):
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = self.kernel.CreateMutexW(None, True, "Local\\Anydes.Desktop.Runtime")
        error = ctypes.get_last_error()
        if not self.handle:
            raise ctypes.WinError(error)
        if error == 183:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise RuntimeAlreadyOpen("Anydes ya esta abierta. Cierra la ventana actual antes de cambiar de modo.")
        return self

    def __exit__(self, *_):
        if self.handle:
            self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def runtime_session(mode):
    from modos_arranque import mode_by_key, prepare_exclusive_brain
    from rc_runtime_cleanup import unload_rc_models
    with RuntimeLease():
        prepare_exclusive_brain(mode_by_key(mode))
        try:
            yield
        finally:
            unload_rc_models(("qwen3:8b", "qwen3:14b", "moondream"))


def run_desktop(mode, callback):
    try:
        with runtime_session(mode):
            return callback()
    except (RuntimeAlreadyOpen, OSError) as exc:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.information(None, "Anydes", str(exc))
        return 1
