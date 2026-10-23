from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import threading

_current_turn = ContextVar("anydes_current_turn", default=None)


def current_turn():
    return _current_turn.get()


class TurnToken:
    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._aborters = set()

    def check(self):
        if self.cancelled.is_set():
            from anydes.realtime.cancellation import ResponseSuperseded
            raise ResponseSuperseded()

    def cancel(self):
        with self._lock:
            self.cancelled.set()
            callbacks = tuple(self._aborters)
        for callback in callbacks:
            callback()

    @contextmanager
    def abort_with(self, callback):
        with self._lock:
            self._aborters.add(callback)
            cancelled = self.cancelled.is_set()
        try:
            if cancelled:
                callback()
            self.check()
            yield
        finally:
            with self._lock:
                self._aborters.discard(callback)

    @contextmanager
    def bind(self):
        handle = _current_turn.set(self)
        try:
            self.check()
            yield
        finally:
            _current_turn.reset(handle)

class LatestTurn:
    """One active response and at most one replacement, never a response backlog."""

    def __init__(self):
        self.revision = 0
        self.pending = None
        self.token = TurnToken(0)
        self.discarded_callbacks = 0

    def submit(self, text):
        self.revision += 1
        old = self.token
        self.token = TurnToken(self.revision)
        self.pending = (self.revision, text)
        old.cancel()
        return self.revision

    def take(self):
        pending, self.pending = self.pending, None
        return pending

    def current(self, revision):
        valid = revision == self.revision
        self.discarded_callbacks += int(not valid)
        return valid

    def close(self):
        self.revision += 1
        self.token.cancel()
        self.pending = None


def restore_input_focus(widget):
    """Restore typing in our active window, without stealing another app's focus."""
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication

    def restore():
        if (widget.isVisible() and widget.isEnabled() and widget.window().isActiveWindow()
                and QApplication.activeModalWidget() is None
                and QApplication.activePopupWidget() is None):
            widget.setFocus(Qt.FocusReason.OtherFocusReason)
    QTimer.singleShot(0, widget, restore)
