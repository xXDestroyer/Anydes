"""Candidate-only coordinated Qt shutdown; no audio implementation changes."""
import sys
import time
import weakref

_coordinator = None


class ShutdownCoordinator:
    def __init__(self, record, wait_ms=10000):
        self.record = record
        self.wait_ms = wait_ms
        self.workers = weakref.WeakSet()
        self.windows = weakref.WeakSet()
        self.shutting_down = False
        self.waiting = False

    def track(self, worker):
        from PySide6.QtCore import Qt
        if worker in self.workers:
            return
        self.workers.add(worker)
        key, kind = hex(id(worker)), type(worker).__name__
        worker.finished.connect(lambda: self.record('thread_finished', worker=key, kind=kind), Qt.ConnectionType.DirectConnection)
        worker.destroyed.connect(lambda: self.record('thread_destroyed', worker=key, kind=kind))

    def running(self):
        import shiboken6
        return [w for w in self.workers if shiboken6.isValid(w) and w.isRunning()]

    def prepare(self, reason):
        from PySide6.QtCore import QTimer
        import shiboken6
        self.shutting_down = True
        self.record('shutdown_requested', reason=reason)
        ready = True

        def action(component, callback):
            nonlocal ready
            try:
                return callback()
            except Exception as error:
                ready = False
                self.record('shutdown_component_error', component=component,
                            error_type=type(error).__name__, policy='KEEP_OWNER_ALIVE')
                return False

        for window in list(self.windows):
            if not shiboken6.isValid(window):
                continue
            window._closing = True
            turns = getattr(window, '_turns', None)
            if turns is not None:
                action('turns', turns.close)
            cancel = getattr(window, '_cancel_active_voice', None)
            if callable(cancel):
                action('voice_cancel', cancel)
            for timer in window.findChildren(QTimer):
                if shiboken6.isValid(timer):
                    timer.stop()
            panel = getattr(window, 'input_panel', None)
            if panel is not None and not action('input_panel', panel.shutdown):
                self.record('shutdown_pending', component='input_panel')
                ready = False
        return ready

    def stop(self, reason):
        from PySide6.QtCore import QThread
        if self.waiting:
            return False
        self.waiting = True
        try:
            owners_ready = self.prepare(reason)
            workers = self.running()  # Scoped ownership until stop verification.
            for worker in workers:
                self.record('worker_stop_requested', worker=hex(id(worker)))
                worker.requestInterruption()
                worker.quit()  # Event-loop workers; no effect on blocking run().
                self.record('thread_quit', worker=hex(id(worker)))
            deadline = time.monotonic() + self.wait_ms / 1000
            for worker in workers:
                if worker is QThread.currentThread():
                    self.record('shutdown_pending', component='current_thread')
                    return False
                remaining = max(0, int((deadline-time.monotonic()) * 1000))
                self.record('thread_wait_start', worker=hex(id(worker)), timeout_ms=remaining)
                worker.wait(remaining)
                if worker.isRunning():
                    self.record('shutdown_pending', component='qthread', worker=hex(id(worker)))
                    return False
            self.record('shutdown_workers_stopped', running=len(self.running()))
            return owners_ready and not self.running()
        finally:
            self.waiting = False

    def drain_exception(self):
        # A failed bounded wait is not permission to destroy a running worker.
        # Stay in an explicit shutdown-pending scope; never terminate() a thread.
        while not self.stop('EXCEPTION_TEARDOWN'):
            time.sleep(0.05)


def install(record=None):
    global _coordinator
    if _coordinator is not None:
        if record is not None:
            _coordinator.record = record
        return _coordinator
    from PySide6.QtCore import QTimer, QObject, Signal, Slot
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    import anydes.voice_input.integration as integration
    coordinator = ShutdownCoordinator(record or (lambda *args, **kwargs: None))
    _coordinator = coordinator
    bridge = None

    class ShutdownBridge(QObject):
        requested = Signal()

        def __init__(self, app):
            super().__init__(app)
            self.requested.connect(self.shutdown)

        @Slot()
        def shutdown(self):
            import shiboken6
            coordinator.drain_exception()
            for window in list(coordinator.windows):
                if shiboken6.isValid(window):
                    window.close()
            QApplication.instance().exit(1)

    def ensure_bridge():
        nonlocal bridge
        if bridge is None:
            bridge = ShutdownBridge(QApplication.instance())

    original_start = QThread.start

    def start(worker, *args, **kwargs):
        if coordinator.shutting_down:
            coordinator.record('worker_start_denied', kind=type(worker).__name__)
            raise RuntimeError('candidate_shutdown_in_progress')
        coordinator.track(worker)
        coordinator.record('thread_started', worker=hex(id(worker)), kind=type(worker).__name__)
        return original_start(worker, *args, **kwargs)
    QThread.start = start

    original_install = integration.install

    def install_window(module, *args, **kwargs):
        result = original_install(module, *args, **kwargs)
        performance = kwargs.get('performance', args[0] if args else False)
        name = 'PerformanceCompanionWindow' if performance else 'AnydesPySideWindow'
        base = getattr(module, name)

        class CoordinatedWindow(base):
            def __init__(self, *args, **kwargs):
                ensure_bridge()
                super().__init__(*args, **kwargs)
                coordinator.windows.add(self)
                app = QApplication.instance()
                app.aboutToQuit.connect(self._candidate_about_to_quit)

            def _handle_realtime_ready(self, *args, **kwargs):
                if not coordinator.shutting_down:
                    return super()._handle_realtime_ready(*args, **kwargs)

            def _candidate_about_to_quit(self):
                coordinator.record('about_to_quit')
                coordinator.drain_exception()

            def closeEvent(self, event):
                coordinator.record('window_close_event')
                if not coordinator.stop('WINDOW_CLOSE'):
                    event.ignore()
                    QTimer.singleShot(250, self.close)
                    return
                super().closeEvent(event)

        setattr(module, name, CoordinatedWindow)
        main = module.main

        def guarded_main(*args, **kwargs):
            try:
                return main(*args, **kwargs)
            except BaseException as error:
                import shiboken6
                app = QApplication.instance()
                if app is not None:
                    for window in app.topLevelWidgets():
                        if hasattr(window, '_voice_warmup_worker'):
                            coordinator.windows.add(window)
                event = 'ui_main_return' if isinstance(error, SystemExit) and not error.code else 'runtime_exception'
                coordinator.record(event, error_type=type(error).__name__,
                                   boundary='UI_MAIN', policy='SHUTDOWN_BEFORE_RERAISE')
                coordinator.drain_exception()
                # Existing closeEvent owns manager/assistant cleanup after all
                # QThreads stopped, before an outer wrapper may show a dialog.
                for window in list(coordinator.windows):
                    if shiboken6.isValid(window):
                        window.close()
                raise
        module.main = guarded_main
        return result
    integration.install = install_window
    old_hook = sys.excepthook

    def exception_hook(kind, error, traceback):
        coordinator.record('runtime_exception', error_type=kind.__name__, boundary='QT_CALLBACK')
        app = QApplication.instance()
        if app is not None and bridge is not None:
            bridge.requested.emit()
        old_hook(kind, error, traceback)
    sys.excepthook = exception_hook
    import threading
    old_thread_hook = threading.excepthook

    def thread_exception_hook(args):
        coordinator.record('runtime_exception', error_type=args.exc_type.__name__, boundary='PYTHON_THREAD')
        if bridge is not None:
            bridge.requested.emit()
        old_thread_hook(args)
    threading.excepthook = thread_exception_hook
    return coordinator
