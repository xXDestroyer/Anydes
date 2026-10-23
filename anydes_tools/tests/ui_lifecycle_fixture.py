"""Offscreen synthetic voice-worker fixture for the actual staging adapter."""
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'programa_principal')]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def main():
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow
    import anydes.voice_input.integration as integration
    from anydes_tools.ui_lifecycle import install
    events = []
    case = sys.argv[1]
    app = QApplication([])
    result = {'stopped_before_owner_close': False, 'normal_return': False,
              'outer_exception_received_after_stop': False}

    class SimulatedVoiceWarmupWorker(QThread):
        def run(self):
            if case == 'timeout':
                self.msleep(150)
            while not self.isInterruptionRequested():
                self.msleep(5)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self._voice_warmup_worker = SimulatedVoiceWarmupWorker()
            self._voice_warmup_worker.start()

        def closeEvent(self, event):
            result['stopped_before_owner_close'] = not self._voice_warmup_worker.isRunning()
            super().closeEvent(event)

    module = SimpleNamespace(AnydesPySideWindow=Window)

    def original_main():
        window = module.AnydesPySideWindow()
        window.show()
        if case in {'exception', 'timeout'}:
            raise OSError('fixture runtime failure after worker start')
        if case == 'callback':
            def fail_callback():
                raise ValueError('fixture callback failure')
            QTimer.singleShot(50, fail_callback)
        else:
            QTimer.singleShot(100, window.close)
        app.exec()
        result['normal_return'] = True

    module.main = original_main
    with patch.object(integration, 'install', lambda *args, **kwargs: None):
        coordinator = install(lambda event, **details: events.append((event, details)))
        if case == 'timeout':
            coordinator.wait_ms = 20
        integration.install(module)
        try:
            module.main()
        except OSError:
            result['outer_exception_received_after_stop'] = not coordinator.running()
            # Error dialog would be shown only here, after guarded_main shutdown.
    gc.collect()
    app.processEvents()
    result.update(running=len(coordinator.running()),
                  started=sum(e == 'thread_started' for e, _ in events),
                  finished=sum(e == 'thread_finished' for e, _ in events),
                  destroyed=sum(e == 'thread_destroyed' for e, _ in events),
                  pending=sum(e == 'shutdown_pending' for e, _ in events))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
