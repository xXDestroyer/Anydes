"""Validation-only screenshots; no input injection or substitute runtime."""
from pathlib import Path
import json
from PySide6.QtCore import QTimer
import run_shell
import shell

original_install = shell.install_shell


def install(*args, **kwargs):
    original_install(*args, **kwargs)
    module=args[0]
    base=module.AnydesPySideWindow

    class CaptureWindow(base):
        def __init__(self):
            super().__init__()
            self.captured=set()
            self.capture_timer=QTimer(self)
            self.capture_timer.timeout.connect(self.capture_ready)
            self.capture_timer.start(1000)

        def capture_ready(self):
            if self._closing or not self.avatar.ready or self.avatar.failed:
                return
            state=self.avatar.frame['state']
            key=(self.presence_kind.value,state)
            if key in self.captured:
                return
            self.captured.add(key)
            self.grab().save(str(run_shell.SESSION/f'shell-{key[0]}-{state}.png'))
            if self.avatar.web:
                self.avatar.web.page().runJavaScript(
                    'JSON.stringify(window.anydesPresence.snapshot())',
                    lambda value:self.save_snapshot(state,value))

        def save_snapshot(self,state,value):
            if not value:
                return
            data=json.loads(value)
            allowed={key:data.get(key) for key in ('mode','state','frames','webgl','petals','workingMode','connectedToAnydes','speech')}
            (run_shell.SESSION/f'render-{state}.json').write_text(json.dumps(allowed),encoding='utf-8')

        def closeEvent(self,event):
            super().closeEvent(event)
            if event.isAccepted():
                self.capture_timer.stop()

    module.AnydesPySideWindow=CaptureWindow


shell.install_shell=install
if __name__=='__main__':
    raise SystemExit(run_shell.stage.host.main())
