"""Staging startup diagnostics only; no task/core/renderer behavior changes."""
from datetime import datetime, timezone
from functools import wraps
import json
import os
import time
import traceback

START=time.monotonic()


def record(event,**fields):
    print(json.dumps({'startup_event':event,'pid':os.getpid(),
        'timestamp':datetime.now(timezone.utc).isoformat(),
        'elapsed_ms':round((time.monotonic()-START)*1000,1),**fields}),flush=True)


def rectangle(rect):
    return [rect.x(),rect.y(),rect.width(),rect.height()]


def snapshot(widget):
    from PySide6.QtWidgets import QApplication
    screens=QApplication.screens()
    frame=widget.frameGeometry()
    handle=widget.windowHandle()
    return {'class_name':type(widget).__name__,'visible':widget.isVisible(),
        'hidden':widget.isHidden(),'minimized':widget.isMinimized(),
        'geometry':rectangle(frame),
        'intersects_active_display':any(frame.intersects(s.availableGeometry()) for s in screens),
        'screen_index':next((i for i,s in enumerate(screens) if widget.screen()==s),None),
        'window_handle_visible':handle.isVisible() if handle else None,
        'exposed':handle.isExposed() if handle else None,
        'top_level_visible_count':sum(w.isVisible() for w in QApplication.topLevelWidgets())}


def wrap(owner,name,label):
    original=getattr(owner,name)
    @wraps(original)
    def traced(*args,**kwargs):
        record(label+'_entered')
        try:
            value=original(*args,**kwargs)
        except BaseException as error:
            record(label+'_exception',exception_type=type(error).__name__,
                traceback=[{'file':f.filename,'function':f.name,'line':f.lineno}
                    for f in traceback.extract_tb(error.__traceback__)])
            raise
        record(label+'_returned',exit_code=value if type(value) is int else None)
        return value
    setattr(owner,name,traced)


def install():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(90,repeat=False)
    import anydes_hybrid_runtime as hybrid
    wrap(hybrid,'_unload_model','startup_model_release')
    import runtime_session
    wrap(runtime_session,'run_desktop','desktop_runtime')
    from anydes.official_runtime import integration
    wrap(integration,'run_mode','runtime_host')
    from PySide6 import QtWidgets
    from PySide6.QtCore import QObject,QEvent,QTimer
    base=QtWidgets.QApplication

    class StartupEvents(QObject):
        def eventFilter(self,obj,event):
            if event.type() not in {QEvent.Type.Show,QEvent.Type.Hide,QEvent.Type.Close,
                    QEvent.Type.WindowStateChange,QEvent.Type.Expose}:
                return False
            try:
                if isinstance(obj,QtWidgets.QWidget) and obj.isWindow():
                    record('top_level_'+event.type().name,**snapshot(obj))
                    if 'ProfessionalWindow' in type(obj).__name__ and event.type()==QEvent.Type.Show:
                        faulthandler.cancel_dump_traceback_later()
                        web=getattr(getattr(obj,'avatar',None),'web',None)
                        if web is not None and not getattr(web,'_startup_observed',False):
                            web._startup_observed=True
                            record('webengine_created',loopback=web.url().host()=='127.0.0.1')
                            web.loadFinished.connect(lambda ok:record('presence_frame_loaded',success=bool(ok)))
                            web.renderProcessTerminated.connect(lambda status,code:record(
                                'webengine_terminated',status=status.name,exit_code=code))
                    QTimer.singleShot(0,lambda w=obj:self.after_show(w))
            except RuntimeError:
                pass
            return False

        def after_show(self,obj):
            try:
                record('event_loop_window_snapshot',**snapshot(obj))
            except RuntimeError:
                pass

    class ObservedApplication(base):
        def __init__(self,*args,**kwargs):
            record('qapplication_constructing')
            super().__init__(*args,**kwargs)
            self._startup_events=StartupEvents(self)
            self.installEventFilter(self._startup_events)
            record('qapplication_created',screen_count=len(self.screens()),
                active_display_rects=[rectangle(s.availableGeometry()) for s in self.screens()])

        def exec(self):
            record('qt_main_event_loop_entered',top_level_visible_count=sum(w.isVisible() for w in self.topLevelWidgets()))
            try:
                return super().exec()
            finally:
                record('qt_main_event_loop_returned')

    QtWidgets.QApplication=ObservedApplication
    # Preserve the selector's behavior; trace construction and its nested event loop.
    import shell
    choice=shell.PresenceChoice
    class ObservedChoice(choice):
        def __init__(self,*args,**kwargs):
            record('selector_constructing')
            super().__init__(*args,**kwargs)
            record('selector_loaded',**snapshot(self))
        def exec(self):
            record('selector_event_loop_entered')
            result=super().exec()
            record('selector_returned',accepted=result==QtWidgets.QDialog.DialogCode.Accepted)
            return result
    shell.PresenceChoice=ObservedChoice
    original=shell.install_shell
    def install_shell(*args,**kwargs):
        result=original(*args,**kwargs)
        module=args[0]
        window=module.AnydesPySideWindow
        class StartupProfessionalWindow(window):
            def __init__(self,*a,**k):
                record('main_window_constructing')
                super().__init__(*a,**k)
                record('main_window_constructed',**snapshot(self))
            def show(self):
                record('main_window_show_called')
                return super().show()
        module.AnydesPySideWindow=StartupProfessionalWindow
        return result
    shell.install_shell=install_shell
    record('startup_observer_installed')
