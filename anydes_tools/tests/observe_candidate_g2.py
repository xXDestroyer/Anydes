"""Opt-in observation; isolated sampling failures and coordinated UI teardown."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path
import threading
import time
import weakref

ROOT = Path(__file__).resolve().parents[2]


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
        *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]]


class ProcessEntry(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD), ('pid', wintypes.DWORD),
        ('heap', ctypes.c_size_t), ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
        ('parent', wintypes.DWORD), ('priority', wintypes.LONG), ('flags', wintypes.DWORD),
        ('exe', wintypes.WCHAR * 260)]


def descendants():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = kernel.Process32FirstW.argtypes
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == wintypes.HANDLE(-1).value:
        return []
    entries = []
    try:
        row = ProcessEntry()
        row.size = ctypes.sizeof(row)
        more = kernel.Process32FirstW(handle, ctypes.byref(row))
        while more:
            entries.append({'pid':row.pid, 'parent':row.parent, 'threads':row.threads, 'exe':row.exe})
            more = kernel.Process32NextW(handle, ctypes.byref(row))
    finally:
        kernel.CloseHandle(handle)
    selected = {os.getpid()}
    while True:
        next_selected = selected | {e['pid'] for e in entries if e['parent'] in selected}
        if next_selected == selected:
            break
        selected = next_selected
    return [e for e in entries if e['pid'] in selected]


def install(output):
    output = Path(output)
    lock = threading.RLock()
    def record(event, **details):
        payload = {'event':event, 'timestamp':datetime.now(timezone.utc).isoformat(),
            'local_timestamp':datetime.now().astimezone().isoformat(),
            'monotonic':time.monotonic(), 'pid':os.getpid(), **details}
        with lock:
            try:
                with (output / 'lifecycle.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(payload, default=str) + '\n')
            except OSError as error:
                # Independent sink: telemetry storage must not abort a worker.
                sys.__stderr__.write(json.dumps({'event':'observer_unexpected_error',
                    'operation':'record', 'error_type':type(error).__name__,
                    'policy':'STDERR_FALLBACK_NOT_RUNTIME_ABORT'})+'\n')

    from PySide6.QtCore import QThread, QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from anydes_tools.observer_filesystem import scan_sizes, read_metadata_bytes, ObserverBoundary
    from anydes_tools.ui_lifecycle import install as install_lifecycle
    install_lifecycle(record)
    boundary = ObserverBoundary(record)
    import shiboken6
    tracked = weakref.WeakSet()
    observed_timers = weakref.WeakSet()
    parity_recorded = False
    original_start = QThread.start
    def start(worker, *args, **kwargs):
        key, kind = hex(id(worker)), type(worker).__name__
        if worker not in tracked:
            tracked.add(worker)
            worker.finished.connect(lambda: record('qthread_finished', worker=key, kind=kind), Qt.ConnectionType.DirectConnection)
            worker.destroyed.connect(lambda: record('qthread_destroyed', worker=key, kind=kind))
        record('qthread_start', worker=key, kind=kind)
        return original_start(worker, *args, **kwargs)
    QThread.start = start

    def snapshot(phase):
        nonlocal parity_recorded
        application = QApplication.instance()
        windows = application.topLevelWidgets() if application else []
        timers = {}
        for window in windows:
            for timer in window.findChildren(QTimer):
                if shiboken6.isValid(timer):
                    if timer not in observed_timers:
                        observed_timers.add(timer)
                        timer_key = hex(id(timer))
                        timer.destroyed.connect(lambda _obj=None, key=timer_key: record('qtimer_destroyed', timer=key))
                    timers[hex(id(timer))] = {'active':timer.isActive(), 'interval':timer.interval()}
        workers = [{'worker':hex(id(w)), 'kind':type(w).__name__, 'running':w.isRunning(),
            'finished':w.isFinished()} for w in list(tracked) if shiboken6.isValid(w)]
        memory = MemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        valid = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb)
        sample_sizes = scan_sizes(output, record)
        from anydes.nucleo.vram_safety import get_active_vram_safety_manager
        safety = get_active_vram_safety_manager()
        sample = safety.latest_sample if safety is not None else None
        if not parity_recorded and 'anydes_tools.conversation.controller' in sys.modules:
            names = ['anydes_tools.conversation.interaction_router', 'anydes_tools.conversation.plan_lowering',
                'anydes_tools.executor', 'anydes_tools.windows_sandbox',
                'anydes_tools.conversation.dataflow', 'anydes_tools.conversation.typed_verification',
                'anydes_tools.conversation.oracle_registry', 'anydes_tools.conversation.publication']
            modules = {}
            for name in names:
                module = sys.modules.get(name)
                filename = getattr(module, '__file__', None)
                if filename:
                    p = Path(filename).resolve()
                    content = read_metadata_bytes(p, record)
                    if content is not None:
                        modules[name] = {'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(content).hexdigest()}
            record('module_parity', modules=modules)
            parity_recorded = True
        record('qt_snapshot', phase=phase, workers=workers, timers=timers,
            processes=descendants(), **sample_sizes,
            vram_used_mb=sample.used_mb if sample else None,
            tts_reserved_mb=sample.tts_reserved_mb if sample else None,
            python_threads=[{'kind':t.name, 'alive':t.is_alive()} for t in threading.enumerate()],
            working_set=memory.WorkingSetSize if valid else None,
            peak_working_set=memory.PeakWorkingSetSize if valid else None,
            private_bytes=memory.PrivateUsage if valid else None)

    original_exec = QApplication.exec
    def execute(application):
        timer = QTimer(application)
        timer.setInterval(5000)
        timer.timeout.connect(lambda: boundary.call('PERIODIC', lambda: snapshot('PERIODIC')))
        application.aboutToQuit.connect(lambda: boundary.call('ABOUT_TO_QUIT', lambda: snapshot('ABOUT_TO_QUIT')))
        try:
            timer.start()
            record('event_loop_enter')
            boundary.call('EVENT_LOOP_START', lambda: snapshot('EVENT_LOOP_START'))
            return original_exec()
        finally:
            timer.stop()
            record('event_loop_return')
            boundary.call('EVENT_LOOP_RETURN', lambda: snapshot('EVENT_LOOP_RETURN'))
    QApplication.exec = execute

    import http.client
    original_request = http.client.HTTPSConnection.request
    original_response = http.client.HTTPSConnection.getresponse
    def request(connection, method, url, body=None, headers={}, **kwargs):
        if connection.host == 'api.groq.com':
            try:
                data = json.loads(body) if body else {}
            except (ValueError, TypeError):
                data = {}
            record('groq_request', model=data.get('model'), streaming=data.get('stream'),
                max_tokens=data.get('max_tokens'), temperature=data.get('temperature'))
        return original_request(connection, method, url, body, headers, **kwargs)
    def response(connection, *args, **kwargs):
        result = original_response(connection, *args, **kwargs)
        if connection.host == 'api.groq.com':
            record('groq_http_response', status=result.status)
        return result
    http.client.HTTPSConnection.request = request
    http.client.HTTPSConnection.getresponse = response
    from anydes.hybrid.router import Ollama8BClient
    original_stream = Ollama8BClient.stream
    def stream(client, *args, **kwargs):
        record('fallback_request', model=client.model)
        return original_stream(client, *args, **kwargs)
    Ollama8BClient.stream = stream
    record('instrumentation_installed', kind='G2_OPT_IN_NO_MOCKS')

