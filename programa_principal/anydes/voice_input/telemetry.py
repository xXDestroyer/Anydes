"""Read-only, session-scoped RAM diagnostics. Never samples on the audio callback."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from .environment import OUT, setup
setup()
import psutil


def nvidia_snapshot():
    """Low-rate NVIDIA telemetry. Failure is explicit and never guessed."""
    command = [
        'nvidia-smi',
        '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ]
    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2,
            check=True, creationflags=creationflags)
        values = [part.strip() for part in completed.stdout.splitlines()[0].split(',')]
        if len(values) != 5:
            raise ValueError('unexpected nvidia-smi column count')
        def number(value):
            return None if value in {'', '[N/A]', 'N/A'} else float(value)
        used, free, utilization, temperature, power = map(number, values)
        return dict(memory_used_mib=used, memory_free_mib=free,
            utilization_percent=utilization, temperature_c=temperature,
            power_w=power)
    except Exception as exc:
        return {'unavailable': f'{type(exc).__name__}: {exc}'}


def windows_memory():
    """Commit is not pagefile usage. Query both independently in bytes."""
    if os.name != 'nt':
        return {'unavailable': 'Windows PSAPI unavailable'}
    size = ctypes.c_size_t
    class Performance(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(key, size) for key in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
            'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
            'KernelNonpaged', 'PageSize')] + [(key, wintypes.DWORD) for key in (
            'HandleCount', 'ProcessCount', 'ThreadCount')]
    class PageFile(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('Reserved', wintypes.DWORD),
                    ('TotalSize', size), ('TotalInUse', size), ('PeakUsage', size)]
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetPerformanceInfo.argtypes = [ctypes.POINTER(Performance), wintypes.DWORD]
    api.GetPerformanceInfo.restype = wintypes.BOOL
    perf = Performance(); perf.cb = ctypes.sizeof(perf)
    if not api.GetPerformanceInfo(ctypes.byref(perf), perf.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    result = {key: int(getattr(perf, source) * perf.PageSize) for key, source in (
        ('commit_bytes', 'CommitTotal'), ('commit_limit_bytes', 'CommitLimit'),
        ('commit_peak_since_boot_bytes', 'CommitPeak'))}
    entries = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p,
        ctypes.POINTER(PageFile), wintypes.LPCWSTR)
    def receive(_, info, _filename):
        entries.append({key: int(getattr(info.contents, source) * perf.PageSize)
            for key, source in (('capacity_bytes', 'TotalSize'),
                ('used_bytes', 'TotalInUse'), ('peak_since_boot_bytes', 'PeakUsage'))})
        return True
    callback = callback_type(receive)
    api.EnumPageFilesW.argtypes = [callback_type, ctypes.c_void_p]
    api.EnumPageFilesW.restype = wintypes.BOOL
    if api.EnumPageFilesW(callback, None):
        result['pagefile'] = {key: sum(row[key] for row in entries)
            for key in ('capacity_bytes', 'used_bytes', 'peak_since_boot_bytes')}
    else:
        result['pagefile'] = None
        result['pagefile_error'] = str(ctypes.WinError(ctypes.get_last_error()))
    return result


class PagingRates:
    """System hard-fault disk reads, not process-level SSD swapping attribution."""
    def __init__(self):
        self.query = ctypes.c_void_p(); self.counters = {}
        self.api = ctypes.WinDLL('pdh') if os.name == 'nt' else None
        if self.api is None:
            raise RuntimeError('Windows PDH unavailable')
        self.api.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p)]
        self.api.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.api.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.api.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        self.api.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        status = self.api.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if status:
            raise RuntimeError(f'PdhOpenQuery: {status:#x}')
        try:
            for key, path in (
                ('page_reads_per_second', r'\Memory\Page Reads/sec'),
                ('pages_input_per_second', r'\Memory\Pages Input/sec')):
                counter = ctypes.c_void_p()
                status = self.api.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(counter))
                if status:
                    raise RuntimeError(f'PdhAddEnglishCounter: {status:#x}')
                self.counters[key] = counter
            self.api.PdhCollectQueryData(self.query)
        except Exception:
            self.close()
            raise

    def sample(self):
        class Value(ctypes.Structure):
            _fields_ = [('status', wintypes.DWORD), ('value', ctypes.c_double)]
        status = self.api.PdhCollectQueryData(self.query)
        if status:
            return {'error': f'PdhCollectQueryData: {status:#x}'}
        result = {}
        for key, counter in self.counters.items():
            value = Value()
            status = self.api.PdhGetFormattedCounterValue(counter, 0x200, None, ctypes.byref(value))
            result[key] = value.value if status == 0 and value.status in (0, 1) else None
            if result[key] is None:
                result[key + '_status'] = [status, value.status]
        return result

    def close(self):
        if self.query.value:
            self.api.PdhCloseQuery(self.query); self.query = ctypes.c_void_p()


def process_memory_rows():
    """Avoid psutil's expensive fallback for inaccessible Windows processes."""
    if os.name != 'nt':
        return [dict(pid=p.pid, name=p.info['name'], parent=p.info['ppid'],
            created_at=p.info['create_time'], rss_bytes=p.info['memory_info'].rss,
            private_bytes=None, page_faults_total=None)
            for p in psutil.process_iter(['name','ppid','create_time','memory_info'])], []
    size = ctypes.c_size_t
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),
            ('pid',wintypes.DWORD),('heap',size),('module',wintypes.DWORD),
            ('threads',wintypes.DWORD),('parent',wintypes.DWORD),
            ('priority',wintypes.LONG),('flags',wintypes.DWORD),('name',wintypes.WCHAR*260)]
    class Memory(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('faults',wintypes.DWORD)] + [
            (key,size) for key in ('peak_rss','rss','peak_paged','paged',
                'peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    snapshot=kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows=[];unavailable=[]
    try:
        entry=Entry();entry.dwSize=ctypes.sizeof(entry)
        valid=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
        while valid:
            handle=kernel.OpenProcess(0x1000,False,entry.pid)
            try:
                mem=Memory();mem.cb=ctypes.sizeof(mem)
                if handle and api.GetProcessMemoryInfo(handle,ctypes.byref(mem),mem.cb):
                    try:created=psutil.Process(entry.pid).create_time()
                    except psutil.Error:created=None
                    rows.append(dict(pid=int(entry.pid),name=entry.name,parent=int(entry.parent),
                        created_at=created,rss_bytes=int(mem.rss),private_bytes=int(mem.private),
                        page_faults_total=int(mem.faults)))
                else:
                    unavailable.append(dict(pid=int(entry.pid),name=entry.name,
                        winerror=ctypes.get_last_error()))
            finally:
                if handle:kernel.CloseHandle(handle)
            valid=kernel.Process32NextW(snapshot,ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return rows,unavailable


class ResourceMonitor:
    def __init__(self, mode, directory=OUT, interval=1.0):
        self.mode = mode; self.interval = interval
        self.started = time.monotonic(); self.wall_started = time.time()
        self.process = psutil.Process(); self.pid = self.process.pid
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.directory = Path(directory) / f'manual-{mode}-{stamp}-{self.pid}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / 'resources.jsonl'
        self.events = queue.Queue(maxsize=512)
        self.lock = threading.Lock(); self.stop_event = threading.Event()
        self.error = ''; self.dropped_events = 0; self.latest = {}
        self.peaks = {}; self.process_growth = {}; self.stages = []; self.underruns = []
        self.producer_waits = []
        self.activity = dict(stt=False, qwen=False, tts=False, playback=False)
        self._gpu_latest = {'unavailable': 'not sampled yet'}
        self._gpu_due = 0.0
        self._process_cpu = {}
        self._process_latest = []; self._process_errors_latest = []
        self._process_due = 0.0; self._process_interval = 5.0
        self._gc_events = queue.SimpleQueue(); self._gc_callback_installed = False
        self._gc_callback = self._capture_gc_event
        psutil.cpu_percent(None, percpu=True)
        self.process.cpu_percent(None)
        self.thread = threading.Thread(target=self._run, name='anydes-ram-monitor', daemon=True)

    def snapshot(self):
        vm = psutil.virtual_memory(); mem = self.process.memory_info()
        row = {'system_used_bytes': vm.total - vm.available, 'system_available_bytes': vm.available,
            'system_total_bytes': vm.total, 'process_rss_bytes': mem.rss,
            'process_private_bytes': getattr(mem, 'private', None),
            'system_cpu_percent': psutil.cpu_percent(None),
            'per_cpu_percent': psutil.cpu_percent(None, percpu=True),
            'process_cpu_percent': self.process.cpu_percent(None),
            'process_threads': self.process.num_threads()}
        try:
            switches = self.process.num_ctx_switches()
            row['process_context_switches'] = {
                'voluntary': switches.voluntary,
                'involuntary': switches.involuntary,
            }
        except (psutil.Error, OSError, AttributeError):
            row['process_context_switches'] = None
        try:
            row.update(windows_memory())
        except Exception as exc:
            row['windows_memory_error'] = f'{type(exc).__name__}: {exc}'
        return row

    def mark(self, kind, source_at=None, **fields):
        """Cheap boundary snapshot on control/worker threads; disk writes are queued."""
        observed = time.monotonic()
        with self.lock:
            if kind in {'input_stt_loading','input_transcribing','input_transcription_started'}:
                self.activity['stt'] = True
            elif kind in {'input_transcript','input_hallucination_suspected','input_error'}:
                self.activity['stt'] = False
            elif kind == 'reasoning_started':
                self.activity['qwen'] = True
            elif kind == 'reasoning_finished':
                self.activity['qwen'] = False
            elif kind == 'output_synthesis_started':
                self.activity['tts'] = True
            elif kind == 'output_playback_started':
                self.activity['playback'] = True
            elif kind in {'output_playback_stopped','output_playback_interrupted','output_voice_failed'}:
                self.activity['tts'] = False; self.activity['playback'] = False
            activity = dict(self.activity)
        row = dict(kind=kind, at=observed, source_at=source_at,
            elapsed_s=observed-self.started,
            wall_time_utc=datetime.now(timezone.utc).isoformat(), activity=activity, **fields)
        # Event producers include the UI, TTS and playback threads.  Reusing the
        # latest monitor sample keeps instrumentation away from their hot path.
        if kind == 'ram_initial_before_runtime':
            try:
                row['memory'] = self.snapshot()
            except Exception as exc:
                row['memory_error'] = f'{type(exc).__name__}: {exc}'
        else:
            with self.lock:
                cached = self.latest
            if cached:
                row['memory'] = dict(cached.get('memory', {}))
                row['gpu'] = dict(cached.get('gpu', {}))
                row['paging'] = dict(cached.get('paging', {}))
                row['telemetry_sample_age_ms'] = max(0.0, (observed-cached['at'])*1000)
        try:
            self.events.put_nowait(row)
        except queue.Full:
            self.dropped_events += 1
            self.error = 'Cola de telemetria llena: registro incompleto.'

    def start(self):
        if self._gc_callback not in gc.callbacks:
            gc.callbacks.append(self._gc_callback)
            self._gc_callback_installed = True
        self.mark('ram_initial_before_runtime', mode=self.mode,
            monotonic_clock=time.get_clock_info('monotonic').implementation,
            perf_counter_clock=time.get_clock_info('perf_counter').implementation)
        self.thread.start()
        return self

    def _capture_gc_event(self, phase, info):
        # Python can invoke GC callbacks in arbitrary application threads.  Do
        # not inspect memory or write files here; the monitor drains this queue.
        self._gc_events.put((time.monotonic(), phase, dict(info)))

    def _drain_gc_events(self, handle):
        while True:
            try:
                observed, phase, info = self._gc_events.get_nowait()
            except queue.Empty:
                return
            with self.lock:
                activity = dict(self.activity)
            self._write(handle, dict(
                kind=f'gc_{phase}', at=observed, source_at=observed,
                elapsed_s=observed-self.started,
                wall_time_utc=datetime.now(timezone.utc).isoformat(),
                activity=activity, generation=info.get('generation'),
                collected=info.get('collected'), uncollectable=info.get('uncollectable'),
            ))

    def _processes(self):
        rows,errors=process_memory_rows();descendants={self.pid}
        while True:
            expanded=descendants | {row['pid'] for row in rows if row['parent'] in descendants}
            if expanded==descendants:break
            descendants=expanded
        for row in rows:
            row['tracked']=row['pid'] in descendants or any(
                term in row['name'].lower() for term in ('ollama','qtwebengine'))
            key = f"{row['pid']}:{row['created_at']}"
            baseline = self.process_growth.setdefault(key, dict(pid=row['pid'], name=row['name'],
                tracked=row['tracked'],
                first_seen_elapsed_s=time.monotonic()-self.started, initial_rss_bytes=row['rss_bytes'],
                peak_rss_bytes=row['rss_bytes'], initial_private_bytes=row['private_bytes'],
                peak_private_bytes=row['private_bytes']))
            baseline['peak_rss_bytes'] = max(baseline['peak_rss_bytes'], row['rss_bytes'])
            baseline['tracked'] = bool(baseline.get('tracked') or row['tracked'])
            row['rss_delta_since_first_seen_bytes'] = row['rss_bytes']-baseline['initial_rss_bytes']
            if row['private_bytes'] is not None:
                baseline['peak_private_bytes'] = max(baseline['peak_private_bytes'], row['private_bytes'])
                row['private_delta_since_first_seen_bytes'] = row['private_bytes']-baseline['initial_private_bytes']
            try:
                proc = self._process_cpu.get(row['pid'])
                if proc is None or proc.create_time() != row['created_at']:
                    proc = psutil.Process(row['pid']); proc.cpu_percent(None)
                    self._process_cpu[row['pid']] = proc
                    row['cpu_percent'] = 0.0
                else:
                    row['cpu_percent'] = proc.cpu_percent(None)
            except (psutil.Error, OSError):
                row['cpu_percent'] = None
        top = {row['pid'] for row in sorted(rows, key=lambda r:r['rss_bytes'], reverse=True)[:10]}
        return [row for row in rows if row['tracked'] or row['pid'] in top], errors

    def _write(self, handle, row):
        handle.write(json.dumps(row, ensure_ascii=True)+'\n'); handle.flush()
        memory = row.get('memory', {})
        for key in ('system_used_bytes', 'process_rss_bytes', 'process_private_bytes', 'commit_bytes'):
            if memory.get(key) is not None:
                self.peaks[key] = max(self.peaks.get(key, 0), memory[key])
        if row['kind'] == 'sample':
            with self.lock:
                self.latest = row
        else:
            self.stages.append(row)
            if row['kind'] == 'output_audible_underrun':
                self.underruns.append(row)
            elif row['kind'] in {'output_producer_wait', 'output_queue_underrun'}:
                self.producer_waits.append(row)

    def _run(self):
        paging = None; paging_error = ''
        try:
            paging = PagingRates()
        except Exception as exc:
            paging_error = f'{type(exc).__name__}: {exc}'
        try:
            with self.path.open('w', encoding='utf-8') as handle:
                due = time.monotonic()
                while not self.stop_event.is_set() or not self.events.empty():
                    self._drain_gc_events(handle)
                    try:
                        self._write(handle, self.events.get(timeout=.05))
                    except queue.Empty:
                        pass
                    if time.monotonic() >= due:
                        started = time.monotonic()
                        if started >= self._gpu_due:
                            self._gpu_latest = nvidia_snapshot()
                            self._gpu_due = time.monotonic() + 2.0
                        with self.lock:
                            activity = dict(self.activity)
                        row = dict(kind='sample', at=started, elapsed_s=started-self.started,
                            wall_time_utc=datetime.now(timezone.utc).isoformat(), memory=self.snapshot(),
                            gpu=dict(self._gpu_latest), activity=activity)
                        inventory_fresh = started >= self._process_due
                        if inventory_fresh:
                            self._process_latest, self._process_errors_latest = self._processes()
                            self._process_due = time.monotonic() + self._process_interval
                        row['processes'] = self._process_latest
                        row['process_errors'] = self._process_errors_latest
                        row['process_inventory_fresh'] = inventory_fresh
                        row['process_inventory_interval_s'] = self._process_interval
                        row['paging'] = paging.sample() if paging else {'unavailable': paging_error}
                        row['sampling_ms'] = (time.monotonic()-started)*1000
                        row['event_queue_drops'] = self.dropped_events
                        self._write(handle, row)
                        due = time.monotonic()+self.interval
        except Exception as exc:
            self.error = f'Registro RAM detenido: {type(exc).__name__}: {exc}'
        finally:
            if paging:
                paging.close()

    def close(self):
        if self.stop_event.is_set():
            return not self.thread.is_alive()
        if self._gc_callback_installed:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass
            self._gc_callback_installed = False
        self.mark('runtime_closed'); self.stop_event.set(); self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = 'El registro RAM no se cerro a tiempo.'
            return False
        growth = sorted(self.process_growth.values(), key=lambda r:
            r['peak_rss_bytes']-r['initial_rss_bytes'], reverse=True)
        summary = dict(mode=self.mode, session_directory=str(self.directory),
            peaks_bytes=self.peaks, process_growth=growth, stages=self.stages,
            underruns=self.underruns, producer_waits=self.producer_waits,
            telemetry_error=self.error, dropped_events=self.dropped_events,
            notes=['RSS includes shared pages; do not add process RSS to estimate system use.',
                'Process page_faults_total includes soft faults; not per-process hard faults.',
                'System page reads can include mapped files, not only the pagefile.',
                'Protected/inaccessible processes are listed under process_errors; their memory is unknown.',
                'Boundary events reuse the latest sample to avoid blocking UI/TTS/playback threads.',
                'Process attribution is refreshed every 5s; system samples retain a 1s default cadence.',
                'Producer waits are not audible underruns unless DAC diagnostics report starvation.'])
        try:
            (self.directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        except OSError as exc:
            self.error = f'No se pudo guardar resumen RAM: {exc}'
            return False
        return not bool(self.error)


_monitor = None


def get_monitor():
    return _monitor


def start_monitor(mode, directory=None):
    global _monitor
    if _monitor is not None:
        raise RuntimeError('Ya existe un monitor RAM en esta sesion.')
    _monitor = ResourceMonitor(mode, OUT if directory is None else directory).start()
    return _monitor


def mark(kind, **fields):
    if _monitor is not None:
        _monitor.mark(kind, **fields)
