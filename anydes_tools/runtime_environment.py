"""Explicit host data destinations; no changes to cognition or tool authority."""
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from datetime import datetime, timezone


@dataclass(frozen=True)
class RuntimeEnvironment:
    kind: str
    project: Path
    data_root: Path

    def __post_init__(self):
        if self.kind not in {'PRODUCTION', 'STAGING', 'CANDIDATE', 'TEST'}:
            raise ValueError('invalid_runtime_kind')
        if not self.project.is_absolute() or not self.data_root.is_absolute():
            raise ValueError('explicit_absolute_roots_required')
        object.__setattr__(self, 'project', self.project.resolve())
        object.__setattr__(self, 'data_root', self.data_root.resolve())
        if self.kind != 'PRODUCTION' and self.protected(self.data_root):
            raise ValueError('runtime_root_is_protected')

    def path(self, relative):
        result = (self.data_root / relative).resolve()
        if not result.is_relative_to(self.data_root.resolve()):
            raise ValueError('data_root_escape')
        return result

    def protected(self, path):
        path = Path(path).resolve()
        roots = (self.project / 'programa_principal', self.project / 'data', self.project / 'datos')
        return any(path.is_relative_to(r.resolve()) for r in roots) or path in {
            self.project / '.env', self.project / 'Abrir Anydes.bat'}

    def assert_write(self, path):
        if self.kind != 'PRODUCTION' and self.protected(path):
            raise PermissionError('ENVIRONMENT_PRODUCTION_WRITE_DENIED')

    def configure(self):
        for key, rel in {
            'ANYDES_DATA_DIR': 'data', 'ANYDES_HYBRID_LOG_DIR': 'telemetry/hybrid',
            'ANYDES_COGNITIVE_V2_LOG_DIR': 'telemetry/continuity',
            'ANYDES_VRAM_TTS_METRICS_PATH': 'telemetry/vram_tts_runtime.json',
            'ANYDES_VRAM_EVENT_LOG_PATH': 'telemetry/vram_safety.jsonl',
        }.items():
            os.environ[key] = str(self.path(rel))


_ACTIVE = None
_GUARD_INSTALLED = False
_TRACE_LOCAL = threading.local()
_TRACE_LOCK = threading.RLock()


def write_observation(environment, event, targets, denied):
    if getattr(_TRACE_LOCAL, 'writing', False):
        return
    _TRACE_LOCAL.writing = True
    try:
        def reference(path):
            path = Path(os.fsdecode(path)).resolve()
            if path.is_relative_to(environment.project):
                return '$PROJECT/' + path.relative_to(environment.project).as_posix()
            import hashlib
            return '$EXTERNAL_SHA256/' + hashlib.sha256(str(path).encode()).hexdigest()
        entry = {'timestamp': datetime.now(timezone.utc).isoformat(), 'monotonic': time.monotonic(),
            'pid': os.getpid(), 'thread_id': threading.get_ident(), 'thread': threading.current_thread().name,
            'environment': environment.kind, 'operation': event,
            'requested_paths': [reference(p) for p in targets if not isinstance(p, int)],
            'decision': 'DENY' if denied else 'OBSERVED',
            'reason': 'PRODUCTION_WRITE_DENIED' if denied else 'HOST_WRITE_OBSERVATION'}
        if denied:
            entry['stack'] = [{'module_path': reference(f.filename), 'function': f.name, 'line': f.lineno}
                for f in traceback.extract_stack(limit=24)[:-1]]
        with _TRACE_LOCK:
            with environment.path('environment_violations.jsonl' if denied else 'host_writes.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
    finally:
        _TRACE_LOCAL.writing = False


def activate(environment):
    global _ACTIVE, _GUARD_INSTALLED
    environment.data_root.mkdir(parents=True, exist_ok=True)
    _ACTIVE = environment
    environment.configure()
    if _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True
    def guard(event, args):
        env = _ACTIVE
        if env is None or env.kind == 'PRODUCTION' or getattr(_TRACE_LOCAL, 'writing', False):
            return
        targets = []
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1], args[2]
            if (mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                targets = [args[0]]
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
            targets = [args[0]]
        elif event in {'os.rename', 'os.link', 'os.symlink'}:
            targets = list(args[:2])
        denied = any(not isinstance(p, int) and env.protected(os.fsdecode(p)) for p in targets)
        if targets and (denied or os.getenv('ANYDES_RUNTIME_WRITE_TRACE') == '1'):
            write_observation(env, event, targets, denied)
        if denied:
            raise PermissionError('ENVIRONMENT_PRODUCTION_WRITE_DENIED')
    sys.addaudithook(guard)


def bind_loaded_destinations():
    """Compatibility bindings for immutable official modules with legacy OUT constants."""
    env = _ACTIVE
    if env is None:
        return
    env.configure()
    for name, relative in {
        'anydes.voice_input.environment': 'telemetry/voice',
        'anydes.voice_input.telemetry': 'telemetry/voice',
        'anydes.voice_input.ui': 'telemetry/voice',
        'anydes.stt.parakeet.common': 'telemetry/parakeet',
        'anydes.stt.parakeet.live_lab': 'telemetry/parakeet',
        'anydes.official_runtime.integration': 'telemetry/ui',
    }.items():
        module = importlib.import_module(name)
        module.OUT = env.path(relative)
    from anydes_tools.settings.store import settings_for
    settings = settings_for(env.project)
    settings.settings_path = env.path('settings.json')
    settings.set_temporary_workspace(env.project / 'anydes_tools/workspaces' / ('runtime_' + env.kind.lower()))


def prepare_workspace():
    if _ACTIVE is not None:
        (_ACTIVE.project / 'anydes_tools/workspaces' / ('runtime_' + _ACTIVE.kind.lower())).mkdir(parents=True, exist_ok=True)


def route_tool_audit(path):
    if _ACTIVE is None:
        return path
    return _ACTIVE.path('telemetry/tools') / Path(path).name


def observe_audit(event, details):
    if _ACTIVE is None:
        return
    from anydes_tools.tests.runtime_observation import safe
    entry = {'event': event, 'thread': threading.current_thread().name,
        'thread_id': threading.get_ident(), 'details': safe(details)}
    with _ACTIVE.path('qt_audit.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def install_audit_routing():
    from anydes.nucleo.audit import AuditLog
    from anydes_tools.audit import ToolAuditLog
    for cls in (AuditLog, ToolAuditLog):
        if getattr(cls, '_environment_observed', False):
            continue
        original = cls.record
        def record(self, event, _original=original, **details):
            observe_audit(event, details)
            return _original(self, event, **details)
        cls.record = record
        cls._environment_observed = True
    original_init = ToolAuditLog.__init__
    if not getattr(ToolAuditLog, '_environment_routed', False):
        def init(self, path):
            original_init(self, route_tool_audit(path))
        ToolAuditLog.__init__ = init
        ToolAuditLog._environment_routed = True
