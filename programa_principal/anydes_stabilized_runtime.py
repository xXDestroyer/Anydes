"""H0 production wiring for the accepted runtime; no staging bootstrap/data."""
from functools import wraps
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / 'production_tools'
OUT = ROOT / 'programa_principal/datos/telemetria/stabilized_runtime'
_lock = threading.RLock()
_installed = False


def record(event, **details):
    entry = dict(event=event, timestamp=datetime.now(timezone.utc).isoformat(),
                 monotonic=time.monotonic(), pid=os.getpid(), **details)
    try:
        with _lock:
            OUT.mkdir(parents=True, exist_ok=True)
            with (OUT/'lifecycle.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=True)+'\n')
    except OSError:
        import logging
        logging.getLogger(__name__).error('production_lifecycle_observation_unavailable')


def install():
    global _installed
    if _installed:
        return
    for name in ('ANYDES_PHASE_A1_OBSERVE','ANYDES_SETTINGS_V001_5_SMOKE',
                 'ANYDES_HYBRID_FAULT_PLAN','ANYDES_STAGING_DATA_ROOT'):
        value = os.getenv(name, '').strip().casefold()
        if value not in {'','0','false','no','off'}:
            raise RuntimeError('PRODUCTION_REJECTS_TEST_ENVIRONMENT:' + name)
    configured = os.getenv('ANYDES_DATA_DIR','').strip()
    if configured and Path(configured).resolve().is_relative_to((ROOT/'anydes_tools').resolve()):
        raise RuntimeError('PRODUCTION_REJECTS_CANDIDATE_DATA')
    sys.dont_write_bytecode = True
    os.environ['ANYDES_RUNTIME_KIND'] = 'PRODUCTION'
    for path in (str(ROOT),str(ROOT/'programa_principal')):
        if path not in sys.path:
            sys.path.insert(0,path)
    from anydes_tools.conversation import runtime_patch
    from anydes_tools.settings.store import settings_for
    settings = settings_for(ROOT, TOOLS)
    factory = runtime_patch.ToolConversationController

    @wraps(factory)
    def production_controller(project_root, *args, **kwargs):
        if Path(project_root).resolve()==ROOT and not args and kwargs.get('tools_root') is None:
            kwargs.update(tools_root=TOOLS, settings=kwargs.get('settings') or settings)
        return factory(project_root,*args,**kwargs)

    runtime_patch.ToolConversationController = production_controller
    from anydes_tools.ui_lifecycle import install as install_lifecycle
    install_lifecycle(record)
    runtime_patch.install(ROOT)
    if any(name.startswith('anydes_tools.tests') for name in sys.modules):
        raise RuntimeError('PRODUCTION_REJECTS_TEST_MODULES')
    import anydes.voice_input.integration as voice
    from anydes_tools.settings import runtime_patch as settings_ui
    previous_settings = settings_ui.settings_for

    def production_settings(project_root, tools_root=None):
        if Path(project_root).resolve()==ROOT:
            return settings
        return previous_settings(project_root,tools_root)

    settings_ui.settings_for = production_settings
    original_install = voice.install

    def install_voice_and_settings(module, performance=False, panel_factory=None):
        base_panel=panel_factory or voice.VoicePanel
        original_install(module, performance=performance,
            panel_factory=settings_ui.make_settings_managed_panel(base_panel))
        settings_ui.install_settings_window(module, ROOT, performance=performance)

    voice.install = install_voice_and_settings
    _installed = True
    record('production_runtime_installed', release='official_stabilized_H0_20261017',
        tools_state='production_tools', official_personal_data_preserved=True,
        candidate_data_loaded=False, test_hooks_loaded=False)


def record_runtime_return():
    record('event_loop_return')
