"""Dedicated staging host; never calls the production bootstrap or launcher."""
from pathlib import Path
import json
import os
import shutil
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
PROGRAM = ROOT/'programa_principal'
SESSION = HERE/'sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(PROGRAM), str(HERE)]


def main():
    import anydes_hybrid_runtime as hybrid
    hybrid._verify_release()
    from lanzar_anydes import load_profile, OFFICIAL_PROFILE, OFFICIAL_VISUAL_PROFILE
    load_profile(OFFICIAL_PROFILE); load_profile(OFFICIAL_VISUAL_PROFILE)
    hybrid._load_non_secret_profile(); hybrid._load_secret()
    SESSION.mkdir(parents=True)
    os.environ['ANYDES_RUNTIME_KIND'] = 'STAGING'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['ANYDES_STAGING_DATA_ROOT'] = str(SESSION)
    from anydes_tools.runtime_environment import RuntimeEnvironment, activate
    environment = RuntimeEnvironment('STAGING', ROOT, SESSION)
    activate(environment)
    def guard(event, args):
        targets = []
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1:3]
            if (mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                targets = [args[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}: targets = [args[0]]
        elif event in {'os.rename','os.link','os.symlink'}: targets = args[:2]
        for value in targets:
            if isinstance(value, int): continue
            path = Path(os.fsdecode(value)).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(SESSION):
                with (SESSION/'write_boundary_violations.jsonl').open('a',encoding='utf-8') as log:
                    import hashlib
                    log.write(json.dumps(dict(operation=event,path_sha256=hashlib.sha256(str(path).encode()).hexdigest()))+'\n')
                raise PermissionError('PRESENCE_STAGING_WRITE_BOUNDARY')
    sys.addaudithook(guard)
    os.environ['ANYDES_HYBRID_LOG_DIR'] = str(SESSION/'telemetry/hybrid')
    os.environ['ANYDES_COGNITIVE_V2_LOG_DIR'] = str(SESSION/'telemetry/continuity')
    primary = os.environ.get('ANYDES_PRIMARY_GROQ_MODEL','openai/gpt-oss-120b')
    os.environ['PRIMARY_GROQ_MODEL'] = primary
    os.environ['ANYDES_REALTIME_MODEL'] = f'groq:{primary}|fallback:qwen3:8b'
    os.environ['ANYDES_AI_MODEL'] = 'qwen3:8b'
    os.environ['ANYDES_EXPECTED_BRAIN'] = os.environ['ANYDES_REALTIME_MODEL']
    # Re-loading an official nonsecret profile cannot reset isolated destinations.
    import lanzar_anydes
    original_load = lanzar_anydes.load_profile
    def isolated_profile(*args, **kwargs):
        result = original_load(*args, **kwargs); environment.configure(); return result
    lanzar_anydes.load_profile = isolated_profile
    from anydes.voice_input.environment import setup
    setup()
    hybrid._install_hybrid_transport(); hybrid._install_runtime_lease()
    from anydes.continuity_v2.runtime_patch import install as continuity
    continuity()
    from anydes_tools.conversation import runtime_patch
    from anydes_tools.settings.store import settings_for
    tools = SESSION/'tools'
    (tools/'manifests').mkdir(parents=True)
    for manifest in (ROOT/'anydes_tools/manifests').glob('*.json'):
        shutil.copy2(manifest, tools/'manifests'/manifest.name)
    settings = settings_for(ROOT, tools)
    controller_factory = runtime_patch.ToolConversationController
    def staging_controller(project_root, *args, **kwargs):
        if Path(project_root).resolve() == ROOT and not args and kwargs.get('tools_root') is None:
            kwargs.update(tools_root=tools, settings=settings)
        return controller_factory(project_root, *args, **kwargs)
    runtime_patch.ToolConversationController = staging_controller
    def record(event, **details):
        with (SESSION/'lifecycle.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **details))+'\n')
    from anydes_tools.ui_lifecycle import install as lifecycle
    lifecycle(record); runtime_patch.install(ROOT)
    from event_service import PresenceService
    from runtime_observer import RuntimeObserver
    service = PresenceService(SESSION/'presence')
    try: service.start()
    except OSError:
        record('presence_service_unavailable')
    observer = RuntimeObserver(service)
    observer.install()
    import anydes.voice_input.integration as voice
    from anydes_tools.settings import runtime_patch as settings_ui
    prior_settings = settings_ui.settings_for
    settings_ui.settings_for = lambda root, tools_root=None: settings if Path(root).resolve()==ROOT else prior_settings(root,tools_root)
    original_voice = voice.install
    def install_voice(module, performance=False, panel_factory=None):
        original_voice(module, performance=performance,
            panel_factory=settings_ui.make_settings_managed_panel(panel_factory or voice.VoicePanel))
        base = module.PerformanceCompanionWindow if performance else module.AnydesPySideWindow
        class ObservedWindow(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observer.observe_window(self)
        if performance: module.PerformanceCompanionWindow = ObservedWindow
        else: module.AnydesPySideWindow = ObservedWindow
        settings_ui.install_settings_window(module, ROOT, performance=performance)
    voice.install = install_voice
    from anydes.official_runtime import integration
    # Immutable core modules retain their behavior, but all writable destinations are isolated.
    import importlib
    for name, relative in {
        'anydes.voice_input.environment':'telemetry/voice', 'anydes.voice_input.telemetry':'telemetry/voice',
        'anydes.voice_input.ui':'telemetry/voice', 'anydes.stt.parakeet.common':'telemetry/parakeet',
        'anydes.stt.parakeet.live_lab':'telemetry/parakeet', 'anydes.official_runtime.integration':'telemetry/ui',
    }.items():
        target = SESSION/relative; target.mkdir(parents=True,exist_ok=True)
        importlib.import_module(name).OUT = target
    print(json.dumps(dict(session=str(SESSION), pid=os.getpid(), presence_port=service.port)), flush=True)
    record('staging_ready', production_bootstrap_loaded=False, tools_state=str(tools))
    try:
        if '--check' in sys.argv:
            class Assistant: pass
            controller = runtime_patch.ensure_controller_for(Assistant(),ROOT)
            try:
                assert controller.tools_root == tools
                assert controller.settings is settings
                print(json.dumps(dict(check='PASS',production_bootstrap_loaded='anydes_stabilized_runtime' in sys.modules,
                    tools_isolated=True,models_called=False)),flush=True)
            finally: controller.close()
            return 0
        return int(integration.run_mode('normal', target_minutes=0, official_runtime=True) or 0)
    finally:
        service.close()
        from anydes.hybrid.router import close_all_routers
        close_all_routers()
        record('staging_return', observer_failures=observer.failures)


if __name__ == '__main__':
    raise SystemExit(main())
