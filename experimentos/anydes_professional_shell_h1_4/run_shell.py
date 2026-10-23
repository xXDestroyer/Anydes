"""H1.4A: reuse the accepted isolated runtime, add presentation at its UI seam."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT/'experimentos/anydes_presence_lab_v001'
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(LAB/'h1_3'), str(ROOT/'programa_principal'), str(ROOT)]
spec = importlib.util.spec_from_file_location('presence_motion_host', LAB/'h1_3/run_staging.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

SESSION = HERE/'sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
stage.host.SESSION = SESSION
services = []
class ShellService(stage.VoiceMotionService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        services.append(self)
stage.event_service.PresenceService = ShellService

from anydes.official_runtime import integration
original_run = integration.run_mode
def shell_run(*args, **kwargs):
    import anydes.voice_input.integration as voice
    original_install = voice.install
    def install(module, *args, **kwargs):
        result = original_install(module, *args, **kwargs)
        from shell import install_shell
        install_shell(module, SESSION, services[0])
        return result
    voice.install = install
    return original_run(*args, **kwargs)
integration.run_mode = shell_run

if __name__ == '__main__':
    raise SystemExit(stage.host.main())
