"""Freeze/check/GUI. UI branch uses the normal staging launcher initialization."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'anydes_tools/logs/phase_g2'
SOURCES = ['probar_anydes_tools_conversacion_v001_5.py', 'anydes_tools/runtime_environment.py',
    'anydes_tools/conversation/planner.py', 'anydes_tools/conversation/structured_text.py',
    'anydes_tools/conversation/executable_plan.py', 'anydes_tools/registry/semantic_operations.json',
    'anydes_tools/tests/observe_candidate_g2.py', 'anydes_tools/tests/validate_candidate_phase_g2.py',
    'anydes_tools/tests/test_candidate_blockers_g2.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from anydes_tools.tests.validate_candidate_phase_g import protected_hashes
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == 'freeze':
        if (OUT / 'freeze.json').exists():
            raise RuntimeError('freeze_already_exists')
        snapshot = OUT / 'snapshot'
        for rel in SOURCES:
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        (OUT / 'freeze.json').write_text(json.dumps({'source_sha256':{p:sha(ROOT/p) for p in SOURCES},
            'protected_before':protected_hashes(), 'historical_vram_state':'HISTORICAL STATE NOT VERIFIABLE',
            'sandbox_backend':'STRONG_WINDOWS_SANDBOX', 'no_mocks':True}, indent=2))
        print('G2 freeze recorded')
    elif mode == 'check':
        before = json.loads((OUT / 'freeze.json').read_text())
        current = protected_hashes()
        result = {'changed':[p for p,h in before['protected_before'].items() if current.get(p)!=h],
            'new_protected_files':sorted(set(current)-set(before['protected_before'])),
            'changed_sources':[p for p,h in before['source_sha256'].items() if sha(ROOT/p)!=h],
            'protected_after':current}
        (OUT / 'after.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k:result[k] for k in ('changed','new_protected_files','changed_sources')}))
    elif mode in {'gui', 'diagnostic'}:
        destination = OUT / ('candidate' if mode == 'gui' else 'diagnostic')
        destination.mkdir(exist_ok=True)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['ANYDES_RUNTIME_KIND'] = 'CANDIDATE'
        os.environ['ANYDES_STAGING_DATA_ROOT'] = str(destination)
        os.environ['ANYDES_RUNTIME_WRITE_TRACE'] = '1'
        os.chdir(ROOT)
        import probar_anydes_tools_conversacion_v001_5 as bootstrap
        from anydes_tools.tests.observe_candidate_g2 import install
        try:
            return bootstrap.main(before_layers=lambda: install(destination))
        except BaseException as error:
            (destination / 'startup_error.json').write_text(json.dumps({'type':type(error).__name__,
                'message':str(error)[:180]}))
            raise


if __name__ == '__main__':
    raise SystemExit(main() or 0)
