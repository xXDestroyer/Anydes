"""Validation only: freeze sources or launch the real graphical staging bootstrap."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/phase_g/candidate_f'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes():
    baseline = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())
    paths = {ROOT / p for p in baseline['official_sha256']}
    for folder in ('data', 'datos', 'programa_principal/data', 'programa_principal/datos'):
        directory = ROOT / folder
        if directory.exists():
            paths.update(p for p in directory.rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'freeze.json').exists():
        raise RuntimeError('candidate_already_frozen')
    files = []
    for p in (ROOT / 'anydes_tools').rglob('*'):
        if p.is_file() and p.suffix in {'.py', '.json', '.toml', '.yaml', '.yml'}:
            if not {'logs', 'workspaces', 'quarantine', '__pycache__', 'runtime_home', 'runtime_temp', 'executions_b2'}.intersection(p.parts):
                files.append(p)
    files += [ROOT / 'probar_anydes_tools_conversacion_v001_5.py', ROOT / 'Probar Anydes Tools Conversacion v001.5.bat']
    snapshot = OUT / 'snapshot'
    hashes = {}
    for p in sorted(files):
        rel = p.relative_to(ROOT)
        hashes[str(rel)] = sha(p)
        # Configuration is fingerprinted only, never copied into reports.
        if p.suffix in {'.py', '.bat'}:
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    restored = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file()}
    manifest = {'candidate': 'F-frozen-for-G', 'git_commit': None,
        'source_sha256': hashes, 'protected_before': protected_hashes(),
        'snapshot_restore_hash_match': all(hashes[k] == h for k, h in restored.items()),
        'config_policy': 'Hashes only; secrets never copied', 'model_primary': 'openai/gpt-oss-120b',
        'model_fallback': 'qwen3:8b', 'sandbox_backend': 'LPAC_ACL_JOB_OBJECTS',
        'oracle_versions': {'units.affine': 1, 'text.character_count': 1}}
    (OUT / 'freeze.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'candidate_files': len(hashes), 'protected_files': len(manifest['protected_before']),
        'snapshot_restore_hash_match': manifest['snapshot_restore_hash_match']}))


def gui():
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['ANYDES_DATA_DIR'] = str(OUT / 'isolated_data')
    from anydes_tools.tests.runtime_comparison import observe_human
    import probar_anydes_tools_conversacion_v001_5 as bootstrap

    def destinations_only():
        os.environ['ANYDES_HYBRID_LOG_DIR'] = str(OUT / 'hybrid')
        os.environ['ANYDES_COGNITIVE_V2_LOG_DIR'] = str(OUT / 'continuity')
        import anydes.voice_input.environment as environment
        import anydes.voice_input.telemetry as telemetry
        environment.OUT = OUT / 'voice_events'
        telemetry.OUT = environment.OUT

    return observe_human(lambda: bootstrap.main(before_layers=destinations_only))


def check():
    old = json.loads((OUT / 'freeze.json').read_text())
    current = protected_hashes()
    changed = [k for k, h in old['protected_before'].items() if current.get(k) != h]
    sources = [k for k, h in old['source_sha256'].items() if sha(ROOT / k) != h]
    result = {'protected_changed': changed, 'candidate_source_changed': sources,
        'protected_after': current}
    (OUT / 'after.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'protected_changed': changed, 'candidate_source_changed': sources}))


if __name__ == '__main__':
    if sys.argv[1] == 'freeze':
        freeze()
    elif sys.argv[1] == 'gui':
        raise SystemExit(gui())
    elif sys.argv[1] == 'check':
        check()
