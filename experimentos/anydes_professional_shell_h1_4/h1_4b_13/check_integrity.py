"""Read existing baselines; never refresh production or accepted source hashes."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def check():
    results={}
    for name in ('protected_before','frozen_before'):
        data=json.loads((HERE.parent/f'evidence/{name}.json').read_text())
        bad=[]
        for path,expected in data.items():
            target=Path(path) if Path(path).is_absolute() else ROOT/path
            if not target.is_file() or sha(target)!=expected:
                bad.append(path)
        results[name]={'count':len(data),'violations':bad}
    data=json.loads((HERE.parent/'evidence/source_manifest.json').read_text())['files']
    results['professional_shell']={'count':len(data),'violations':[
        p['path'] for p in data if sha(HERE.parent/p['path'])!=p['sha256']]}
    freeze=HERE/'evidence/candidate_freeze.json'
    current={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')}
    if '--freeze' in sys.argv:
        if freeze.exists():
            raise RuntimeError('candidate_already_frozen')
        freeze.parent.mkdir(exist_ok=True)
        freeze.write_text(json.dumps(current,indent=2),encoding='utf-8')
    if freeze.exists():
        data=json.loads(freeze.read_text())
        results['candidate_freeze']={'count':len(data),'violations':[
            p for p,h in data.items() if current.get(p)!=h]}
    label='before' if '--freeze' in sys.argv else 'after'
    (HERE/f'evidence/integrity_{label}.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))
    return int(any(r['violations'] for r in results.values()))


if __name__=='__main__':
    raise SystemExit(check())
