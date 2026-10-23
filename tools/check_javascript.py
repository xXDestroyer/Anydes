"""Optional syntax-only Node check, without starting servers or installing packages."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', default='node')
    args = parser.parse_args()
    failures = []
    files = sorted(p for p in ROOT.rglob('*') if p.suffix in ('.js', '.mjs'))
    for path in files:
        result = subprocess.run([args.node, '--check', str(path)], capture_output=True,
                                text=True, timeout=30, cwd=ROOT)
        if result.returncode:
            # Syntax error payloads can echo source; retain only a safe file/type summary.
            failures.append(dict(path=path.relative_to(ROOT).as_posix(), exit_code=result.returncode))
    report = dict(files=len(files), failures=failures, ports_opened=0, dependencies_installed=False)
    (ROOT / 'audit/javascript_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
