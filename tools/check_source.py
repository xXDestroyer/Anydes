"""Offline syntax preflight. Does not import the application or create bytecode."""
from pathlib import Path
import ast
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]


def check():
    counts = {"python": 0, "json": 0}
    failures = []
    notes = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(x in path.parts for x in ('node_modules', '.git', '__pycache__')):
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            if path.suffix == '.py':
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter('always')
                    ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
                counts['python'] += 1
                notes.extend({'path': relative, 'category': type(w.message).__name__} for w in recorded)
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8-sig'))
                counts['json'] += 1
        except (SyntaxError, ValueError, UnicodeError) as exc:
            failures.append({'path': relative, 'type': type(exc).__name__, 'line': getattr(exc, 'lineno', None)})
    return dict(counts=counts, failures=failures, warnings=notes, application_imported=False,
                bytecode_written=False)


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(bool(result['failures']))
