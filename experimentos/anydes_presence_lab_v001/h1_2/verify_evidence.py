"""Summarize only hashes and lifecycle metadata; never copy conversation text."""
from pathlib import Path
import collections
import hashlib
import json

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parents[1]
SESSION = HERE / 'sessions/20261018T181955'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    before = json.loads((HERE/'evidence/production_before.json').read_text())
    freeze = json.loads((HERE/'evidence/ui_freeze.json').read_text())
    lifecycle = [json.loads(line) for line in (SESSION/'lifecycle.jsonl').read_text().splitlines()]
    presence = [json.loads(line) for line in (SESSION/'presence/presence_events.jsonl').read_text().splitlines()]
    result = {
        'protected_total': len(before),
        'protected_changed': [p for p, sha in before.items() if digest(ROOT/p) != sha],
        'source_changes_after_session_close': [p for p, sha in freeze.items() if digest(LAB/p) != sha],
        'post_close_change_reason': 'Native browser timer receiver fix; no changes during submitted turn.',
        'lifecycle_counts': dict(collections.Counter(row['event'] for row in lifecycle)),
        'observer_shutdown': [row for row in presence if row['kind'] == 'observer_shutdown'],
        'observer_failures': [row['observer_failures'] for row in lifecycle if row['event'] == 'staging_return'],
        'violation_files': [str(p.relative_to(SESSION)) for p in SESSION.rglob('*violations*')],
        'execution_temp_file_count': sum(p.is_file() for p in (SESSION/'tools/sandbox/runtime_temp').rglob('*')),
        'live_ui': json.loads((HERE/'evidence/live_ui/summary.json').read_text()),
        'post_close_browser_check': json.loads((HERE/'evidence/browser_watchdog_check.json').read_text()),
        'physical_mic_test': 'NOT_REACHED_USER_UNAVAILABLE',
        'verdict': 'PRESENCE RUNTIME STATE ADAPTER NOT READY',
    }
    (HERE/'evidence/final_integrity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (HERE/'evidence/current_source_manifest.json').write_text(json.dumps({p: digest(LAB/p) for p in freeze}, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
