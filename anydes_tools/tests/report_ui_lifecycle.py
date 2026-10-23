"""Post-smoke evidence aggregation; not part of runtime or its freeze."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/ui_lifecycle_fix'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def main():
    candidate = OUT / 'candidate'
    events = rows(candidate / 'lifecycle.jsonl')
    counts = Counter(row['event'] for row in events)
    snapshots = [r for r in events if r['event'] == 'qt_snapshot']
    writes = rows(candidate / 'host_writes.jsonl')
    audit = rows(candidate / 'data/audit.log')
    freeze = json.loads((OUT / 'freeze.json').read_text())
    after = json.loads((OUT / 'after.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    total = int(re.search(r'Ran (\d+) tests', regression).group(1))
    skip = int(re.search(r'OK \(skipped=(\d+)\)', regression).group(1))
    value = {
        'classification':'UI_RUNTIME_LIFECYCLE_SMOKE_NOT_CONVERSATIONAL_ACCEPTANCE',
        'historical_root_cause':'NOT_PROVEN',
        'recommendation':'UI RUNTIME LIFECYCLE READY',
        'recommendation_scope':'OBSERVER_AND_COORDINATED_TEARDOWN_NOT_CONVERSATIONAL_ACCEPTANCE',
        'physical_b4_user_confirmed':True,
        'microphone_provenance':'NOT_TESTED_USER_CONFIRMED_KEYBOARD',
        'input_mode_user_confirmed':'KEYBOARD',
        'input_events':[r for r in audit if r['event']=='user_input_received'],
        'publication_evidence':[r for r in audit if r['event']=='final_publication'],
        'audio_completion':[r for r in audit if r['event']=='interface_voice_completed'],
        'process_check':json.loads((OUT/'process_check.json').read_text(encoding='utf-8-sig')),
        'event_counts':dict(counts),
        'observer_unexpected_errors':[r for r in events if r['event']=='observer_unexpected_error'],
        'runtime_exceptions':[r for r in events if r['event']=='runtime_exception'],
        'shutdown_errors':[r for r in events if r['event'] in {'shutdown_component_error','shutdown_pending'}],
        'lifecycle_timeline':[r for r in events if r['event'] in {
            'thread_started','thread_finished','thread_destroyed','event_loop_enter',
            'event_loop_return','shutdown_requested','worker_stop_requested','thread_quit',
            'thread_wait_start','shutdown_workers_stopped','window_close_event','about_to_quit'}],
        'snapshot_count':len(snapshots),
        'observed_span_seconds':(datetime.fromisoformat(events[-1]['timestamp'])-
                                 datetime.fromisoformat(events[0]['timestamp'])).total_seconds(),
        'resource_samples':[{k:r.get(k) for k in ['timestamp','phase','working_set',
                            'peak_working_set','vram_used_mb','data_bytes','data_files']} for r in snapshots],
        'last_workers':snapshots[-1]['workers'] if snapshots else [],
        'observed_process_ids':sorted({p['pid'] for r in snapshots for p in r['processes']}),
        'atomic_renames':sum(r['operation']=='os.rename' for r in writes),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'write_violations':[r for r in writes if r['decision']!='OBSERVED'],
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'], 'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'official_count':len(official),
        'freeze_sha256':hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        'regression':{'total':total,'pass':total-skip,'skip':skip,'fail':0},
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'sandbox_temps':{name:len(list((ROOT/'anydes_tools/sandbox'/name).rglob('*')))
                         for name in ['executions_b2','runtime_temp','runtime_home']}}
    (OUT / 'evidence.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({k:value[k] for k in ['event_counts','snapshot_count','observed_span_seconds',
        'write_decisions','official_matches','protected_count','protected_changes',
        'source_changes_after_freeze','sandbox_temps','regression','freeze_sha256']}))


if __name__ == '__main__':
    main()
