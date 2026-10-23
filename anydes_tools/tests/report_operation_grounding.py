"""Post-attempt metadata only. Not part of the candidate execution path."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'anydes_tools/logs/operation_grounding'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def main():
    candidate=OUT/'candidate'
    life=rows(candidate/'lifecycle.jsonl')
    audit=rows(candidate/'data/audit.log')
    tool=rows(candidate/'telemetry/tools/conversation-v001_5.jsonl')
    writes=rows(candidate/'host_writes.jsonl')
    counts=Counter(r['event'] for r in life)
    phases=[r for r in tool if r['event']=='phase_c_turn']
    freeze=json.loads((OUT/'freeze.json').read_text())
    after=json.loads((OUT/'after.json').read_text())
    official=json.loads((ROOT/'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    value={'human_turns':len(phases),'task_completion_rate':None,
        'human_sequence_status':'NOT_REACHED_UI_RUNTIME_LIFECYCLE_CRASH',
        'incident_classification':'UI_RUNTIME_QTHREAD_LIFECYCLE_CRASH',
        'primary_event_assessment':'PROBABLE_OBSERVER_INITIAL_SNAPSHOT_FILE_STAT_RACE_NOT_HISTORICALLY_STACK_CONFIRMED',
        'secondary_fatal':'QThread: Destroyed while thread is still running',
        'startup_user_confirmation':{
            'closed_manually':False,
            'missing_json_dialog_reported':True,
            'missing_json_filename':None,
            'filename_not_recalled':True,
            'ui_was_open_before_spontaneous_close':True,
            'json_error_causal_link_to_qthread_abort':'SUPPORTED_HYPOTHESIS_NOT_DIRECTLY_CAPTURED'},
        'ui_stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8').strip(),
        'lifecycle_event_counts':dict(counts),
        'worker_starts':counts['qthread_start'],'worker_finished':counts['qthread_finished'],
        'worker_destroyed':counts['qthread_destroyed'],
        'normal_event_loop_return':any(r.get('phase')=='EVENT_LOOP_RETURN' for r in life),
        'last_qt_snapshot':next((r for r in reversed(life) if r['event']=='qt_snapshot'),None),
        'groq_statuses':dict(Counter(r.get('status') for r in life if r['event']=='groq_http_response')),
        'fallback_observations':counts['fallback_request'],
        'publications':sum(r['event']=='final_publication' for r in audit),
        'memory_retrievals':sum(r['event']=='conversation_memory_retrieval' for r in audit),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'write_violations':[r for r in writes if r['decision'] != 'OBSERVED'],
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'],'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'official_count':len(official),
        'freeze_sha256':hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        'sandbox_temps':{str(p.relative_to(ROOT)):len(list(p.rglob('*'))) for p in
            (ROOT/'anydes_tools/sandbox/executions_b2',ROOT/'anydes_tools/sandbox/runtime_temp',ROOT/'anydes_tools/sandbox/runtime_home')},
        'final_regression':{'pass':224,'skip':1,'fail':1,'total':226},
        'earlier_regression':{'pass':222,'skip':1,'fail':0,'total':223},
        'single_b2_cpp_diagnostic':{'pass':1,'fail':0},
        'recommendation':'CONVERSATIONAL ACCEPTANCE NOT READY'}
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('human_turns','worker_starts','worker_finished',
        'worker_destroyed','normal_event_loop_return','write_decisions','official_matches',
        'protected_count','protected_changes','source_changes_after_freeze','freeze_sha256')}))


if __name__=='__main__':
    main()
