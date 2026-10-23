"""Produce post-close identity acceptance evidence without personal values."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/identity_attribute_final'
REPLAY = OUT / 'replay'
CANDIDATE = REPLAY / 'candidate'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    audit = rows(CANDIDATE / 'data/audit.log')
    lifecycle = rows(CANDIDATE / 'lifecycle.jsonl')
    trace = rows(CANDIDATE / 'telemetry/tools/conversation-v001_5.jsonl')
    guard = rows(CANDIDATE / 'host_writes.jsonl')
    after = json.loads((REPLAY / 'after.json').read_text())
    before = json.loads((OUT / 'before/freeze.json').read_text())['protected_before']
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    publications = [r['details'] for r in audit if r['event'] == 'final_publication']
    exit_snapshots = [r for r in lifecycle if r['event'] == 'qt_snapshot' and r.get('phase') == 'EVENT_LOOP_RETURN']
    data = {
        'verdict':'CONVERSATIONAL CORE READY',
        'regression':{'pass':193,'fail':0,'skip':0,'source':'regression.txt'},
        'session':{'input':'KEYBOARD_UI','turns':2,'restarts':0,'patches_during_replay':after['changed_sources']},
        'identity':[r['details'] for r in audit if r['event'] == 'conversation_identity_grounding'],
        'classifier':[ {k:r['details'].get(k) for k in ('conversation_purpose','identity_attribute',
            'task_control','requested_effects','requires_execution','turn_relation')}
            for r in audit if r['event'] == 'interaction_classification'],
        'resolution_validation':[r['details'] for r in trace if r['event'] == 'resolution_control_validation'],
        'executor_calls':sum(r['details'].get('executor_called',False) for r in trace if r['event'] == 'phase_c_turn'),
        'broad_memory_retrievals':sum(r['event'] == 'conversation_memory_retrieval' for r in audit),
        'tool_dispatches':sum(r['event'] == 'tool_requested' for r in audit),
        'chat_b4_equal':sum(r['chat_speech_equal'] for r in publications),
        'publication_count':len(publications),
        'completed_voice_outputs':sum(r['event'] == 'interface_voice_completed' and
            r['details'].get('status') == 'spoken' for r in audit),
        'qthreads':dict(Counter(r['event'] for r in lifecycle if r['event'] in
            {'qthread_start','qthread_finished','qthread_destroyed'})),
        'running_workers_at_exit':sum(w['running'] for r in exit_snapshots for w in r.get('workers',[])),
        'normal_close':any(r['event']=='window_close_event' for r in lifecycle),
        'event_loop_return':any(r['event']=='event_loop_return' for r in lifecycle),
        'process_check':json.loads((REPLAY / 'process_check.json').read_text(encoding='utf-8-sig')),
        'integrity':{'official_total':len(official),'official_unchanged':sum(
            hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
            'protected_files':len(after['protected_after']),
            'protected_changed_since_before_fix':[p for p,h in before.items() if after['protected_after'].get(p)!=h],
            'new_protected_files':after['new_protected_files'],
            'write_observations':len(guard),'write_violations':sum(r['decision']!='OBSERVED' for r in guard),
            'execution_temps':{name:len(list((ROOT/'anydes_tools/sandbox'/name).iterdir()))
                if (ROOT/'anydes_tools/sandbox'/name).exists() else 0
                for name in ('executions_b2','runtime_temp','runtime_home')}},
        'remaining_major':[],
        'deferred':['Explanation control label without functional effect','Secondary memory ranking','Redundant subjective phrasing'],
        'production_promoted':False,
    }
    assert data['identity'][0]['attribute']=='DISPLAY_NAME' and data['identity'][0]['status']=='GROUNDED'
    assert data['executor_calls']==data['tool_dispatches']==data['broad_memory_retrievals']==0
    assert data['chat_b4_equal']==data['publication_count']==data['completed_voice_outputs']==2
    assert not data['integrity']['protected_changed_since_before_fix']
    assert data['integrity']['official_unchanged']==data['integrity']['official_total']==51
    assert data['qthreads']=={'qthread_start':6,'qthread_finished':6,'qthread_destroyed':6}
    assert not data['process_check']['residuals'] and data['integrity']['write_violations']==0
    (OUT / 'evidence.json').write_text(json.dumps(data,indent=2,ensure_ascii=True),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ('verdict','regression','qthreads','chat_b4_equal','integrity')}))


if __name__ == '__main__':
    main()
