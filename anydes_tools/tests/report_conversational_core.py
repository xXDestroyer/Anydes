"""Summarize closed-run evidence without copying personal context or model prose."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/conversational_core'
REPLAY = OUT / 'replay'
CANDIDATE = REPLAY / 'candidate'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    audit = rows(CANDIDATE / 'data/audit.log')
    lifecycle = rows(CANDIDATE / 'lifecycle.jsonl')
    tools = rows(CANDIDATE / 'telemetry/tools/conversation-v001_5.jsonl')
    guard = rows(CANDIDATE / 'host_writes.jsonl')
    final = [r['details'] for r in audit if r['event'] == 'final_publication']
    after = json.loads((REPLAY / 'after.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    process_check = json.loads((REPLAY / 'process_check.json').read_text(encoding='utf-8-sig'))
    session = [r for r in lifecycle if r.get('pid') == 33120]
    checks = [r for r in session if r['event'] == 'qt_snapshot' and r.get('phase') == 'EVENT_LOOP_RETURN']
    per_turn = []
    for r in tools:
        if r['event'] == 'phase_c_turn':
            detail = r['details']
            per_turn.append({k:detail.get(k) for k in ('turn_id','interaction_kind','task_control',
                'reasoner_called','executor_called','failure_id')})
    identity = [r['details'] for r in audit if r['event'] == 'conversation_identity_grounding']
    result = {
        'verdict':'CONVERSATIONAL CORE NOT READY',
        'regression':{'pass':182,'fail':0,'skip':0,'source':'regression.txt'},
        'run':{'input':'KEYBOARD_UI','turns':len(per_turn),'pid':33120,
            'pre_turn_interruption':'USER_CONFIRMED_CODEX_CREDITS_INTERRUPTION',
            'code_changes_during_replay':after['changed_sources'],
            'normal_close':any(r['event']=='window_close_event' for r in session),
            'event_loop_return':any(r['event']=='event_loop_return' for r in session)},
        'turns':per_turn,
        'identity':identity,
        'failure_context_exit':[r['details'] for r in tools if r['event']=='resolution_context_after'
            and r['details'].get('turn_id')=='8'],
        'publication':{'count':len(final),'chat_b4_equal':sum(r['chat_speech_equal'] for r in final),
            'physical_audio':'USER_CONFIRMED_MATCH',
            'voice_completions':sum(r['event']=='interface_voice_completed' for r in audit)},
        'lifecycle':{'session_counts':dict(Counter(r['event'] for r in session if 'thread_' in r['event'])),
            'all_counts_including_pre_turn_open':dict(Counter(r['event'] for r in lifecycle if 'thread_' in r['event'])),
            'running_workers_at_exit':sum(w['running'] for r in checks for w in r.get('workers',[])),
            'process_check':process_check},
        'integrity':{'official_total':len(official),'official_unchanged':sum(
            hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
            'protected_files':len(after['protected_after']),'protected_changed':after['changed'],
            'new_protected_files':after['new_protected_files'],
            'write_observations':len(guard),'write_violations':sum(r['decision']!='OBSERVED' for r in guard),
            'execution_temp_entries':{name:len(list((ROOT/'anydes_tools/sandbox'/name).iterdir()))
                if (ROOT/'anydes_tools/sandbox'/name).exists() else 0
                for name in ('executions_b2','runtime_temp','runtime_home')}},
        'issues':[
            {'severity':'MAJOR','layer':'PERSONAL_IDENTITY_SEMANTIC_CONTRACT',
             'observed':'PERSONAL_IDENTITY accepted with identity_attribute UNSPECIFIED despite a display-name question',
             'proposed_only':'Require an explicit typed identity attribute for this branch, one semantic repair when absent; retain honest unsupported/no-access handling.'},
            {'severity':'MINOR','layer':'SUBJECTIVE_RESPONSE_PRESENTATION',
             'observed':'Host framing and model phrase can be grammatically redundant; no incorrect effect or memory claim.'},
            {'severity':'DEFER','layer':'PATH_FAILURE_DIAGNOSTICS',
             'observed':'PathDeniedError retained without a detailed safe constraint; explanation honestly declares missing detail.'}
        ]}
    (OUT / 'evidence.json').write_text(json.dumps(result,indent=2,ensure_ascii=True),encoding='utf-8')
    print(json.dumps({'verdict':result['verdict'],'turns':len(per_turn),
        'integrity':result['integrity'],'threads':result['lifecycle']['session_counts']}))


if __name__ == '__main__':
    main()
