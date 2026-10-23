"""Post-session evidence collection only. No model calls, cleanup or runtime edits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/explanation_continuation_final'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = OUT / 'candidate'
    life = rows(data / 'lifecycle.jsonl')
    audit = rows(data / 'data/audit.log')
    events = rows(data / 'telemetry/tools/conversation-v001_5.jsonl')
    writes = rows(data / 'host_writes.jsonl')
    after = json.loads((OUT / 'after.json').read_text())
    freeze = json.loads((OUT / 'freeze.json').read_text())
    previous = json.loads((ROOT / 'anydes_tools/logs/two_major_fix/freeze.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    total = int(re.search(r'Ran (\d+) tests', regression).group(1))
    publications = [r for r in audit if r['event'] == 'final_publication']
    turns = [r for r in events if r['event'] == 'phase_c_turn']
    contexts = [r for r in events if r['event'] == 'resolution_context_after']
    snapshots = [r for r in life if r['event'] == 'qt_snapshot']
    pids = {r['pid'] for r in life}
    pids.update(p['pid'] for r in snapshots for p in r['processes'])
    pids.update(p['parent'] for r in snapshots for p in r['processes'] if p['pid'] == r['pid'])
    query = ' OR '.join(f'ProcessId = {pid}' for pid in sorted(pids))
    command = (f'Get-CimInstance Win32_Process -Filter "{query}" | '
               'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    process_result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                    check=True, capture_output=True, text=True)
    processes = json.loads(process_result.stdout) if process_result.stdout.strip() else []
    controls = [r['details']['task_control'] for r in turns]
    explanation_turns = [r['details'] for r in turns if r['details']['task_control'] in {
        'EXPLAIN_FAILURE','CONTINUE_EXPLANATION','REMEDIATION_QUERY'}]
    value = {
        'ui_sessions': 1, 'runtime_changes_during_replay': False,
        'regression': {'total':total,'pass':total-1,'skip':1,'fail':0,
                       'clean':bool(re.search(r'^OK \(skipped=1\)',regression,re.M))},
        'human_turns':len(turns), 'controls':controls,
        'turn_outcomes':turns,
        'resolution_evidence':[r for r in events if r['event'] in {
            'resolution_context_before','resolution_context_after','resolution_control_validation',
            'interaction_resolution','conversation_tool_failure_state'}],
        'classifier_evidence':[r for r in audit if r['event'] in {
            'interaction_classification','interaction_protocol_rejected'}],
        'same_failure_ids':sorted({r['failure_id'] for r in explanation_turns}),
        'explanations_no_executor_or_reasoner':all(not r['executor_called'] and not r['reasoner_called']
                                                   for r in explanation_turns),
        'post_turn_contexts':contexts,
        'publications':publications,
        'chat_b4_consistent':sum(r['details']['chat_speech_equal'] for r in publications),
        'voice_completions':[r for r in audit if r['event']=='interface_voice_completed'],
        'physical_voice_input_this_round':False,
        'audibility_confirmation':'USER_COULD_NOT_CHECK_THIS_SESSION',
        'recommendation':'FAILURE EXPLANATION CONTINUITY NOT READY',
        'remaining_major':'Depth follow-up used EXPLAIN_FAILURE with unspecified semantic relation, not CONTINUE_EXPLANATION; root and deeper response preserved',
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1],
        'observed_pids':sorted(pids),'remaining_processes':processes,
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'write_violations':[r for r in writes if r['decision']!='OBSERVED'],
        'official_matches':sum(sha(ROOT/p)==h for p,h in official.items()),
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'], 'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'edit_lowerer_unchanged':sha(ROOT/'anydes_tools/conversation/plan_lowering.py') ==
            next(h for p,h in previous['source_sha256'].items() if p.replace('\\','/') ==
                 'anydes_tools/conversation/plan_lowering.py'),
        'launcher_sha256':sha(ROOT/'Abrir Anydes.bat'),
        'freeze_sha256':sha(OUT/'freeze.json'),
        'sandbox_temps':{name:len(list((ROOT/'anydes_tools/sandbox'/name).rglob('*')))
                         for name in ('executions_b2','runtime_temp','runtime_home')},
    }
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('regression','human_turns','controls','same_failure_ids',
        'explanations_no_executor_or_reasoner','chat_b4_consistent','lifecycle_counts',
        'remaining_processes','stderr','write_decisions','official_matches','protected_count',
        'protected_changes','source_changes_after_freeze','edit_lowerer_unchanged',
        'sandbox_temps','freeze_sha256')}))


if __name__ == '__main__':
    main()
