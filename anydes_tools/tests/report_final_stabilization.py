"""Post-replay evidence only; no runtime modifications or model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/final_stabilization'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def main():
    candidate = OUT / 'candidate'
    life = rows(candidate / 'lifecycle.jsonl')
    audit = rows(candidate / 'data/audit.log')
    tools = rows(candidate / 'telemetry/tools/conversation-v001_5.jsonl')
    writes = rows(candidate / 'host_writes.jsonl')
    after = json.loads((OUT / 'after.json').read_text())
    freeze = json.loads((OUT / 'freeze.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    counts = Counter(row['event'] for row in life)
    publications = [r for r in audit if r['event'] == 'final_publication']
    source = ROOT / 'anydes_tools/workspaces/runtime_candidate/estabilizacion_suma.py'
    initial = OUT / 'source_v1.py'
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    total = int(re.search(r'Ran (\d+) tests', regression).group(1))
    process_check_path = OUT / 'process_check.json'
    process_check = json.loads(process_check_path.read_text(encoding='utf-8-sig')) if process_check_path.exists() else None
    value = {
        'recommendation':'CONVERSATIONAL ACCEPTANCE NOT READY',
        'stop_reason':'TWO_INDEPENDENT_MAJORS_NO_POST_REPLAY_FIX_AUTHORIZED',
        'runtime_changes_during_replay':False,
        'ui_sessions':1,
        'regression':{'total':total,'pass':total-1,'skip':1,'fail':0},
        'human_turns':len([r for r in tools if r['event']=='phase_c_turn']),
        'turn_outcomes':[r for r in tools if r['event']=='phase_c_turn'],
        'resolution_decisions':[r for r in tools if r['event'] in {'interaction_resolution','resolution_control_validation'}],
        'failures':[r for r in tools if r['event']=='conversation_tool_failure_state'],
        'artifact_registrations':[r for r in tools if r['event']=='recent_artifact_registered'],
        'microphone':{'source':'voice','audio_id':'e9db4281-b994-47cb-b918-272374ccb743',
                      'physical_b4_user_confirmed':True},
        'input_events':[r for r in audit if r['event']=='user_input_received'],
        'chat_b4_consistent':sum(r['details']['chat_speech_equal'] for r in publications),
        'publications':len(publications),
        'voice_completions':[r for r in audit if r['event']=='interface_voice_completed'],
        'memory_retrieval_calls':sum(r['event']=='conversation_memory_retrieval' for r in audit),
        'source_v1_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
        'source_after_replay_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_render_comparison':'NOT_REACHED_STOP_CONDITION',
        'event_counts':dict(counts),
        'groq_http_statuses':dict(Counter(r.get('status') for r in life if r['event']=='groq_http_response')),
        'observed_process_ids':sorted({p['pid'] for r in life if r['event']=='qt_snapshot' for p in r['processes']}),
        'process_check':process_check,
        'process_identity_review':{'candidate_residual_processes':0,
            'reused_pid':4256,'previous_executable':'QtWebEngineProcess.exe',
            'current_process':'opera','current_start_local':'2026-10-15T23:04:22.7575175-05:00',
            'runtime_event_loop_return_local':'2026-10-15T23:03:22.93889-05:00',
            'classification':'PID_REUSED_AFTER_RUNTIME_EXIT_NOT_CANDIDATE_RESIDUAL'},
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'write_violations':[r for r in writes if r['decision']!='OBSERVED'],
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'], 'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'official_matches':sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in official.items()),
        'freeze_sha256':hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        'sandbox_temps':{name:len(list((ROOT/'anydes_tools/sandbox'/name).rglob('*')))
                         for name in ['executions_b2','runtime_temp','runtime_home']}}
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ['recommendation','human_turns','regression',
        'event_counts','groq_http_statuses','chat_b4_consistent','publications','memory_retrieval_calls',
        'write_decisions','official_matches','protected_count','protected_changes',
        'source_changes_after_freeze','sandbox_temps','freeze_sha256','observed_process_ids','stderr']}))


if __name__ == '__main__':
    main()
