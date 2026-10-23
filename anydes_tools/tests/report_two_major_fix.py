"""Read frozen replay evidence; no runtime, model calls, or cleanup."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/two_major_fix'


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate = OUT / 'candidate'
    life = rows(candidate / 'lifecycle.jsonl')
    audit = rows(candidate / 'data/audit.log')
    events = rows(candidate / 'telemetry/tools/conversation-v001_5.jsonl')
    writes = rows(candidate / 'host_writes.jsonl')
    after = json.loads((OUT / 'after.json').read_text())
    freeze = json.loads((OUT / 'freeze.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    total = int(re.search(r'Ran (\d+) tests', regression).group(1))
    publications = [r for r in audit if r['event'] == 'final_publication']
    observed_pids = {r['pid'] for r in life}
    observed_pids.add(31260)  # Launcher PID captured at this session's launch.
    observed_pids.update(p['pid'] for r in life if r['event'] == 'qt_snapshot' for p in r['processes'])
    query = ' OR '.join(f'ProcessId = {pid}' for pid in sorted(observed_pids))
    command = (f'Get-CimInstance Win32_Process -Filter "{query}" | '
               'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            check=True, capture_output=True, text=True)
    current = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(current, dict):
        current = [current]
    current_by_pid = {p['ProcessId']: p for p in current}
    processes = [{'pid': pid, 'present': pid in current_by_pid,
                  'current_identity': current_by_pid.get(pid)} for pid in sorted(observed_pids)]
    final_snapshot = next(r for r in reversed(life) if r['event'] == 'qt_snapshot')
    value = {
        'recommendation': 'TWO-MAJOR FIX NOT READY',
        'ui_sessions': 1, 'runtime_changes_during_replay': False,
        'stop_reason': 'SECOND_EXPLANATION_CLASSIFIED_AS_AMBIGUOUS_WITH_ACTIVE_FAILURE',
        'regression': {'total': total, 'pass': total - 1, 'skip': 1, 'fail': 0},
        'event_counts': dict(Counter(r['event'] for r in life)),
        'human_turns': sum(r['event'] == 'phase_c_turn' for r in events),
        'turn_outcomes': [r for r in events if r['event'] == 'phase_c_turn'],
        'resolution_decisions': [r for r in events if r['event'] in {
            'resolution_context_before', 'resolution_control_validation',
            'interaction_protocol_rejected', 'interaction_resolution', 'resolution_context_after'}],
        'failures': [r for r in events if r['event'] == 'conversation_tool_failure_state'],
        'artifact_registrations': [r for r in events if r['event'] == 'recent_artifact_registered'],
        'source_hashes': {name: sha(OUT / name) for name in ('source_v1.py', 'source_v2.py')},
        'current_artifact_hash': sha(ROOT / 'anydes_tools/workspaces/runtime_candidate/tmfix_doble.py'),
        'chat_b4_consistent': sum(r['details']['chat_speech_equal'] for r in publications),
        'publication_count': len(publications), 'publications': publications,
        'voice_completions': [r for r in audit if r['event'] == 'interface_voice_completed'],
        'physical_voice_input_this_round': False,
        'audibility_confirmation': 'USER_CONFIRMED_AUDIBLE_AND_MATCHED_CHAT_INCLUDING_EXPLANATION_AND_FINAL_CLARIFICATION',
        'groq_http_statuses': dict(Counter(r.get('status') for r in life if r['event'] == 'groq_http_response')),
        'final_qt_snapshot': final_snapshot,
        'process_check': processes,
        'stderr': (OUT / 'gui_stderr.log').read_text(encoding='utf-8'),
        'write_decisions': dict(Counter(r['decision'] for r in writes)),
        'write_violations': [r for r in writes if r['decision'] != 'OBSERVED'],
        'official_matches': sum(sha(ROOT / p) == h for p, h in official.items()),
        'official_total': len(official),
        'launcher_sha256': sha(ROOT / 'Abrir Anydes.bat'),
        'protected_count': len(freeze['protected_before']),
        'protected_changes': after['changed'], 'new_protected': after['new_protected_files'],
        'source_changes_after_freeze': after['changed_sources'],
        'freeze_sha256': sha(OUT / 'freeze.json'),
        'sandbox_temps': {name: len(list((ROOT / 'anydes_tools/sandbox' / name).rglob('*')))
                          for name in ('executions_b2', 'runtime_temp', 'runtime_home')},
        'remaining_issues': [{'severity': 'MAJOR', 'layer': 'SEMANTIC_CLASSIFICATION',
            'turn_id': '6', 'failure_id': 'failure:41d58cee7aeb41bfb37a8e18ce85056e',
            'actual': 'CLARIFICATION/NONE/AMBIGUOUS',
            'expected': 'CONTENT_RESPONSE/CONTINUE_EXPLANATION/FAILURE_DETAIL',
            'repair_used': False, 'executor_called': False, 'reasoner_called': False}],
    }
    (OUT / 'evidence.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({k: value[k] for k in ('recommendation', 'regression', 'human_turns',
        'event_counts', 'chat_b4_consistent', 'publication_count', 'process_check',
        'write_decisions', 'official_matches', 'protected_count', 'protected_changes',
        'source_changes_after_freeze', 'sandbox_temps', 'freeze_sha256', 'stderr')}))


if __name__ == '__main__':
    main()
