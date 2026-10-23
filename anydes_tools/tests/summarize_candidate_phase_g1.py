"""Offline evidence collation. Does not import or start the runtime."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/phase_g1/candidate'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def main():
    audit = load_lines(OUT / 'qt_audit.jsonl')
    def events(name):
        return [e['details'] for e in audit if e['event'] == name]
    before = json.loads((OUT / 'before.json').read_text())
    after = json.loads((OUT / 'after.json').read_text())['hashes']
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    freeze = json.loads((ROOT / 'anydes_tools/logs/phase_g/candidate_f/freeze.json').read_text())
    source_changes = {p: {'before': h, 'after': sha(ROOT / p)}
                      for p, h in freeze['source_sha256'].items() if sha(ROOT / p) != h}
    publications = events('final_publication')
    voice = events('interface_voice_trace')
    dirs = list((ROOT / 'anydes_tools').rglob('executions_b2'))
    workspace = ROOT / 'anydes_tools/workspaces/runtime_candidate'
    old_bootstrap = ROOT / 'anydes_tools/logs/phase_g/candidate_f/snapshot/probar_anydes_tools_conversacion_v001_5.py'
    rollback_dir = ROOT / 'anydes_tools/logs/phase_g1/rollback_check'
    rollback_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=rollback_dir) as temp:
        restored = Path(temp) / old_bootstrap.name
        shutil.copy2(old_bootstrap, restored)
        rollback_ok = sha(restored) == freeze['source_sha256'][old_bootstrap.name]
    sources = ['probar_anydes_tools_conversacion_v001_5.py', 'anydes_tools/runtime_environment.py',
               'anydes_tools/tests/test_runtime_environment_g1.py',
               'anydes_tools/tests/validate_candidate_phase_g1.py',
               'anydes_tools/tests/recover_telemetry_phase_g1.py',
               'anydes_tools/tests/summarize_candidate_phase_g1.py']
    hashes = {
        'official_before': official,
        'official_after': {p: sha(ROOT / p) for p in official},
        'official_unchanged': sum(sha(ROOT / p) == h for p, h in official.items()),
        'protected_count': len(before),
        'g1_changed_existing': [p for p, h in before.items() if after.get(p) != h],
        'g1_new_protected_files': sorted(set(after) - set(before)),
        'g1_missing_protected_files': sorted(set(before) - set(after)),
        'candidate_changes_from_g_freeze': source_changes,
        'g1_source_sha256': {p: sha(ROOT / p) for p in sources},
        'human_session_sources_unchanged': all(sha(ROOT / p) == sha(OUT / 'snapshot' / p)
            for p in ('probar_anydes_tools_conversacion_v001_5.py', 'anydes_tools/runtime_environment.py')),
        'restoration': json.loads((ROOT / 'anydes_tools/logs/phase_g1/recovery/restoration.json').read_text()),
        'rollback_bootstrap_fixture_exact': rollback_ok,
    }
    evidence = {
        'classification': 'FULL_HUMAN_RUNTIME; physical microphone corroborated by user',
        'mocks': [], 'failure_injection': False,
        'event_counts': dict(Counter(e['event'] for e in audit)),
        'turns': events('phase_c_turn'),
        'publication': publications,
        'explain_failure': events('conversation_failure_explained'),
        'attempt_history': events('conversation_attempt_outcome'),
        'failures': events('conversation_tool_failure_state'),
        'semantic_protocol': events('semantic_plan_protocol'),
        'oracle_selection': events('oracle_selection'),
        'reasoner_failures': events('general_task_reasoner_failed'),
        'security': events('security_decision'),
        'voice_traces': voice,
        'voice_completion': events('interface_voice_completed'),
        'realtime_latency': events('realtime_response'),
        'input_to_first_text_seconds': [round(t['monotonic'] - i['monotonic'], 3)
            for i, t in zip(events('user_input_received'), events('voice_input_first_text'))],
        'input_sources': events('user_input_received'),
        'worker_threads_seen': sorted({(e['thread'], e['thread_id']) for e in audit}),
        'vram_samples': events('vram_sample'),
        'chat_speech_equal': sum(p['chat_speech_equal'] for p in publications),
        'publication_count': len(publications),
        'provider_http_attempts': None, 'groq_429_count': None, 'fallback_count': None,
        'provider_observability_note': 'Configured hybrid label is not per-request HTTP evidence.',
        'environment_denials': load_lines(OUT / 'environment_violations.jsonl'),
        'denial_attribution': 'UNKNOWN: event has operation only, no target/call stack.',
        'b2_human_execution': [e for e in load_lines(ROOT / 'anydes_tools/logs/sandbox_b2.jsonl')
            if e['request_id'] in {'e75773c0ea574cd08b8d3b0dd21a56cb', '3be3ec4d9af9445898441aaa9e38b7ce'}],
        'b2_log_scope': 'Existing shared staging log, not candidate-private; no B2 source change.',
        'remaining_execution_entries': [str(p.relative_to(ROOT)) for d in dirs for p in d.iterdir()],
        'remaining_workspace_files': [str(p.relative_to(ROOT)) for p in workspace.rglob('*') if p.is_file()],
        'parakeet_retained_media': dict(Counter(p.suffix for p in (OUT / 'telemetry/parakeet').rglob('*') if p.is_file())),
        'retained_media_note': 'Intentional local diagnostic evidence, not zero retained files; no upload.',
        'microphone_user_confirmation': 'Ya hablé y escuché B4',
        'regression': {'isolation_tests': {'pass': 5, 'skip': 0, 'fail': 0},
                       'B1_B2_C_D_E4_F': {'pass': 126, 'skip': 1, 'fail': 0}},
        'result': 'CANDIDATE NOT READY',
    }
    (ROOT / 'ANYDES_PHASE_G1_HASHES.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    (ROOT / 'ANYDES_PHASE_G1_RUNTIME_EVIDENCE.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({k: hashes[k] for k in ('official_unchanged', 'protected_count',
        'g1_changed_existing', 'g1_new_protected_files', 'g1_missing_protected_files',
        'human_session_sources_unchanged', 'rollback_bootstrap_fixture_exact')}))
    print(json.dumps({'publications': len(publications), 'chat_speech_equal': evidence['chat_speech_equal'],
        'execution_residuals': len(evidence['remaining_execution_entries']),
        'workspace_files': len(evidence['remaining_workspace_files'])}))


if __name__ == '__main__':
    main()
