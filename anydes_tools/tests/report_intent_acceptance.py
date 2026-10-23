"""Post-run evidence reduction only. Never imported by the human runtime."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'anydes_tools/logs/intent_acceptance_replay'
CANDIDATE = OUT/'candidate'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    tools = rows(CANDIDATE/'telemetry/tools/conversation-v001_5.jsonl')
    audit = rows(CANDIDATE/'data/audit.log')
    lifecycle = rows(CANDIDATE/'lifecycle.jsonl')
    writes = rows(CANDIDATE/'host_writes.jsonl')
    def details(source, event):
        return [r.get('details', {}) for r in source if r['event'] == event]
    phases = details(tools, 'phase_c_turn')
    decisions = details(tools, 'interaction_resolution')
    validations = details(tools, 'resolution_control_validation')
    publications = details(audit, 'final_publication')
    reads = details(tools, 'artifact_content_read')
    snapshots = [r for r in lifecycle if r['event'] == 'qt_snapshot']
    entries = details(tools, 'recent_artifact_registered')
    files = []
    for entry in entries:
        matches = [p for p in (ROOT/'anydes_tools/workspaces/runtime_candidate').iterdir()
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == entry['hash']]
        files.append({**entry, 'physical_names':[p.name for p in matches],
                      'physical_hash_match':bool(matches)})
    official = json.loads((ROOT/'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    official_matches = sum(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in official.items())
    freeze = json.loads((OUT/'freeze.json').read_text())
    after = json.loads((OUT/'after.json').read_text())
    counts = Counter(r['event'] for r in lifecycle)
    inputs = [r for r in audit if r['event'] == 'user_input_received']
    finals = [r for r in audit if r['event'] == 'final_publication']
    from datetime import datetime
    latencies = [(datetime.fromisoformat(f['time'])-datetime.fromisoformat(i['time'])).total_seconds()
        for i,f in zip(inputs,finals)] if len(inputs) == len(finals) else []
    runtime_failures = details(tools, 'conversation_tool_failure_state')
    summary = {
        'human_turns':len(phases), 'input_sources':dict(Counter(d['details'].get('source','unknown') for d in inputs)),
        'decisions':decisions, 'control_validation':validations, 'phase_c_turns':phases,
        'publications':publications, 'read_evidence':reads, 'artifacts':files,
        'memory_retrieval':details(audit,'conversation_memory_retrieval'),
        'memory_selection':details(audit,'conversational_reply_contract'),
        'failure_explanations':details(tools,'conversation_failure_explained'),
        'failure_records':runtime_failures,
        'cancel':details(tools,'semantic_cancel'),
        'tools_dispatched':details(tools,'conversation_executor_dispatch'),
        'ui_bridge':details(tools,'ui_bridge_outcome'),
        'publication_chat_b4_equal':sum(p['chat_speech_equal'] for p in publications),
        'actual_presentation_requires_qualification':'READ_CODE_MATH_FORMATTING_CHANGED_VISIBLE_CONTENT',
        'audio_completed':sum(d.get('status') == 'completed' for d in details(audit,'interface_voice_trace')),
        'groq_statuses':dict(Counter(r.get('status') for r in lifecycle if r['event']=='groq_http_response')),
        'fallback_requests':counts['fallback_request'],
        'workers':{k:counts[k] for k in ('qthread_start','qthread_finished','qthread_destroyed')},
        'last_snapshot':snapshots[-1] if snapshots else None,
        'max_threads_observed':max((len(s.get('python_threads',[])) for s in snapshots),default=0),
        'ram_initial':snapshots[0].get('working_set') if snapshots else None,
        'ram_peak':max((s.get('peak_working_set') or 0 for s in snapshots),default=0),
        'ram_final':snapshots[-1].get('working_set') if snapshots else None,
        'vram_initial_mb':snapshots[0].get('vram_used_mb') if snapshots else None,
        'vram_peak_mb':max((s.get('vram_used_mb') or 0 for s in snapshots),default=0),
        'vram_final_mb':snapshots[-1].get('vram_used_mb') if snapshots else None,
        'latencies_seconds':latencies,
        'writes':dict(Counter(w['decision'] for w in writes)),
        'production_write_violations':[w for w in writes if w['decision'] != 'OBSERVED'],
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'], 'new_protected_files':after['new_protected_files'],
        'source_changes_during_replay':after['changed_sources'],
        'official_hash_matches':official_matches, 'official_hash_count':len(official),
        'freeze_sha256':hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
        'execution_temp_files':{str(p.relative_to(ROOT)):sum(1 for f in p.rglob('*') if f.is_file())
            for p in (ROOT/'anydes_tools/sandbox/executions_b2',
                      ROOT/'anydes_tools/sandbox/runtime_temp', ROOT/'anydes_tools/sandbox/runtime_home')},
        'rollback_snapshot_files':len(list((OUT/'snapshot').rglob('*.py'))),
    }
    (OUT/'evidence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('human_turns','publication_chat_b4_equal','audio_completed',
        'groq_statuses','fallback_requests','workers','writes','official_hash_matches','official_hash_count',
        'protected_count','protected_changes','new_protected_files','source_changes_during_replay',
        'freeze_sha256','ram_initial','ram_peak','ram_final','vram_peak_mb')}))


if __name__ == '__main__':
    main()
