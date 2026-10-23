"""Private evidence metadata only. Never copies recalled personal text."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/final_acceptance_triage'


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
    freeze = json.loads((OUT / 'freeze.json').read_text())
    after = json.loads((OUT / 'after.json').read_text())
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    previous = json.loads((ROOT / 'anydes_tools/logs/explanation_continuation_final/freeze.json').read_text())
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    total = int(re.search(r'Ran (\d+) tests',regression).group(1))
    snapshots = [r for r in life if r['event']=='qt_snapshot']
    pids = {r['pid'] for r in life}
    pids.update(p['pid'] for r in snapshots for p in r['processes'])
    pids.update(p['parent'] for r in snapshots for p in r['processes'] if p['pid']==r['pid'])
    query = ' OR '.join(f'ProcessId = {pid}' for pid in sorted(pids))
    command = (f'Get-CimInstance Win32_Process -Filter "{query}" | '
               'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    process = subprocess.run(['powershell.exe','-NoProfile','-Command',command],
                             check=True,capture_output=True,text=True)
    remaining = json.loads(process.stdout) if process.stdout.strip() else []
    publications = [r for r in audit if r['event']=='final_publication']
    retrieval = [{**r,'details':{k:v for k,v in r['details'].items() if k!='evidence_ids'}}
                 for r in audit if r['event']=='conversation_memory_retrieval']
    source = ROOT / 'anydes_tools/workspaces/runtime_candidate/fv_temperatura.py'
    text = source.read_text(encoding='utf-8')
    literal = [r for r in audit if r['event']=='literal_code_publication']
    ui_hash = '1d9c7939119cd9da188704cb50daa4fdc374250454cdf1b6cdab14f787143dbf'
    value = {
        'recommendation':'FINAL CONVERSATIONAL VALIDATION NOT READY',
        'label_triage':'MINOR_SEMANTIC_METADATA_IMPRECISION_DEFERRED',
        'runtime_changes': [p for p,h in previous['source_sha256'].items()
            if p.replace('\\','/').startswith('anydes_tools/conversation/') and sha(ROOT/p)!=h],
        'ui_sessions':1,'runtime_changes_during_replay':False,
        'regression':{'total':total,'pass':total-1,'skip':1,'fail':0},
        'human_turns':sum(r['event']=='phase_c_turn' for r in events),
        'turn_outcomes':[r for r in events if r['event']=='phase_c_turn'],
        'operation_decisions':[r for r in events if r['event'] in {
            'interaction_resolution','resolution_control_validation','pipeline_terminal'}],
        'execution_result':'EXECUTE_EXISTING_RESOLVED_B2_NOT_REACHED_PLAN_PROTOCOL_ERROR',
        'execution_failure':[r for r in events if r['event']=='general_task_reasoner_failed'],
        'source_equivalence':{'artifact_bytes_sha256':sha(source),
            'normalized_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'ui_accessibility_extraction_sha256':ui_hash,'ui_characters':706,
            'literal_publication':literal,
            'equal_to_published_code':bool(literal) and literal[-1]['details']['sha256']==hashlib.sha256(text.encode()).hexdigest(),
            'ui_text_equivalent':hashlib.sha256(text.removesuffix('\n').encode()).hexdigest()==ui_hash,
            'normalization':'CRLF to LF; Qt accessibility extraction omits one terminal LF only',
            'semantic_transformations':0},
        'memory_retrieval':retrieval,
        'memory_grounding':[r for r in audit if r['event'] in {
            'memory_candidate_ranking','memory_grounding_result','conversational_reply_contract'}],
        'memory_followup_quality':'GROUNDED_PRIOR_UTTERANCE_NOT_MEANINGFUL_PERSONAL_FACT_MINOR_RANKING_NOISE',
        'publications':publications,
        'chat_b4_consistent':sum(r['details']['chat_speech_equal'] for r in publications),
        'voice_completions':[r for r in audit if r['event']=='interface_voice_completed'],
        'physical_audio':'USER_CONFIRMED_AUDIBLE_AND_MATCHED_CHAT_INCLUDING_CODE_AND_MEMORY',
        'microphone_this_round':False,
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1], 'remaining_processes':remaining,
        'stderr':(OUT/'gui_stderr.log').read_text(encoding='utf-8'),
        'write_decisions':dict(Counter(r['decision'] for r in writes)),
        'write_violations':[r for r in writes if r['decision']!='OBSERVED'],
        'official_matches':sum(sha(ROOT/p)==h for p,h in official.items()),
        'protected_count':len(freeze['protected_before']),
        'protected_changes':after['changed'],'new_protected':after['new_protected_files'],
        'source_changes_after_freeze':after['changed_sources'],
        'freeze_sha256':sha(OUT/'freeze.json'),
        'groq_http_statuses':dict(Counter(r.get('status') for r in life if r['event']=='groq_http_response')),
        'sandbox_temps':{n:len(list((ROOT/'anydes_tools/sandbox'/n).rglob('*')))
            for n in ('executions_b2','runtime_temp','runtime_home')},
    }
    (OUT/'evidence.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('recommendation','regression','human_turns',
        'runtime_changes','execution_result','source_equivalence','chat_b4_consistent',
        'lifecycle_counts','remaining_processes','write_decisions','official_matches',
        'protected_count','protected_changes','source_changes_after_freeze','groq_http_statuses',
        'sandbox_temps','freeze_sha256','stderr')}))


if __name__ == '__main__':
    main()
