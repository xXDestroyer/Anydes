"""Post-close evidence only; no provider calls or cleanup."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'anydes_tools/logs/planner_resilience'


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
    official = json.loads((ROOT / 'ANYDES_PHASE_F_HASHES.json').read_text())['official_sha256']
    regression = (OUT / 'regression.txt').read_text(encoding='utf-8-sig')
    snapshots = [r for r in life if r['event']=='qt_snapshot']
    pids = {r['pid'] for r in life}
    pids.update(p['pid'] for r in snapshots for p in r['processes'])
    pids.update(p['parent'] for r in snapshots for p in r['processes'] if p['pid']==r['pid'])
    query = ' OR '.join(f'ProcessId = {pid}' for pid in sorted(pids))
    command = (f'Get-CimInstance Win32_Process -Filter "{query}" | '
               'Select-Object ProcessId,Name,CreationDate,ExecutablePath | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],
                            check=True,capture_output=True,text=True)
    publications = [r for r in audit if r['event']=='final_publication']
    value = {
        'recommendation':'CORE STABILIZATION NOT READY',
        'reason':'REAL_UI_B2_NOT_REACHED_NO_EXECUTION_VERIFICATION_RESULT',
        'regression':{'pass':int(re.search(r'Ran (\d+) tests',regression).group(1)),
                      'fail':0,'skip':0,'clean':bool(re.search(r'^OK$',regression,re.M))},
        'sessions':1,'human_turns':sum(r['event']=='phase_c_turn' for r in events),
        'runtime_changes_during_replay':False,
        'provider_policy':[r for r in audit if r['event'].startswith('planner_') or
                           r['event'] in {'semantic_plan_protocol','semantic_proposal_shape'}],
        'turn_outcomes':[r for r in events if r['event']=='phase_c_turn'],
        'resolutions':[r for r in events if r['event']=='resolution_control_validation'],
        'failures':[r for r in events if r['event']=='conversation_tool_failure_state'],
        'pipeline_stages':[r for r in events if r['event']=='pipeline_stage_trace'],
        'publications':publications,
        'chat_b4_consistent':sum(r['details']['chat_speech_equal'] for r in publications),
        'voice_completions':[r for r in audit if r['event']=='interface_voice_completed'],
        'audibility_confirmation':'USER_COULD_NOT_CHECK_THIS_SESSION',
        'lifecycle_counts':dict(Counter(r['event'] for r in life)),
        'last_snapshot':snapshots[-1],
        'remaining_processes':json.loads(result.stdout) if result.stdout.strip() else [],
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
        'chat_b4_consistent','lifecycle_counts','remaining_processes','write_decisions',
        'official_matches','protected_count','protected_changes','source_changes_after_freeze',
        'groq_http_statuses','sandbox_temps','freeze_sha256','stderr')}))


if __name__ == '__main__':
    main()
